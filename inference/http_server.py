from contextlib import asynccontextmanager
import json
import time
import traceback
import uuid
from fastapi import FastAPI, APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

from inference.config.model_config import ModelConfig
from inference.engine.async_llm import AsyncLLM
from inference.openai_protocol import ChatCompletionRequest
from inference.server_args import ServerArgs
import logging

router = APIRouter()

logging.basicConfig(level=logging.INFO)


CHAT_PAGE = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Chat</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  html,body{height:100%}body{margin:0;font:15px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:#0b1020;color:#e7ebff;display:grid;place-items:center;padding:24px}
  .app{width:min(820px,100%);display:grid;grid-template-rows:1fr auto;gap:12px;border:1px solid rgba(255,255,255,.08);border-radius:14px;overflow:hidden;background:#121a34}
  .chat{padding:16px;height:65vh;overflow:auto;background:radial-gradient(800px 200px at 50% 0,rgba(255,255,255,.06),transparent)}
  .msg{display:flex;gap:10px;margin:8px 0}
  .role{flex:0 0 auto;width:26px;height:26px;border-radius:8px;display:grid;place-items:center;font-size:12px;font-weight:700}
  .role.user{background:#2e3a78}.role.assistant{background:#2a6e46}.role.system{background:#6b3a2e}
  .bubble{background:#1b2550;border:1px solid rgba(255,255,255,.08);padding:8px 10px;border-radius:12px;white-space:pre-wrap;max-width:100%}
  .composer{display:flex;gap:8px;padding:12px;background:#0f1734;border-top:1px solid rgba(255,255,255,.06)}
  textarea{flex:1;min-height:44px;max-height:160px;resize:vertical;background:#0e1631;border:1px solid #24305c;color:#e7ebff;border-radius:12px;padding:10px}
  button{background:#7aa2ff;color:#081125;border:0;padding:10px 14px;border-radius:12px;font-weight:700;cursor:pointer}
  button.secondary{background:transparent;color:#e7ebff;border:1px solid #3a477d}
</style>
</head>
<body>
  <div class="app">
    <div id="chat" class="chat" aria-live="polite"></div>
    <form id="composer" class="composer">
      <button type="button" id="clear" class="secondary">Clear</button>
      <textarea id="prompt" placeholder="Type a message…" required></textarea>
      <button id="send" type="submit">Send</button>
      <button id="stop" type="button" class="secondary" disabled>Stop</button>
    </form>
  </div>
<script>
  const LS_KEY="chat_messages_v1";
  const chat=document.getElementById("chat");
  const form=document.getElementById("composer");
  const promptEl=document.getElementById("prompt");
  const clearBtn=document.getElementById("clear");
  const sendBtn=document.getElementById("send");
  const stopBtn=document.getElementById("stop");
  let messages=load();
  let controller=null;

  render();

  clearBtn.onclick=()=>{ if(confirm("Clear conversation?")){ messages=[]; save(); render(); } };
  form.onsubmit=(e)=>{ e.preventDefault(); const t=promptEl.value.trim(); if(!t) return; promptEl.value=""; send(t); };
  promptEl.onkeydown=(e)=>{ if(e.key==="Enter" && !e.shiftKey){ e.preventDefault(); form.requestSubmit(); } };
  stopBtn.onclick=()=>{ if(controller) controller.abort(); };

  function load(){ try{ const raw=localStorage.getItem(LS_KEY); return raw?JSON.parse(raw):[] }catch{ return [] } }
  function save(){ localStorage.setItem(LS_KEY, JSON.stringify(messages)) }
  function render(){ chat.innerHTML=""; messages.forEach((m,i)=>chat.appendChild(renderMsg(m))); chat.scrollTop=chat.scrollHeight }
  function renderLast(){ const nodes=chat.querySelectorAll(".msg"); const i=messages.length-1; if(nodes.length===messages.length){ chat.replaceChild(renderMsg(messages[i]), nodes[i]) } else { render() } chat.scrollTop=chat.scrollHeight }
  function renderMsg(m){ const d=document.createElement("div"); d.className="msg"; const r=document.createElement("div"); r.className="role "+m.role; r.textContent=m.role[0].toUpperCase(); const b=document.createElement("div"); b.className="bubble"; b.textContent=m.content||""; d.appendChild(r); d.appendChild(b); return d }

  async function send(userText){
    messages.push({role:"user",content:userText});
    const aiIndex=messages.push({role:"assistant",content:""})-1;
    save(); render();

    const body={ model:"test", messages:messages, stream:true };
    controller=new AbortController();
    setBusy(true);
    try{
      const res=await fetch("/v1/chat/completions",{ method:"POST", headers:{ "Content-Type":"application/json" }, body:JSON.stringify(body), signal:controller.signal });
      if(!res.ok||!res.body) throw new Error("HTTP "+res.status+" "+res.statusText);
      const reader=res.body.getReader();
      const decoder=new TextDecoder("utf-8");
      let buffer="", doneFlag=false;
      while(true){
        const {value,done}=await reader.read();
        if(done) break;
        buffer+=decoder.decode(value,{stream:true});
        const parts=buffer.split("\\n\\n"); buffer=parts.pop();
        for(const part of parts){
          const lines=part.split("\\n");
          for(const line of lines){
            if(!line.startsWith("data:")) continue;
            const s=line.slice(5).trim();
            if(s==="[DONE]"){ doneFlag=true; break; }
            try{
              const evt=JSON.parse(s);
              const delta=evt && evt.choices && evt.choices[0] && evt.choices[0].delta ? evt.choices[0].delta : {};
              if(delta.content){ messages[aiIndex].content+=delta.content; save(); renderLast(); }
            }catch(e){}
          }
          if(doneFlag) break;
        }
        if(doneFlag) break;
      }
    }catch(err){
      if(err.name!=="AbortError"){ messages.push({role:"system",content:"Error: "+err.message}); save(); render(); }
    }finally{
      setBusy(false); controller=null; save(); renderLast();
    }
  }

  function setBusy(b){ sendBtn.disabled=b; stopBtn.disabled=!b; promptEl.disabled=b; sendBtn.textContent=b?"Sending…":"Send" }
  if(messages.length===0){ messages.push({role:"system",content:"New chat started."}); save(); render(); }
</script>
</body>
</html>'''



def get_engine(request: Request) -> AsyncLLM:
    return request.app.state.engine


@router.get("/health")
async def health_route():
    return {"status": "ok"}


@router.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=CHAT_PAGE)


async def stream_response(engine: AsyncLLM, data: ChatCompletionRequest):
    request_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())
    yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'test', 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
    response = engine.chat_cmpl_continous_batching(request=data)

    async for token in response:
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": "test",
            "choices": [
                {"index": 0, "delta": {"content": token}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(chunk)} \n\n"

    final_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": "test",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


async def non_stream_response(engine: AsyncLLM, data: ChatCompletionRequest):
    request_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())
    choices = {}
    choices[0] = {
        "index": 0,
        "message": {"role": "assistant", "content": ""},
        "finish_reason": None,
    }
    data = engine.chat_cmpl_continous_batching(request=data)

    async for token in data:
        choices[0]["message"]["content"] += token

    choices[0]["finish_reason"] = "stop"

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": "test",
        "choices": list(choices.values()),
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@router.post("/v1/chat/completions")
async def create_chat_completion(data: ChatCompletionRequest, request: Request):
    engine = get_engine(request)
    try:
        if data.stream:
            return StreamingResponse(
                stream_response(engine=engine, data=data),
                media_type="text/event-stream",
            )
        else:
            return await non_stream_response(engine=engine, data=data)
    except Exception:
        traceback.print_exc()


def build_app(args: ServerArgs):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config = ModelConfig(model_path=args.model_path)

        engine = AsyncLLM(model_config=config)
        app.state.engine = engine
        engine.start()
        try:
            yield
        finally:
            engine.stop()

    app = FastAPI(title="Macos inf", lifespan=lifespan)

    app.include_router(router=router)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.server_args = args

    return app


def server_start(args: ServerArgs):
    app = build_app(args=args)

    uvicorn.run(app=app, host=args.host, port=args.port, loop="uvloop")
