import glob
import os
from huggingface_hub import snapshot_download
import torch
from tqdm import tqdm
from inference.config.model_config import ModelConfig
from safetensors.torch import safe_open


class ModelLoader:
    def _prepare_weights(self, model_name: str):
        hf_folder = snapshot_download(model_name)

        hf_weights_files: list[str] = glob.glob(
            os.path.join(hf_folder, "*.safetensors")
        )

        return hf_folder, hf_weights_files

    def get_weights(self, model_name: str):
        hf_folder, weights_files = self._prepare_weights(model_name)
        model_params: dict[str, torch.Tensor] = {}
        for st_file in tqdm(weights_files):
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():
                    param = f.get_tensor(name)
                    if name == "lm_head.weight":
                        name = "model.lm_head.weight"
                    model_params[name] = param

        return model_params

    def load_model(self, config: ModelConfig, model: torch.nn.Module):
        weights = self.get_weights(config.model_path)

        info = model.load_state_dict(weights)
        print(info)
