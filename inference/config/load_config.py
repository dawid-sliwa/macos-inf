from dataclasses import dataclass
from typing import Optional


@dataclass
class LoadConfig:
    download_dir: Optional[str] = None