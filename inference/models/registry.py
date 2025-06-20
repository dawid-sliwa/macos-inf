from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Set, Union
import torch as nn


@dataclass
class _ModelRegistry:
    # Keyed by model_arch
    models: dict[str, nn.Module] = field(default_factory=dict)

    def get_supported_archs(self) -> Set[str]:
        return self.models.keys()

    def register_model(
        self,
        model_arch: str,
        model: type[nn.Module],
    ) -> None:
        if not isinstance(model_arch, str):
            msg = f"`model_arch` should be a string, not a {type(model_arch)}"
            raise TypeError(msg)

        self.models[model_arch] = model

    def _raise_for_unsupported(self, architectures: list[str]):
        all_supported_archs = self.get_supported_archs()

        if any(arch in all_supported_archs for arch in architectures):
            raise ValueError(
                f"Model architectures {architectures} failed "
                "to be inspected. Please check the logs for more details."
            )

        raise ValueError(
            f"Model architectures {architectures} are not supported for now. "
            f"Supported architectures: {all_supported_archs}"
        )

    def resolve_model_cls(
        self,
        architectures: Union[str, list[str]],
    ) -> tuple[type[nn.Module], str]:
        architectures = self._normalize_archs(architectures)

        for arch in architectures:
            model_cls = self._try_load_model_cls(arch)
            if model_cls is not None:
                return (model_cls, arch)

        return self._raise_for_unsupported(architectures)

    def is_text_generation_model(
        self,
        architectures: Union[str, list[str]],
    ) -> bool:
        model_cls, _ = self.inspect_model_cls(architectures)
        return model_cls.is_text_generation_model
