"""Base class enforcing the initialize()/process() contract for every module."""
from abc import ABC, abstractmethod
from typing import Any


class PipelineModule(ABC):
    name: str = "module"

    def __init__(self) -> None:
        self._ready = False

    @abstractmethod
    def initialize(self) -> None:
        """Load model weights and any auxiliary resources into memory."""

    @abstractmethod
    def process(self, input: Any) -> Any:
        """Run inference on a single input and return the module's output."""

    def _require_ready(self) -> None:
        if not self._ready:
            raise RuntimeError(
                f"{self.name}: initialize() must be called before process()"
            )
