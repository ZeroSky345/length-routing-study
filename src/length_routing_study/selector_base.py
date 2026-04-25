from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class SelectionResult:
    mask: Optional[torch.Tensor]
    sparsity: float
    layout: Optional[Dict[str, Any]] = None


class BaseSelector(ABC):
    name: str

    @abstractmethod
    def select(self, q: torch.Tensor, k: torch.Tensor) -> SelectionResult:
        raise NotImplementedError
