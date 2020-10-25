import torch
import os
from abc import ABC, abstractmethod

class BaseKDOp(ABC):

    @abstractmethod
    def apply(self, net:torch.nn.Module, inplace:bool = True) -> torch.nn.Module:
        pass

