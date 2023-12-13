import torch
import torch.nn as nn
from torch import autograd as autograd
from torch import Tensor


class SoftParameterMasker(autograd.Function):
    """Dynamic STE (straight-through estimator) parameter masker"""

    @staticmethod
    def forward(ctx, weight: torch.Tensor, mask: torch.Tensor):
        return weight * mask

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


class SoftParameterMaskerV2(autograd.Function):
    """Dynamic STE (straight-through estimator) parameter masker"""

    @staticmethod
    def forward(ctx, weight: torch.Tensor, mask: torch.Tensor):
        ctx.save_for_backward(mask)
        return weight * mask

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        mask, = ctx.saved_tensors
        return grad_output * torch.clamp(mask, min=1.), None


class HardParameterMasker(autograd.Function):
    """Hard parameter masker"""

    @staticmethod
    def forward(ctx, weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(mask)
        return weight * mask

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        mask, = ctx.saved_tensors
        return grad_output * mask, None


class SoftConv2D(nn.Conv2d):
    def forward(self, input: Tensor, mask: Tensor) -> Tensor:
        weight = SoftParameterMasker.apply(self.weight, mask)
        return self._conv_forward(input, weight, self.bias)


class SoftConv2DV2(nn.Conv2d):
    def forward(self, input: Tensor, mask: Tensor) -> Tensor:
        weight = SoftParameterMaskerV2.apply(self.weight, mask)
        return self._conv_forward(input, weight, self.bias)


class HardConv2D(nn.Conv2d):
    def forward(self, input: Tensor, mask: Tensor) -> Tensor:
        weight = HardParameterMasker.apply(self.weight, mask)
        return self._conv_forward(input, weight, self.bias)
