import torch
import torch.nn as nn
import torch.nn.functional as F

# Multi-class focal loss with optional per-class alpha (vector) or scalar alpha.
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        if isinstance(alpha, (list, tuple)):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, (float, int)):
            alpha = torch.tensor(float(alpha), dtype=torch.float32)

        self.register_buffer('alpha', alpha if alpha is not None else None, persistent=False)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)         
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()                                 

        if self.alpha is None:
            alpha_t = 1.0
        else:
            alpha = self.alpha
            if alpha.ndim == 0:
                alpha_t = alpha
            else:
                alpha_t = alpha.to(logits.device, logits.dtype)[targets]

        focal_factor = (1.0 - pt).pow(self.gamma)
        loss = - alpha_t * focal_factor * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

