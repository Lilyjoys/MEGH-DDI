import torch
from torch import nn
import torch.nn.functional as F


class AdaptiveMarginSigmoidLoss(nn.Module):
    def __init__(self, base_temp=1.0, margin=0.3, beta=0.5, adaptive_temp=True):
        super().__init__()
        self.base_temp = base_temp
        self.margin = margin
        self.beta = beta
        self.adaptive_temp = adaptive_temp

    def forward(self, p_scores, n_scores):
        if self.adaptive_temp:
            std = torch.std(torch.cat([p_scores, n_scores]))
            temperature = self.base_temp * (1 + 0.5 * std.detach())
        else:
            temperature = self.base_temp
        weights = F.softmax(n_scores / temperature, dim=-1).detach()
        n_scores = (weights * n_scores).sum(dim=-1) if n_scores.ndim > 1 else n_scores

        p_loss = -torch.log(torch.sigmoid((p_scores - self.margin) / temperature)).mean()
        n_loss = -torch.log(torch.sigmoid(-(n_scores + self.margin) / temperature)).mean()

        total_loss = self.beta * p_loss + (1 - self.beta) * n_loss

        return total_loss, p_loss.detach(), n_loss.detach()
