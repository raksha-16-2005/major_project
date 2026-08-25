"""Hinton (2015) knowledge distillation loss with multi-class and multi-label branches."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HintonKDLoss(nn.Module):
    """L = alpha * CE(student, hard) + (1-alpha) * T^2 * KL(softmax(student/T) || softmax(teacher/T))

    For multi-label tasks (e.g. ChestMNIST), CE -> BCE-with-logits and KL is replaced by
    per-element BCE between the temperature-softened sigmoid distributions.
    """

    def __init__(self, temperature: float = 2.0, alpha: float = 0.2, multilabel: bool = False):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.multilabel = multilabel

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                hard_labels: torch.Tensor) -> torch.Tensor:
        if self.multilabel:
            return self._forward_multilabel(student_logits, teacher_logits, hard_labels)
        return self._forward_multiclass(student_logits, teacher_logits, hard_labels)

    def _forward_multiclass(self, s, t, y):
        ce = F.cross_entropy(s, y)
        T = self.T
        log_p_s = F.log_softmax(s / T, dim=-1)
        p_t = F.softmax(t / T, dim=-1)
        kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
        return self.alpha * ce + (1.0 - self.alpha) * kl

    def _forward_multilabel(self, s, t, y):
        bce = F.binary_cross_entropy_with_logits(s, y)
        T = self.T
        # Pseudo-soft targets: temperature-softened sigmoid of teacher logits.
        # KL between Bernoullis collapses to a constant + BCE between distributions.
        p_t = torch.sigmoid(t / T).detach()
        soft = F.binary_cross_entropy_with_logits(s / T, p_t) * (T * T)
        return self.alpha * bce + (1.0 - self.alpha) * soft
