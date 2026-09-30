"""ArcFace classification loss for 512-dimensional face embeddings."""

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """Add an angular margin to target-class logits and return cross-entropy.

    Args:
        num_classes: Number of identities/classes to classify.
    """

    embedding_size = 512
    scale = 64.0
    margin = 0.5

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.empty(num_classes, self.embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self._cos_m = math.cos(self.margin)
        self._sin_m = math.sin(self.margin)
        self._threshold = math.cos(math.pi - self.margin)
        self._margin_adjustment = math.sin(math.pi - self.margin) * self.margin

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        """Compute ArcFace cross-entropy loss.

        Args:
            embeddings: Feature tensor with shape ``[batch_size, 512]``.
            labels: Target class indices with shape ``[batch_size]``.
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_size:
            raise ValueError(
                "embeddings must have shape [batch_size, 512], "
                f"got {tuple(embeddings.shape)}"
            )
        if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
            raise ValueError(
                "labels must have shape [batch_size] matching embeddings"
            )
        if labels.dtype != torch.long:
            raise TypeError("labels must have dtype torch.long")

        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - cosine.square(), min=0.0))
        phi = cosine * self._cos_m - sine * self._sin_m
        phi = torch.where(cosine > self._threshold, phi, cosine - self._margin_adjustment)

        target_logits = phi.gather(1, labels.unsqueeze(1))
        logits = cosine.scatter(1, labels.unsqueeze(1), target_logits) * self.scale
        return F.cross_entropy(logits, labels)
