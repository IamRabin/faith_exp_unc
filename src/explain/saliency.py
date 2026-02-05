# src/explain/saliency.py
from __future__ import annotations

import torch


def select_logits_for_labels(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Select z_y(x) for each sample in a batch.

    Args:
        logits: Tensor of shape (B, C)
        y: LongTensor of shape (B,) with values in {0,...,C-1}

    Returns:
        z_y: Tensor of shape (B,)
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape (B,C), got {tuple(logits.shape)}")
    if y.ndim != 1:
        raise ValueError(f"y must have shape (B,), got {tuple(y.shape)}")
    if logits.shape[0] != y.shape[0]:
        raise ValueError(f"Batch mismatch: logits B={logits.shape[0]} vs y B={y.shape[0]}")

    # gather along class dimension
    return logits.gather(dim=1, index=y.view(-1, 1)).squeeze(1)


def input_gradient_of_selected_logit(
    logits: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    create_graph: bool,
    retain_graph: bool | None = None,
) -> torch.Tensor:
    """
    Compute saliency map g(x,y) = ∇_x z_y(x), batched.

    Args:
        logits: model(x), shape (B, C)
        x: input tensor, shape (B, ...) (e.g. (B,3,32,32))
        y: labels, shape (B,)
        create_graph: True during training (enables backprop through g to theta)
        retain_graph: set True if you will call autograd multiple times on same graph

    Returns:
        g: Tensor same shape as x
    """
    z_y = select_logits_for_labels(logits, y)  # (B,)

    # Important: autograd.grad requires a scalar output, so sum over batch.
    scalar = z_y.sum()
    
    if retain_graph is None:
        retain_graph = create_graph

    g = torch.autograd.grad(
        outputs=scalar,
        inputs=x,
        create_graph=create_graph,
        retain_graph=retain_graph,
        only_inputs=True,
        allow_unused=False,
    )[0]
    return g


def compute_saliency(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    create_graph: bool,
) -> torch.Tensor:
    """
    Convenience wrapper: compute g(x,y) from model directly.

    Notes:
      - Ensures x.requires_grad = True.
      - Returns g with same shape as x.
    """
    if not x.requires_grad:
        # we clone/detach to avoid modifying upstream tensors accidentally
        x = x.detach().clone()
        x.requires_grad_(True)

    logits = model(x)  # (B,C)
    g = input_gradient_of_selected_logit(logits, x, y, create_graph=create_graph)
    return g
