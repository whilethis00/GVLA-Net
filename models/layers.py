import math
from typing import Dict

import torch
from torch import nn

from utils.geometry import initialize_orthogonal_basis


class OrthogonalProjectionLayer(nn.Module):
    """Project latent states onto orthogonal hyperplanes and emit a binary hash.

    GVLA-Net replaces an ``O(N)`` softmax scan with ``k = ceil(log2(N))``
    orthogonal geometric questions. Each learnable row of ``W`` defines one
    hyperplane in the latent space; the sign of the projection answers a 1-bit
    question, and the concatenation of those answers forms the logarithmic-time
    routing pattern used by the future inference engine.
    """

    def __init__(
        self,
        input_dim: int,
        num_codes: int,
        *,
        basis_size: int | None = None,
        use_ste: bool = True,
        eps: float = 1e-6,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Create the learnable orthogonal observer.

        Args:
            input_dim: Latent dimensionality ``d``.
            num_codes: Target discrete search space size ``N``.
            basis_size: Optional override for ``k``. Defaults to ``ceil(log2(N))``.
            use_ste: Whether to use a straight-through estimator for hashing.
            eps: Numerical stabilizer used by the soft surrogate path.
            device: Optional device placement for the learnable basis.
            dtype: Optional dtype for the learnable basis.
        """
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"'input_dim' must be positive, got {input_dim}.")
        if num_codes <= 1:
            raise ValueError(f"'num_codes' must be greater than 1, got {num_codes}.")

        inferred_basis_size = int(math.ceil(math.log2(num_codes)))
        self.input_dim = input_dim
        self.num_codes = num_codes
        self.basis_size = basis_size if basis_size is not None else inferred_basis_size
        self.use_ste = use_ste
        self.eps = eps

        if self.basis_size <= 0:
            raise ValueError(f"'basis_size' must be positive, got {self.basis_size}.")
        if self.basis_size > self.input_dim:
            raise ValueError(
                "OrthogonalProjectionLayer requires basis_size <= input_dim "
                f"to maintain row-wise orthogonality, got {self.basis_size} > {self.input_dim}."
            )

        basis = initialize_orthogonal_basis(
            d=self.input_dim,
            k=self.basis_size,
            device=device,
            dtype=dtype,
        )
        self.weight = nn.Parameter(basis)

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Project a latent state and produce its logarithmic binary routing code.

        The matrix multiplication ``state @ W^T`` costs ``O(d * log N)`` because
        the observer only asks ``log2(N)`` orthogonal questions instead of scoring
        every class. During training, the straight-through path preserves the hard
        binary geometry at the forward pass while leaving a surrogate gradient for
        optimization through the hash boundary.

        Args:
            state: Input tensor with shape ``(..., d)``.

        Returns:
            A dictionary containing:
            - ``projections``: Continuous hyperplane responses with shape ``(..., k)``.
            - ``binary_code``: Hash in ``{0, 1}`` with shape ``(..., k)``.
            - ``signed_code``: Hash in ``{-1, 1}`` with shape ``(..., k)``.
        """
        if state.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input last dimension {self.input_dim}, got {state.size(-1)}."
            )

        projections = state @ self.weight.transpose(0, 1)
        signed_hard = torch.where(
            projections >= 0,
            torch.ones_like(projections),
            -torch.ones_like(projections),
        )

        if self.use_ste:
            scale = projections.abs().detach().clamp_min(self.eps)
            signed_soft = projections / scale
            signed_code = signed_hard + (signed_soft - signed_soft.detach())
        else:
            signed_code = torch.tanh(projections)

        binary_code = 0.5 * (signed_code + 1.0)
        return {
            "projections": projections,
            "binary_code": binary_code,
            "signed_code": signed_code,
        }

    def orthogonality_loss(self) -> torch.Tensor:
        """Return the mandatory orthogonality penalty ``||WW^T - I||_F^2``.

        This loss keeps the observer's hyperplanes close to mutually orthogonal
        directions, which preserves the handover document's central promise:
        each bit should encode new information rather than a redundant view of an
        already-asked question.
        """
        gram_matrix = self.weight @ self.weight.transpose(0, 1)
        identity = torch.eye(
            self.basis_size, device=self.weight.device, dtype=self.weight.dtype
        )
        deviation = gram_matrix - identity
        return torch.sum(deviation * deviation)
