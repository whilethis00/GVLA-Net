import torch


def initialize_orthogonal_basis(
    d: int,
    k: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Initialize a row-wise orthonormal basis for the geometric observer.

    In GVLA-Net, each row of ``W in R^{k x d}`` defines one orthogonal hyperplane
    that asks an independent 1-bit geometric question about the latent state.
    Using QR decomposition makes those questions mutually non-redundant at
    initialization, matching the handover document's requirement that the model
    extract ``k = log2(N)`` bits without overlap.

    Args:
        d: Ambient latent dimension.
        k: Number of orthogonal basis vectors / geometric questions.
        device: Optional device for the returned tensor.
        dtype: Optional dtype for the returned tensor.

    Returns:
        A tensor of shape ``(k, d)`` whose rows are orthonormal.

    Raises:
        ValueError: If ``d`` or ``k`` is non-positive, or if ``k > d`` so a
            row-wise orthonormal basis cannot exist in ``R^d``.
    """
    if d <= 0:
        raise ValueError(f"'d' must be positive, got {d}.")
    if k <= 0:
        raise ValueError(f"'k' must be positive, got {k}.")
    if k > d:
        raise ValueError(
            f"Cannot initialize {k} orthogonal basis vectors in R^{d}; require k <= d."
        )

    target_dtype = dtype if dtype is not None else torch.get_default_dtype()

    # QR factorization is not implemented for CUDA half precision on older
    # PyTorch builds, so we construct the orthogonal frame in float32/float64
    # and cast only after the geometry is established.
    qr_dtype = target_dtype
    if device is not None and torch.device(device).type == "cuda":
        if target_dtype in (torch.float16, torch.bfloat16):
            qr_dtype = torch.float32

    gaussian_matrix = torch.randn((d, k), device=device, dtype=qr_dtype)
    q_matrix, _ = torch.linalg.qr(gaussian_matrix, mode="reduced")
    basis = q_matrix.transpose(0, 1).contiguous()
    if basis.dtype != target_dtype:
        basis = basis.to(dtype=target_dtype)
    return basis


def check_orthogonality(matrix: torch.Tensor) -> float:
    """Measure how far the current geometric basis drifts from perfect orthogonality.

    The handover document defines the observer weights as a matrix ``W`` whose
    rows should remain orthogonal so that each binary decision contributes new
    information. This function computes ``||WW^T - I||_F``: zero means the rows
    still form an ideal orthonormal family, while larger values quantify how much
    redundancy or geometric distortion has entered the projection layer.

    Args:
        matrix: A 2D weight matrix ``W`` with shape ``(k, d)``.

    Returns:
        The Frobenius norm of ``WW^T - I`` as a Python float.

    Raises:
        ValueError: If ``matrix`` is not 2-dimensional.
    """
    if matrix.ndim != 2:
        raise ValueError(
            f"'matrix' must be 2-dimensional with shape (k, d), got ndim={matrix.ndim}."
        )

    row_count = matrix.size(0)
    identity = torch.eye(row_count, device=matrix.device, dtype=matrix.dtype)
    gram_matrix = matrix @ matrix.transpose(0, 1)
    deviation = gram_matrix - identity
    return torch.linalg.matrix_norm(deviation, ord="fro").item()
