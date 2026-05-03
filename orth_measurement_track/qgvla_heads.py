from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ProductQGVLAOutput:
    logits: torch.Tensor
    probs: torch.Tensor


@dataclass
class MPSQGVLAOutput:
    local_tensors: torch.Tensor
    amplitude: torch.Tensor | None = None
    normalization: torch.Tensor | None = None


def code_bits_from_indices(indices: torch.Tensor, num_bits: int) -> torch.Tensor:
    bits = []
    for shift in range(num_bits - 1, -1, -1):
        bits.append(((indices >> shift) & 1).to(torch.float32))
    return torch.stack(bits, dim=-1)


class ProductQGVLAHead(nn.Module):
    """Product-qubit action head with Born-rule loss equal to bit-wise BCE."""

    def __init__(self, input_dim: int, num_bins: int, action_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_bins = num_bins
        self.action_dim = action_dim
        self.num_bits = int(math.ceil(math.log2(num_bins)))
        self.proj = nn.Linear(input_dim, action_dim * self.num_bits)

    def forward(self, latent: torch.Tensor) -> ProductQGVLAOutput:
        logits = self.proj(latent).view(latent.size(0), self.action_dim, self.num_bits)
        probs = torch.sigmoid(logits)
        return ProductQGVLAOutput(logits=logits, probs=probs)

    def born_nll(self, latent: torch.Tensor, target_bits: torch.Tensor) -> torch.Tensor:
        output = self.forward(latent)
        return F.binary_cross_entropy_with_logits(
            output.logits,
            target_bits,
            reduction="none",
        ).sum(dim=(-1, -2)).mean()

    def code_probability(self, latent: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        output = self.forward(latent)
        probs = output.probs
        return torch.prod(
            probs.pow(codes) * (1.0 - probs).pow(1.0 - codes),
            dim=-1,
        )

    def enumerate_code_probabilities(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.action_dim != 1:
            raise ValueError("Enumeration helper currently expects action_dim=1.")
        if self.num_bits > 16:
            raise ValueError("Enumeration is only intended for small synthetic tasks.")
        num_codes = 1 << self.num_bits
        code_ids = torch.arange(num_codes, device=latent.device, dtype=torch.long)
        codes = code_bits_from_indices(code_ids, self.num_bits).to(latent.device)
        probs = self.code_probability(
            latent,
            codes.unsqueeze(0).expand(latent.size(0), -1, -1),
        )
        return code_ids, probs


class MPSQGVLAHead(nn.Module):
    """Entangled Q-GVLA action head using a low-rank MPS amplitude model.

    This implementation is intended for the separate orthogonal-measurement
    research track. It models one action dimension at a time and is best suited
    to synthetic or controlled experiments before BC integration.
    """

    def __init__(self, input_dim: int, num_bins: int, bond_dim: int = 4, hidden_dim: int = 128) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_bins = num_bins
        self.num_bits = int(math.ceil(math.log2(num_bins)))
        self.bond_dim = bond_dim
        self.hidden_dim = hidden_dim

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.local_tensor_head = nn.Linear(
            hidden_dim,
            self.num_bits * 2 * bond_dim * bond_dim,
        )
        self.alpha = nn.Parameter(torch.randn(bond_dim) / math.sqrt(bond_dim))
        self.omega = nn.Parameter(torch.randn(bond_dim) / math.sqrt(bond_dim))

    def forward(self, latent: torch.Tensor) -> MPSQGVLAOutput:
        hidden = self.backbone(latent)
        tensors = self.local_tensor_head(hidden)
        tensors = tensors.view(
            latent.size(0),
            self.num_bits,
            2,
            self.bond_dim,
            self.bond_dim,
        )
        return MPSQGVLAOutput(local_tensors=tensors)

    def amplitude_for_bits(self, local_tensors: torch.Tensor, target_bits: torch.Tensor) -> torch.Tensor:
        batch_size = local_tensors.size(0)
        state = self.alpha.unsqueeze(0).expand(batch_size, -1)
        for bit_idx in range(self.num_bits):
            selected = local_tensors[
                torch.arange(batch_size, device=local_tensors.device),
                bit_idx,
                target_bits[:, bit_idx].long(),
            ]
            state = torch.einsum("bi,bij->bj", state, selected)
        return torch.einsum("bi,i->b", state, self.omega)

    def normalization_constant(self, local_tensors: torch.Tensor) -> torch.Tensor:
        batch_size = local_tensors.size(0)
        alpha_outer = torch.outer(self.alpha, self.alpha)
        state = alpha_outer.unsqueeze(0).expand(batch_size, -1, -1)
        for bit_idx in range(self.num_bits):
            next_state = torch.zeros_like(state)
            for bit_value in (0, 1):
                tensor = local_tensors[:, bit_idx, bit_value]
                next_state = next_state + torch.einsum("bij,bik,bmk->bjm", tensor, state, tensor)
            state = next_state
        omega_outer = torch.outer(self.omega, self.omega).to(local_tensors.device)
        return torch.einsum("bij,ij->b", state, omega_outer)

    def born_nll(self, latent: torch.Tensor, target_bits: torch.Tensor) -> torch.Tensor:
        output = self.forward(latent)
        amplitude = self.amplitude_for_bits(output.local_tensors, target_bits)
        normalization = self.normalization_constant(output.local_tensors).clamp_min(1e-9)
        probability = amplitude.square().clamp_min(1e-9) / normalization
        return -torch.log(probability).mean()

    def enumerate_code_probabilities(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_bits > 16:
            raise ValueError("Enumeration is only intended for small synthetic tasks.")
        output = self.forward(latent)
        normalization = self.normalization_constant(output.local_tensors).clamp_min(1e-9)
        num_codes = 1 << self.num_bits
        code_ids = torch.arange(num_codes, device=latent.device, dtype=torch.long)
        codes = code_bits_from_indices(code_ids, self.num_bits).to(latent.device)
        amplitudes = []
        for code in codes:
            target_bits = code.unsqueeze(0).expand(latent.size(0), -1)
            amplitudes.append(self.amplitude_for_bits(output.local_tensors, target_bits))
        amplitude_tensor = torch.stack(amplitudes, dim=1)
        probabilities = amplitude_tensor.square() / normalization.unsqueeze(1)
        return code_ids, probabilities
