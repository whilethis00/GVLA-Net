# Project GVLA-Net: Handover Documentation for AI Agent

## 1. Project Vision
**GVLA-Net (Geometric Vision-Language-Action Network)** is a novel AI architecture designed to break the "Inference Wall." Current models (LLMs, VLAs) suffer from $O(N)$ inference complexity due to Softmax operations. This project implements an **Orthogonal Geometric Search** that achieves **$O(\log N)$** complexity.

## 2. Core Mathematical Principles
- **Orthogonal Basis:** The model projects high-dimensional latent states onto $k = \log_2 N$ orthogonal hyperplanes.
- **Invariant Observation:** Each projection yields a 1-bit response, forming a unique binary hash for each target (action/word).
- **Gap = 0:** Proven in early experiments (HPW benchmark) that orthogonal questions recover 100% of information with zero redundancy.

## 3. Experimental Legacy (Colab Phase)
The following results must be maintained and used as baselines:
- **Mathematical (HPW):** Demonstrated perfect recovery of hidden parameters via orthogonal probing.
- **Vision (D=512):** Search time reduced from linear to logarithmic.
- **Molecule (Drug Discovery):** Identified targets among 50,000 candidates in 0.009s.
- **LLM Simulation:** Proved 100x+ speedup over Softmax in 50k-vocab settings.

## 4. Target Architecture: GVLA-Net
### Input
- Vision Features (ViT/CLIP)
- Language Features (Prompt Embedding)
### Core Module: Orthogonal Geometry Observer
- **Learnable Orthogonal Basis:** A weight matrix $W \in \mathbb{R}^{k \times d}$ where rows are constrained to be orthogonal.
- **Binarization:** $b = \text{sign}(W \cdot s) > 0$.
### Output
- Discrete Action Index matched via the generated binary hash.

## 5. Server Migration Task List
1. **[Core]** Implement `OrthogonalProjectionLayer` with a custom orthogonality loss.
2. **[Benchmark]** Compare against Flow Matching and Diffusion Policy in terms of FLOPs vs. Accuracy.
3. **[Scaling]** Extend Vocabulary Size ($N$) up to $10^7$ and measure VRAM consumption.
4. **[VLA]** Train on a simulated robotic reaching task with high-resolution action space.

## 6. Coding Standards
- Use **PyTorch** with CUDA acceleration.
- Prioritize **Memory Efficiency** (use chunking for large N).
- Document all geometric transformations mathematically.

# 🚀 PROJECT GVLA-Net: Master Implementation Plan

## 1. Executive Summary
**GVLA-Net (Geometric Vision-Language-Action Network)** is a next-generation inference engine designed to bypass the $O(N)$ Softmax bottleneck in large-scale discrete action/token spaces. By utilizing **Orthogonal Subspace Partitioning**, we achieve **$O(\log N)$** inference latency.

## 2. Theoretical Core (Mandatory Compliance)
AI Agent must implement all modules based on these mathematical foundations:

### A. Orthogonal Projection Layer
Given an input state $s \in \mathbb{R}^d$ and a learnable orthogonal basis $W \in \mathbb{R}^{k \times d}$ where $k = \lceil \log_2 N \rceil$:
- **Projection:** $y = Ws$
- **Binarization (Hashing):** $b = \text{sign}(y) \in \{0, 1\}^k$
- **Optimization Goal:** $W$ must satisfy the orthogonality constraint: $L_{ortho} = \|WW^T - I\|^2_F$.

### B. Geometric Hashing & Matching
Instead of a linear scan or softmax, the target index is retrieved by matching the generated binary code $b$ against a pre-computed/learned codebook of actions $A \in \mathbb{R}^{N \times d}$.

## 3. Project Directory Structure
AI Agent should organize files as follows:
- `/models/gvla_net.py`: Main architecture (ViT Encoder + Geometric Head).
- `/models/layers.py`: Implementation of `OrthogonalProjectionLayer`.
- `/utils/geometry.py`: QR decomposition, orthogonality checks, and basis initialization.
- `/experiments/benchmarks.py`: Comparison scripts (Softmax vs. Flow Matching vs. GVLA-Net).
- `/configs/params.yaml`: Hyperparameters ($N$, $d\_model$, learning rates).

## 4. Implementation Requirements for AI Agent
### I. Memory Management
- When $N > 100,000$, use **tensor chunking** (`torch.chunk`) during the matching phase to avoid CUDA OOM.
- Implement memory-efficient logging to track VRAM usage across different scales of $N$.

### II. Inference Optimization
- Use `torch.cuda.HalfTensor` (FP16) where applicable to maximize throughput.
- Implement a dedicated `inference()` method that skips gradient tracking and uses optimized bitwise operations.

### III. Training Strategy
- Use a **Warm-up period** for the Orthogonal Basis to stabilize the subspace before full VLA training.
- Apply **Straight-Through Estimator (STE)** or soft-step functions for backpropagating through the binarization step.

## 5. Key Benchmarks to Replicate on Server
1. **Scaling Law Test:** Measure latency as $N$ increases from $10^4$ to $10^6$. The curve must follow $\log N$.
2. **Invariance Test:** Verify that $Gap = 0$ is maintained under varying noise levels in the latent space.
3. **VLA Real-time Loop:** Achieve > 100Hz action selection frequency on a single A100/H100 GPU for $N=65,536$.

## 6. Coding Standards
- **Strict Typing:** Use Python type hints for all function signatures.
- **Documentation:** Every class/method must include a docstring explaining the geometric intuition.
- **Logging:** Use `wandb` for experiment tracking.