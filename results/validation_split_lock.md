# Validation Split Lock

This file freezes the offline validation split for BC checkpoint metric export.

## Dataset

- source file: `data/robomimic/lift/ph/low_dim_v141.hdf5`
- task family: `RoboMimic Lift PH low-dimensional demos`

## Default Split Rule

- split method: deterministic shuffled holdout
- split seed: `20260503`
- validation fraction: `0.10`

## Intent

Every compared checkpoint must be evaluated on the exact same held-out indices.

Do not regenerate the split with a different seed or fraction unless the artifact is explicitly versioned as a different protocol.
