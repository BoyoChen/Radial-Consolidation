"""Minimal NumPy reference implementation, not the original training engine.

Each row contains one receiver's learnable incoming coefficients. The caller
supplies a raw local update; this module does not implement STDP traces,
spiking dynamics, weight floors, masking, or homeostatic adaptation.
"""
import numpy as np


def reference_direction(weights):
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2 or not np.isfinite(weights).all():
        raise ValueError('Expected a finite receiver-by-coefficient matrix')
    return weights / np.maximum(np.linalg.norm(weights, axis=1, keepdims=True), 1e-12)


def constrain_update(update, reference, alpha=0.0):
    """Algorithm 1 projection; zero references retain only alpha * update."""
    update = np.asarray(update, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if update.shape != reference.shape or not np.isfinite(update).all():
        raise ValueError('Update and reference must be matching finite matrices')
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('alpha must be in [0, 1]')
    q = reference_direction(reference)
    parallel = q * np.sum(q * update, axis=1, keepdims=True)
    return parallel + alpha * (update - parallel)


def normalize_l1(weights, target=1.0):
    """Ideal fixed-target normalization used in the cancellation argument."""
    weights = np.asarray(weights, dtype=float)
    return target * weights / np.maximum(np.abs(weights).sum(1, keepdims=True), 1e-12)
