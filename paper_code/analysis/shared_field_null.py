"""Pure field/null functions copied from the current shared-field analysis."""
import numpy as np
import math,itertools
from collections import defaultdict

def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if int(keep.sum()) < 3:
        return float("nan")
    x = x[keep] - float(np.mean(x[keep]))
    y = y[keep] - float(np.mean(y[keep]))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(x @ y / denom) if denom > 0 else float("nan")

def _group_indices(shafts: list[str], mode: str) -> list[np.ndarray]:
    if mode == "channel":
        return [np.arange(len(shafts), dtype=int)]
    if mode != "within_shaft":
        raise ValueError(mode)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, shaft in enumerate(shafts):
        grouped[str(shaft)].append(index)
    return [np.asarray(indices, dtype=int) for indices in grouped.values()]

def _permutation_indices(
    shafts: list[str], mode: str, n_perm: int, rng: np.random.Generator
) -> tuple[np.ndarray, int, bool]:
    groups = _group_indices(shafts, mode)
    n_unique = math.prod(math.factorial(len(group)) for group in groups)
    n_contacts = len(shafts)
    if n_unique <= n_perm:
        group_permutations = [list(itertools.permutations(group.tolist())) for group in groups]
        result = np.tile(np.arange(n_contacts, dtype=int), (n_unique, 1))
        for row_index, combination in enumerate(itertools.product(*group_permutations)):
            for target, source in zip(groups, combination):
                result[row_index, target] = np.asarray(source, dtype=int)
        return result, int(n_unique), True

    result = np.tile(np.arange(n_contacts, dtype=int), (n_perm, 1))
    for group in groups:
        if len(group) < 2:
            continue
        orders = np.argsort(rng.random((n_perm, len(group))), axis=1)
        result[:, group] = group[orders]
    return result, int(n_unique), False

def _gaussian_kernel(points: np.ndarray, sigma: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    distance2 = ((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-distance2 / (2.0 * float(sigma) ** 2))

def _null_correlations(
    field_a: np.ndarray,
    earliness_b: np.ndarray,
    support_b: np.ndarray,
    points_b: np.ndarray,
    sigma_b: float,
    permutation_indices: np.ndarray,
    *,
    chunk_size: int = 500,
) -> np.ndarray:
    kernel = _gaussian_kernel(points_b, sigma_b)
    centered_a = np.asarray(field_a, dtype=float) - float(np.mean(field_a))
    norm_a = float(np.linalg.norm(centered_a))
    output = np.full(len(permutation_indices), np.nan, dtype=float)
    for start in range(0, len(permutation_indices), chunk_size):
        stop = min(start + chunk_size, len(permutation_indices))
        permutations = permutation_indices[start:stop]
        support = support_b[permutations]
        values = earliness_b[permutations]
        weighted_support = support * values
        numerator = np.einsum("jk,bk->bj", kernel, weighted_support, optimize=True)
        denominator = np.einsum("jk,bk->bj", kernel, support, optimize=True)
        fields = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 0,
        )
        centered_b = fields - np.nanmean(fields, axis=1, keepdims=True)
        norm_b = np.linalg.norm(centered_b, axis=1)
        valid = np.isfinite(centered_b).all(axis=1) & (norm_b > 0) & (norm_a > 0)
        output[start:stop][valid] = centered_b[valid] @ centered_a / (norm_b[valid] * norm_a)
    return output[np.isfinite(output)]
