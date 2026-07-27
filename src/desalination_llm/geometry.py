from __future__ import annotations

import math
from collections import deque
from typing import Iterable

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree, distance

FEATURE_NAMES = [
    "removed_atoms_count",
    "hull_area",
    "hull_perimeter",
    "circularity",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "aspect_ratio",
    "centroid_offset",
    "radius_mean",
    "radius_std",
    "radius_max",
    "anisotropy",
    "pair_distance_mean",
    "pair_distance_std",
    "removed_bonds",
    "boundary_bonds",
    "boundary_atoms",
    "connected_components",
    "largest_component_fraction",
    "internal_unremoved_count",
]


def build_bond_graph(xy: np.ndarray, cutoff: float = 1.65) -> tuple[list[set[int]], set[tuple[int, int]]]:
    pairs = {tuple(sorted(pair)) for pair in cKDTree(xy).query_pairs(r=cutoff)}
    adjacency = [set() for _ in range(len(xy))]
    for left, right in pairs:
        adjacency[left].add(right)
        adjacency[right].add(left)
    return adjacency, pairs


def edge_mask(xy: np.ndarray, margin: float) -> np.ndarray:
    low = xy.min(axis=0) + margin
    high = xy.max(axis=0) - margin
    return np.any((xy < low) | (xy > high), axis=1)


def nearest_candidates(
    xy: np.ndarray,
    removed_indices: Iterable[int],
    k: int,
    edge_margin: float = 2.5,
) -> list[int]:
    removed = set(int(value) for value in removed_indices)
    excluded_edge = edge_mask(xy, edge_margin)
    available = np.asarray(
        [idx for idx in range(len(xy)) if idx not in removed and not excluded_edge[idx]],
        dtype=int,
    )
    if not len(available):
        return []
    if not removed:
        center = xy.mean(axis=0)
        scores = np.linalg.norm(xy[available] - center, axis=1)
    else:
        removed_xy = xy[sorted(removed)]
        tree = cKDTree(removed_xy)
        scores, _ = tree.query(xy[available], k=1)
    order = np.argsort(scores, kind="stable")
    return available[order[: min(k, len(order))]].tolist()


def colonize_internal_atoms(
    xy: np.ndarray,
    current_removed: Iterable[int],
    proposed_removed: Iterable[int],
    max_new_atoms: int,
) -> set[int]:
    current = set(int(value) for value in current_removed)
    proposed = set(int(value) for value in proposed_removed)
    if len(proposed) < 3 or max_new_atoms <= 0:
        return proposed
    remaining_capacity = max_new_atoms - len(proposed - current)
    if remaining_capacity <= 0:
        return proposed
    try:
        hull = ConvexHull(xy[sorted(proposed)], qhull_options="QJ")
    except QhullError:
        return proposed
    unremoved = np.asarray([idx for idx in range(len(xy)) if idx not in proposed], dtype=int)
    if not len(unremoved):
        return proposed
    equations = hull.equations
    inside = np.all(xy[unremoved] @ equations[:, :2].T + equations[:, 2] <= 1e-9, axis=1)
    internal = unremoved[inside]
    if not len(internal):
        return proposed
    centroid = xy[sorted(proposed)].mean(axis=0)
    ranked = internal[np.argsort(np.linalg.norm(xy[internal] - centroid, axis=1))]
    proposed.update(ranked[:remaining_capacity].tolist())
    return proposed


def geometry_features(
    xy: np.ndarray,
    removed_indices: Iterable[int],
    adjacency: list[set[int]] | None = None,
    bond_pairs: set[tuple[int, int]] | None = None,
) -> dict[str, float]:
    removed = sorted(set(int(value) for value in removed_indices))
    n_removed = len(removed)
    values = {name: 0.0 for name in FEATURE_NAMES}
    values["removed_atoms_count"] = float(n_removed)
    if n_removed == 0:
        values["aspect_ratio"] = 1.0
        values["anisotropy"] = 1.0
        return values

    points = xy[removed]
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    width, height = (upper - lower).tolist()
    values["bbox_width"] = float(width)
    values["bbox_height"] = float(height)
    values["bbox_area"] = float(width * height)
    if n_removed < 2 or min(width, height) < 1e-8:
        values["aspect_ratio"] = 1.0
    else:
        values["aspect_ratio"] = float(min(100.0, max(width, height) / min(width, height)))

    centroid = points.mean(axis=0)
    sheet_center = xy.mean(axis=0)
    values["centroid_offset"] = float(np.linalg.norm(centroid - sheet_center))
    radii = np.linalg.norm(points - centroid, axis=1)
    values["radius_mean"] = float(radii.mean())
    values["radius_std"] = float(radii.std())
    values["radius_max"] = float(radii.max())

    if n_removed >= 2:
        covariance = np.cov(points.T)
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
        if eigenvalues[0] <= 1e-10:
            values["anisotropy"] = 100.0
        else:
            values["anisotropy"] = float(min(100.0, eigenvalues[-1] / eigenvalues[0]))
        pair_distances = distance.pdist(points)
        values["pair_distance_mean"] = float(pair_distances.mean())
        values["pair_distance_std"] = float(pair_distances.std())
    else:
        values["anisotropy"] = 1.0

    if adjacency is None or bond_pairs is None:
        adjacency, bond_pairs = build_bond_graph(xy)
    removed_set = set(removed)
    values["removed_bonds"] = float(
        sum(1 for left, right in bond_pairs if left in removed_set and right in removed_set)
    )
    boundary_edges = [
        (left, right)
        for left, right in bond_pairs
        if (left in removed_set) != (right in removed_set)
    ]
    values["boundary_bonds"] = float(len(boundary_edges))
    values["boundary_atoms"] = float(
        len({left if left in removed_set else right for left, right in boundary_edges})
    )

    visited: set[int] = set()
    component_sizes: list[int] = []
    for start in removed:
        if start in visited:
            continue
        queue = deque([start])
        visited.add(start)
        size = 0
        while queue:
            current = queue.popleft()
            size += 1
            for neighbor in adjacency[current]:
                if neighbor in removed_set and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        component_sizes.append(size)
    values["connected_components"] = float(len(component_sizes))
    values["largest_component_fraction"] = float(max(component_sizes) / n_removed)

    if n_removed >= 3:
        try:
            hull = ConvexHull(points, qhull_options="QJ")
            area = float(hull.volume)
            perimeter = float(hull.area)
            values["hull_area"] = area
            values["hull_perimeter"] = perimeter
            values["circularity"] = float(4.0 * math.pi * area / (perimeter**2 + 1e-12))
            unremoved = np.asarray([idx for idx in range(len(xy)) if idx not in removed_set], dtype=int)
            if len(unremoved):
                inside = np.all(
                    xy[unremoved] @ hull.equations[:, :2].T + hull.equations[:, 2] <= 1e-9,
                    axis=1,
                )
                values["internal_unremoved_count"] = float(inside.sum())
        except QhullError:
            pass
    return values
