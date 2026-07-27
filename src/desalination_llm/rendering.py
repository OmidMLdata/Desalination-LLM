from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull, QhullError


def render_pore(
    xy: np.ndarray,
    removed_indices: Iterable[int],
    output: str | Path,
    show_hull: bool = False,
    dpi: int = 140,
) -> Path:
    removed = sorted(set(int(value) for value in removed_indices))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(xy[:, 0], xy[:, 1], s=7, c="black", linewidths=0)
    if removed:
        axis.scatter(xy[removed, 0], xy[removed, 1], s=34, c="white", linewidths=0)
    if show_hull and len(removed) >= 3:
        try:
            hull = ConvexHull(xy[removed], qhull_options="QJ")
            boundary = xy[removed][hull.vertices]
            boundary = np.vstack([boundary, boundary[0]])
            axis.plot(boundary[:, 0], boundary[:, 1], linewidth=1.2)
        except QhullError:
            pass
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")
    figure.tight_layout(pad=0)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(figure)
    return output_path
