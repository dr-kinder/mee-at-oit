"""
Distortion field analysis and visualization tools.

Load, filter, evaluate, compare, and visualize the polynomial distortion
fields produced by compute_distortion. Intended for notebook analysis outside
the pipeline, not as a pipeline stage.

Entry points:
    load_distortion_field(result_dir)  → DistortionField
    load_star_residuals(result_dir)    → DataFrame

DistortionField supports:
    .evaluate(x, y)                    → (dx, dy) at arbitrary points
    .select(order_min=3)               → field with only high-order terms
    .grid(nx=50)                       → (x, y, dx, dy) on a regular grid
    field_a - field_b                  → difference DistortionField
"""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mee2024.distortion_polynomial import get_basis


# ---------------------------------------------------------------------------
# Legacy coefficient name → canonical [px, py] conversion
# ---------------------------------------------------------------------------

def _parse_coeff_key(key: str) -> str:
    """Convert any coefficient key format to canonical '[px, py]' string."""
    if re.match(r'^\[\d+, \d+\]$', key):
        return key
    if key == '1':
        return '[0, 0]'
    xm = re.search(r'\bx(?:\^(\d+))?', key)
    ym = re.search(r'\by(?:\^(\d+))?', key)
    px = int(xm.group(1) or '1') if xm else 0
    py = int(ym.group(1) or '1') if ym else 0
    return f'[{px}, {py}]'


# ---------------------------------------------------------------------------
# DistortionField
# ---------------------------------------------------------------------------

@dataclass
class DistortionField:
    """
    A polynomial distortion field loaded from a compute_distortion output.

    coeff_x, coeff_y : full OLS parameter vectors; index 0 is the constant
        (intercept), remaining indices are polynomial terms in the same order
        as names[1:].
    names : canonical '[px, py]' strings, length == len(coeff_x).
    basis_type : 'polynomial' or 'legendre'.
    order : 'quintic', 'cubic', etc.
    w : normalization factor = max(img_shape)/2, used when evaluating the
        polynomial basis. Must match the value used when the field was fitted.
    img_shape : [height, width] in pixels; used to set accurate grid bounds.
        If None a square ±w grid is used.
    source : descriptive label (typically the result directory path).
    """
    coeff_x: np.ndarray
    coeff_y: np.ndarray
    names: list[str]
    basis_type: str
    order: str
    w: float
    img_shape: list[int] | None = None
    source: str = ''

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the distortion at pixel positions (x, y).

        x, y are measured from the image centre (same convention as the
        pipeline: x = px - img_width/2, y = py - img_height/2).

        Returns (dx, dy) arrays in pixels, same shape as x and y.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        scalar = x.ndim == 0
        x, y = np.atleast_1d(x), np.atleast_1d(y)
        opts = {'distortionOrder': self.order, 'basis_type': self.basis_type}
        basis = get_basis(y, x, self.w, opts)   # (n_pts, n_poly_terms)
        dx = basis @ self.coeff_x[1:] + self.coeff_x[0]
        dy = basis @ self.coeff_y[1:] + self.coeff_y[0]
        if scalar:
            return float(dx[0]), float(dy[0])
        return dx, dy

    # ------------------------------------------------------------------
    # Term selection / filtering
    # ------------------------------------------------------------------

    def select(self, *, terms: list[str] | None = None,
               order_min: int | None = None,
               order_max: int | None = None) -> DistortionField:
        """
        Return a new DistortionField with only the selected terms nonzero.

        terms     : explicit list of '[px, py]' names to keep
        order_min : keep terms where px+py >= order_min
        order_max : keep terms where px+py <= order_max

        If terms is given it overrides the range filters. The constant
        '[0, 0]' term is treated like any other: include it explicitly or
        via order_min=0 / order_max=0.
        """
        mask = np.zeros(len(self.names), dtype=bool)
        for i, name in enumerate(self.names):
            if terms is not None:
                mask[i] = name in terms
            else:
                px, py = map(int, re.findall(r'\d+', name))
                degree = px + py
                lo = True if order_min is None else degree >= order_min
                hi = True if order_max is None else degree <= order_max
                mask[i] = lo and hi
        return replace(self,
                       coeff_x=np.where(mask, self.coeff_x, 0.0),
                       coeff_y=np.where(mask, self.coeff_y, 0.0))

    # ------------------------------------------------------------------
    # Grid convenience
    # ------------------------------------------------------------------

    def grid(self, nx: int = 50,
             ny: int | None = None) -> tuple[np.ndarray, np.ndarray,
                                              np.ndarray, np.ndarray]:
        """
        Evaluate on a regular grid covering the image sensor area.

        Returns (x, y, dx, dy) where x and y are 2-D coordinate arrays
        measured from the image centre, and dx, dy are the distortion maps.

        If img_shape is known, the grid spans the actual sensor dimensions.
        Otherwise it uses a symmetric ±w square. ny defaults to nx scaled
        by the sensor aspect ratio.
        """
        if self.img_shape is not None:
            half_y = self.img_shape[0] / 2
            half_x = self.img_shape[1] / 2
        else:
            half_x = self.w
            half_y = self.w
        if ny is None:
            ny = max(1, round(nx * half_y / half_x))
        x1d = np.linspace(-half_x, half_x, nx)
        y1d = np.linspace(-half_y, half_y, ny)
        x, y = np.meshgrid(x1d, y1d)
        dx, dy = self.evaluate(x.ravel(), y.ravel())
        return x, y, dx.reshape(x.shape), dy.reshape(y.shape)

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __sub__(self, other: DistortionField) -> DistortionField:
        if self.basis_type != other.basis_type:
            raise ValueError(
                f'Cannot subtract fields with different basis_type: '
                f'{self.basis_type!r} vs {other.basis_type!r}')
        if self.order != other.order:
            raise ValueError(
                f'Cannot subtract fields with different order: '
                f'{self.order!r} vs {other.order!r}')
        if self.names != other.names:
            raise ValueError(
                'Cannot subtract fields with different coefficient names')
        label = f'({Path(self.source).name}) − ({Path(other.source).name})'
        return replace(self,
                       coeff_x=self.coeff_x - other.coeff_x,
                       coeff_y=self.coeff_y - other.coeff_y,
                       source=label)

    def __repr__(self) -> str:
        return (f'DistortionField(order={self.order!r}, '
                f'basis={self.basis_type!r}, '
                f'w={self.w:.0f}, '
                f'n_terms={len(self.names)}, '
                f'source={Path(self.source).name!r})')


# ---------------------------------------------------------------------------
# File-finding helpers
# ---------------------------------------------------------------------------

def _find_distortion_json(result_dir: Path) -> Path:
    hits = sorted(result_dir.glob('DISTORTION_OUTPUT*/distortion/distortion_results.txt'))
    if not hits:
        raise FileNotFoundError(
            f'No distortion_results.txt found under {result_dir}\n'
            'Expected DISTORTION_OUTPUT*/distortion/distortion_results.txt')
    return hits[-1]


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _img_shape_from_centroids_dir(result_dir: Path) -> list[int] | None:
    hits = sorted(result_dir.glob('CENTROID_OUTPUT*/data/results.txt'))
    if hits:
        return _read_json(hits[-1]).get('img_shape')
    return None


def _img_shape_from_centroids_zip(parent: Path) -> list[int] | None:
    for czip in sorted(parent.glob('*_centroids.zip')):
        try:
            z = zipfile.ZipFile(czip)
            for key in ('results.txt', 'data/results.txt'):
                if key in z.namelist():
                    return json.load(z.open(key)).get('img_shape')
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_distortion_field(result_dir) -> DistortionField:
    """
    Load a DistortionField from a compute_distortion output.

    result_dir can be:
      - a directory containing DISTORTION_OUTPUT*/distortion/ (Kaggle download)
      - a .zip file whose root contains distortion_results.txt (local output)

    Old-format coefficient names ('1', 'x', 'x^2', 'x * y', …) are
    converted automatically to canonical '[px, py]' strings.
    """
    p = Path(result_dir)

    if p.suffix == '.zip':
        z = zipfile.ZipFile(p)
        nl = z.namelist()
        key = ('distortion_results.txt'
               if 'distortion_results.txt' in nl
               else 'distortion/distortion_results.txt')
        d = json.loads(z.read(key))
        img_shape = d.get('img_shape') or _img_shape_from_centroids_zip(p.parent)
    else:
        dist_json = _find_distortion_json(p)
        d = _read_json(dist_json)
        img_shape = (d.get('img_shape')
                     or _img_shape_from_centroids_dir(p)
                     or _img_shape_from_centroids_zip(p))

    if img_shape is None:
        raise FileNotFoundError(
            f'Cannot determine img_shape for {p}.\n'
            'Run compute_distortion again (img_shape is now saved in new outputs), '
            'or ensure centroids output is in the same directory.')

    raw_x = d['distortion coeffs x']
    raw_y = d['distortion coeffs y']
    names = [_parse_coeff_key(k) for k in raw_x]
    coeff_x = np.array(list(raw_x.values()))
    coeff_y = np.array(list(raw_y.values()))

    return DistortionField(
        coeff_x=coeff_x,
        coeff_y=coeff_y,
        names=names,
        basis_type=d.get('basis_type', 'polynomial'),
        order=d['distortion order'],
        w=max(img_shape) / 2,
        img_shape=img_shape,
        source=str(p),
    )


def load_star_residuals(result_dir) -> pd.DataFrame:
    """
    Load CATALOGUE_MATCHED_ERRORS.csv from a compute_distortion output.

    Returns a DataFrame with all star positions, catalogue coordinates,
    residuals, and quality flags. px, py are in raw image pixels (origin
    at corner). To get centred coordinates for distortion evaluation:
        x = df['px'] - img_shape[1] / 2
        y = df['py'] - img_shape[0] / 2
    """
    p = Path(result_dir)
    if p.suffix == '.zip':
        z = zipfile.ZipFile(p)
        nl = z.namelist()
        key = ('CATALOGUE_MATCHED_ERRORS.csv'
               if 'CATALOGUE_MATCHED_ERRORS.csv' in nl
               else 'distortion/CATALOGUE_MATCHED_ERRORS.csv')
        return pd.read_csv(z.open(key), index_col=0)

    hits = sorted(p.glob('DISTORTION_OUTPUT*/distortion/CATALOGUE_MATCHED_ERRORS.csv'))
    if not hits:
        raise FileNotFoundError(f'No CATALOGUE_MATCHED_ERRORS.csv found under {p}')
    return pd.read_csv(hits[-1], index_col=0)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_distortion_field(ax, field: DistortionField, component: str = 'magnitude',
                          nx: int = 40, cmap: str = 'RdBu_r', **kwargs):
    """
    Plot the distortion field as a filled colour map on ax.

    component : 'magnitude' | 'x' | 'y'
    Returns the QuadMesh (pcolormesh) for colorbar attachment.
    """
    x, y, dx, dy = field.grid(nx=nx)
    if component == 'magnitude':
        z = np.hypot(dx, dy)
        label = '|distortion| (px)'
        default_cmap = 'viridis'
    elif component == 'x':
        z = dx
        label = 'dx (px)'
        default_cmap = 'RdBu_r'
    else:
        z = dy
        label = 'dy (px)'
        default_cmap = 'RdBu_r'

    cmap = cmap if cmap != 'RdBu_r' or component == 'magnitude' else default_cmap
    vabs = np.percentile(np.abs(z), 98)
    vmax = kwargs.pop('vmax', vabs)
    vmin = kwargs.pop('vmin', 0 if component == 'magnitude' else -vabs)

    pcm = ax.pcolormesh(x, y, z, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    ax.set_aspect('equal')
    ax.set_xlabel('x (px from centre)')
    ax.set_ylabel('y (px from centre)')
    ax.set_title(f'{label} — {Path(field.source).name}')
    return pcm


def plot_distortion_quiver(ax, field: DistortionField, nx: int = 20, **kwargs):
    """Plot distortion field as a quiver (arrow) map on ax."""
    x, y, dx, dy = field.grid(nx=nx)
    q = ax.quiver(x, y, dx, dy, **kwargs)
    ax.set_aspect('equal')
    ax.set_xlabel('x (px from centre)')
    ax.set_ylabel('y (px from centre)')
    ax.set_title(f'distortion — {Path(field.source).name}')
    return q


def plot_distortion_summary(field: DistortionField, nx: int = 40) -> plt.Figure:
    """
    Four-panel summary: dx map, dy map, magnitude map, and quiver.
    Returns the Figure.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    x, y, dx, dy = field.grid(nx=nx)
    mag = np.hypot(dx, dy)
    vabs = np.percentile(np.abs(np.stack([dx, dy])), 98)

    for ax, z, title, cmap, vmin, vmax in [
        (axs[0, 0], dx,  'dx (px)',           'RdBu_r', -vabs,  vabs),
        (axs[0, 1], dy,  'dy (px)',           'RdBu_r', -vabs,  vabs),
        (axs[1, 0], mag, '|distortion| (px)', 'viridis',  0,    None),
    ]:
        pcm = ax.pcolormesh(x, y, z, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(pcm, ax=ax, label='pixels')
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xlabel('x (px)')
        ax.set_ylabel('y (px)')

    axs[1, 1].quiver(x, y, dx, dy)
    axs[1, 1].set_aspect('equal')
    axs[1, 1].set_title('quiver')
    axs[1, 1].set_xlabel('x (px)')
    axs[1, 1].set_ylabel('y (px)')

    fig.suptitle(f'Distortion field — {Path(field.source).name}', fontsize=12)
    fig.tight_layout()
    return fig
