from __future__ import annotations
import numpy as np




def apply_homography(pts_uv: np.ndarray, H: np.ndarray) -> np.ndarray:
"""Project pixel points (u,v) to world plane using homography H (3x3)."""
uv1 = np.c_[pts_uv, np.ones(len(pts_uv))]
XY = (H @ uv1.T).T
XY = XY[:, :2] / XY[:, 2:3]
return XY




def load_H(path_npz: str) -> np.ndarray:
return np.load(path_npz)["H"]




def save_H(H: np.ndarray, path_npz: str):
np.savez_compressed(path_npz, H=H)
