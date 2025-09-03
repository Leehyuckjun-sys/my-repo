from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path


NUMERIC_DTYPES = ["int16","int32","int64","float16","float32","float64"]




def load_csv(path: str | Path) -> pd.DataFrame:
df = pd.read_csv(path)
if "t_ms" in df.columns:
df = df.sort_values("t_ms").reset_index(drop=True)
return df




def sync_and_resample(df: pd.DataFrame, time_col: str, hz: int | None) -> pd.DataFrame:
if hz in (None, 0):
return df
if time_col not in df.columns:
raise ValueError(f"time_col '{time_col}' not in columns")
# 균일 그리드 생성
t0, t1 = int(df[time_col].min()), int(df[time_col].max())
step = int(1000 / hz) # ms
grid = np.arange(t0, t1 + step, step, dtype=int)
num_cols = df.select_dtypes(include=NUMERIC_DTYPES).columns
# 중복 제거 후 보간
gdf = (
df.drop_duplicates(time_col)
.set_index(time_col)
.reindex(grid)
.interpolate(limit_direction="both")
.reset_index()
.rename(columns={"index": time_col})
)
# 비수치 컬럼은 forward fill
for c in df.columns:
if c not in gdf.columns:
gdf[c] = df[c]
if c not in num_cols and c != time_col:
gdf[c] = gdf[c].fillna(method="ffill")
return gdf




def select_xy(df: pd.DataFrame, feature_cols: list[str], target_cols: list[str] | None):
X = df[feature_cols].to_numpy(dtype=float)
y = None if not target_cols else df[target_cols].to_numpy(dtype=float)
return X, y




def train_val_split(df: pd.DataFrame, val_ratio: float = 0.2, seed: int = 42):
n = len(df)
n_val = int(n * val_ratio)
rng = np.random.default_rng(seed)
idx = rng.permutation(n)
val_idx = idx[:n_val]
tr_idx = idx[n_val:]
return df.iloc[tr_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)
