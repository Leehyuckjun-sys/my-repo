from __future__ import annotations
if time_col not in df.columns:
raise ValueError(f"time_col '{time_col}' not in columns: {list(df.columns)}")


step = int(1000 / hz) # ms
t0, t1 = int(df[time_col].min()), int(df[time_col].max())
grid = np.arange(t0, t1 + step, step, dtype=int)


num_cols = df.select_dtypes(include=_NUM_DTYPES).columns
gdf = (
df.drop_duplicates(time_col)
.set_index(time_col)
.reindex(grid)
.interpolate(limit_direction="both")
.reset_index()
.rename(columns={"index": time_col})
)


# Forward fill non-numeric columns
for c in df.columns:
if c not in gdf.columns:
gdf[c] = df[c]
if c not in num_cols and c != time_col:
gdf[c] = gdf[c].fillna(method="ffill")
return gdf




def ensure_columns(df: pd.DataFrame, cols: List[str]):
missing = [c for c in cols if c not in df.columns]
if missing:
raise KeyError(f"Missing required columns: {missing}. Found: {list(df.columns)[:20]}...")




def select_xy(df: pd.DataFrame, feature_cols: List[str], target_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
ensure_columns(df, feature_cols)
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
