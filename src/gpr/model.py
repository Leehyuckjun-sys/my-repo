from __future__ import annotations


class GPRModel:
"""Multi-output wrapper: 1 regressor per target column."""
def __init__(self, n_features: int, n_targets: int, cfg: Optional[GPRConfig] = None):
self.cfg = cfg or GPRConfig()
self.n_features = int(n_features)
self.n_targets = int(max(1, n_targets))
self.x_scaler = StandardScaler()
self.models: List[GaussianProcessRegressor] = []
base_kernel = _build_kernel(n_features, self.cfg.kernel, self.cfg.length_scale, self.cfg.ard, self.cfg.noise, self.cfg.matern_nu)
for _ in range(self.n_targets):
self.models.append(
GaussianProcessRegressor(
kernel=base_kernel,
alpha=float(self.cfg.alpha),
normalize_y=True,
random_state=self.cfg.random_state,
)
)
self.feature_names: Optional[List[str]] = None
self.target_names: Optional[List[str]] = None


def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], target_names: List[str]):
if y.ndim == 1:
y = y.reshape(-1, 1)
self.feature_names, self.target_names = list(feature_names), list(target_names)
Xs = self.x_scaler.fit_transform(X)
for i, mdl in enumerate(self.models):
mdl.fit(Xs, y[:, i])
return self


def predict(self, X: np.ndarray, return_std: bool = True):
Xs = self.x_scaler.transform(X)
means = []
stds = []
for mdl in self.models:
mu, std = mdl.predict(Xs, return_std=True)
means.append(mu)
stds.append(std)
Y = np.vstack(means).T # (N, T)
S = np.vstack(stds).T # (N, T)
if not return_std:
return Y
return Y, S


def save(self, path: str):
joblib.dump({
"cfg": self.cfg,
"n_features": self.n_features,
"n_targets": self.n_targets,
"x_scaler": self.x_scaler,
"models": self.models,
"feature_names": self.feature_names,
"target_names": self.target_names,
}, path)


@staticmethod
def load(path: str) -> "GPRModel":
d = joblib.load(path)
obj = GPRModel(d["n_features"], d["n_targets"], d["cfg"])
obj.x_scaler = d["x_scaler"]
obj.models = d["models"]
obj.feature_names = d.get("feature_names")
obj.target_names = d.get("target_names")
return obj
