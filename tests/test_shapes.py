import numpy as np
from src.gpr.model import GPRModel, GPRConfig




def test_gpr_shapes():
X = np.random.randn(40, 3)
y = (X[:,0]*0.3 + 0.1*np.random.randn(40)).reshape(-1,1)
m = GPRModel(n_features=3, n_targets=1, cfg=GPRConfig())
m.fit(X, y, ["f1","f2","f3"], ["t"])
mu, std = m.predict(X[:5])
assert mu.shape == (5,1) and std.shape == (5,1)
