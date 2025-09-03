import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

def train_and_validate_gpr(features, labels, n_splits=3):
    x_scaler = StandardScaler()
    X = x_scaler.fit_transform(features)
    y_scaler = StandardScaler()
    Y = y_scaler.fit_transform(labels)
    kernel = RBF(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2,1e2)) + WhiteKernel()
    gpr_x = GaussianProcessRegressor(kernel=kernel, alpha=1e-4)
    gpr_y = GaussianProcessRegressor(kernel=kernel, alpha=1e-4)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = {'mae_x':[], 'mae_y':[], 'rmse_x':[], 'rmse_y':[]}
    for train_idx, test_idx in kf.split(X):
        gpr_x.fit(X[train_idx], Y[train_idx,0])
        gpr_y.fit(X[train_idx], Y[train_idx,1])
        pred_x = gpr_x.predict(X[test_idx])
        pred_y = gpr_y.predict(X[test_idx])
        mae_x = np.mean(np.abs(pred_x - Y[test_idx,0]))
        mae_y = np.mean(np.abs(pred_y - Y[test_idx,1]))
        rmse_x = np.sqrt(np.mean((pred_x - Y[test_idx,0])**2))
        rmse_y = np.sqrt(np.mean((pred_y - Y[test_idx,1])**2))
        metrics['mae_x'].append(mae_x)
        metrics['mae_y'].append(mae_y)
        metrics['rmse_x'].append(rmse_x)
        metrics['rmse_y'].append(rmse_y)
    gpr_x.fit(X, Y[:,0])
    gpr_y.fit(X, Y[:,1])
    return (gpr_x, gpr_y, x_scaler, y_scaler, metrics)