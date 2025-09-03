import numpy as np

def run_online(yolo_flat, uwb_sync, features, gpr_x, gpr_y, x_scaler, y_scaler):
    X_online = x_scaler.transform(features)
    dx_mean, dx_std = gpr_x.predict(X_online, return_std=True)
    dy_mean, dy_std = gpr_y.predict(X_online, return_std=True)
    dx_mean = y_scaler.inverse_transform(np.stack([dx_mean, dy_mean], axis=1))[:,0]
    dy_mean = y_scaler.inverse_transform(np.stack([dx_mean, dy_mean], axis=1))[:,1]
    corrected_x = yolo_flat['flat_x'] + dx_mean
    corrected_y = yolo_flat['flat_y'] + dy_mean
    results = {
        'x': corrected_x,
        'y': corrected_y,
        'cov_x': dx_std**2,
        'cov_y': dy_std**2
    }
    fused_x = 0.5 * corrected_x + 0.5 * uwb_sync['x']
    fused_y = 0.5 * corrected_y + 0.5 * uwb_sync['y']
    results['fused_x'] = fused_x
    results['fused_y'] = fused_y
    return results