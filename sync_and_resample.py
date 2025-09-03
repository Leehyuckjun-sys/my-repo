import pandas as pd
import numpy as np

def load_data(yolo_file, uwb_file, anchor_file):
    yolo = pd.read_csv(yolo_file)
    uwb = pd.read_csv(uwb_file)
    anchors = pd.read_csv(anchor_file)
    return yolo, uwb, anchors

def fit_time_sync(uwb_time, cam_time):
    b = cam_time.iloc[0] - uwb_time.iloc[0]
    a = 1.0
    t_uwb_sync = uwb_time * a + b
    return t_uwb_sync

def time_sync_and_resample(yolo_df, uwb_df, target_freq=50):
    t_uwb_sync = fit_time_sync(uwb_df['time'], yolo_df['time'])
    uwb_df['time_sync'] = t_uwb_sync

    t_start = max(yolo_df['time'].min(), uwb_df['time_sync'].min())
    t_end = min(yolo_df['time'].max(), uwb_df['time_sync'].max())
    grid_times = np.arange(t_start, t_end, 1/target_freq)

    yolo_interp = yolo_df.set_index('time').reindex(grid_times).interpolate().reset_index().rename(columns={'index':'time'})
    uwb_interp = uwb_df.set_index('time_sync').reindex(grid_times).interpolate().reset_index().rename(columns={'index':'time'})
    return yolo_interp, uwb_interp
