# scripts/make_fake_data.py
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

def make_set(n_sec=60, hz=10, bias=(0.2, -0.1), noise=0.03, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(0, n_sec*1000, 1000/hz)  # ms
    t_ms = np.round(t).astype(int)

    # YOLO(px->m) 궤적: 8자 모양
    s = np.linspace(0, 4*np.pi, len(t_ms))
    x_ip = 2.0*np.sin(s)
    y_ip = 1.0*np.sin(s)*np.cos(s)

    # 유도 속도(검증용): 실제 파이프라인은 train.py에서 필요 시 자동 생성
    dt = (np.diff(t_ms, prepend=t_ms[0])/1000.0)
    vx = np.gradient(x_ip, dt, edge_order=2)
    vy = np.gradient(y_ip, dt, edge_order=2)
    v_ip = np.hypot(vx, vy)

    # UWB: YOLO + (고정 바이어스) + 가우시안 노이즈
    uwb_x = x_ip + bias[0] + rng.normal(0, noise, size=len(t_ms))
    uwb_y = y_ip + bias[1] + rng.normal(0, noise, size=len(t_ms))

    # 부가 피처
    speed = np.hypot(np.gradient(uwb_x, dt), np.gradient(uwb_y, dt))
    accel = np.gradient(speed, dt)
    anch_count = rng.integers(4, 9, size=len(t_ms))  # 4~8개 앵커 가정

    df = pd.DataFrame(dict(
        t_ms=t_ms, x_ip=x_ip, y_ip=y_ip, v_ip=v_ip,
        uwb_x=uwb_x, uwb_y=uwb_y,
        speed=speed, accel=accel, anch_count=anch_count
    ))
    return df

if __name__ == "__main__":
    Path("data").mkdir(parents=True, exist_ok=True)
    train = make_set(n_sec=90, hz=10, seed=42)
    eval_  = make_set(n_sec=30, hz=10, seed=777)  # 다른 시드
    train.to_csv("data/train.csv", index=False)
    eval_.to_csv("data/eval.csv", index=False)
    print("Wrote data/train.csv, data/eval.csv")
