from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from src.gpr.data import load_csv, sync_and_resample, select_xy
from src.gpr.model import GPRModel, GPRConfig


def _load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_train(args):
    cfg = _load_cfg(args.config)
    df = load_csv(args.train)
    df = sync_and_resample(df, cfg.get("time_col", "t_ms"), cfg.get("resample_hz"))

    X, y = select_xy(df, cfg["feature_cols"], cfg["target_cols"])
    n_targets = y.shape[1] if y.ndim > 1 else 1
    gcfg = GPRConfig(
        kernel=cfg.get("model", {}).get("kernel", "RBF"),
        length_scale=float(cfg.get("model", {}).get("length_scale", 1.0)),
        ard=bool(cfg.get("model", {}).get("ard", True)),
        noise=float(cfg.get("model", {}).get("noise", 1e-3)),
        alpha=float(cfg.get("model", {}).get("alpha", 1e-6)),
        matern_nu=float(cfg.get("model", {}).get("matern_nu", 1.5)),
        random_state=int(cfg.get("seed", 42)),
    )

    model = GPRModel(n_features=X.shape[1], n_targets=n_targets, cfg=gcfg)
    model.fit(X, y, cfg["feature_cols"], cfg["target_cols"])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.joblib"
    model.save(str(model_path))
    print(f"[train] saved: {model_path}")


def cmd_infer(args):
    model = GPRModel.load(args.model)
    df = load_csv(args.input)

    cfg = _load_cfg(args.config) if args.config else {}
    time_col = cfg.get("time_col", "t_ms")
    if cfg.get("resample_hz"):
        df = sync_and_resample(df, time_col, cfg.get("resample_hz"))

    feat_cols = cfg.get("feature_cols", model.feature_names)
    if feat_cols is None:
        raise RuntimeError("feature_cols not provided and not stored in model. Provide --config.")

    X, _ = select_xy(df, feat_cols, None)
    mu, std = model.predict(X, return_std=True)

    out = pd.DataFrame({time_col: df[time_col] if time_col in df.columns else np.arange(len(df))})
    if mu.ndim == 1 or mu.shape[1] == 1:
        out["pred"] = mu.ravel()
        out["std"] = std.ravel()
    else:
        out[["pred_dx", "pred_dy"]] = mu
        out[["std_dx", "std_dy"]] = std

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[infer] wrote: {args.out}  shape={out.shape}")


def cmd_eval(args):
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    truth = load_csv(args.truth)
    pred = load_csv(args.pred)
    time_keys = [c for c in ["t_ms", "time", "timestamp"] if c in truth.columns and c in pred.columns]

    if time_keys:
        key = time_keys[0]
        df = truth.merge(pred, on=key, how="inner")
    else:
        n = min(len(truth), len(pred))
        df = truth.iloc[:n].reset_index(drop=True)
        pred = pred.iloc[:n].reset_index(drop=True)
        df = pd.concat([df, pred], axis=1)

    if {"target_dx", "target_dy"}.issubset(df.columns) and {"pred_dx", "pred_dy"}.issubset(df.columns):
        y = df[["target_dx", "target_dy"]].to_numpy()
        yhat = df[["pred_dx", "pred_dy"]].to_numpy()
    elif "target" in df.columns and "pred" in df.columns:
        y = df[["target"]].to_numpy()
        yhat = df[["pred"]].to_numpy()
    else:
        y = truth.select_dtypes(include=[float, int]).to_numpy()
        yhat = pred.select_dtypes(include=[float, int]).to_numpy()
        y = y[:, : yhat.shape[1]]

    rmse = float(np.sqrt(mean_squared_error(y, yhat)))
    mae = float(mean_absolute_error(y, yhat))
    print(f"RMSE={rmse:.4f}  MAE={mae:.4f}  N={len(df)}")


def build_parser():
    p = argparse.ArgumentParser(description="GPR indoor correction pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train")
    pt.add_argument("--config", required=True)
    pt.add_argument("--train", required=True)
    pt.add_argument("--out", required=True)
    pt.set_defaults(func=cmd_train)

    pi = sub.add_parser("infer")
    pi.add_argument("--input", required=True)
    pi.add_argument("--model", required=True)
    pi.add_argument("--out", required=True)
    pi.add_argument("--config", required=False)
    pi.set_defaults(func=cmd_infer)

    pe = sub.add_parser("eval")
    pe.add_argument("--truth", required=True)
    pe.add_argument("--pred", required=True)
    pe.set_defaults(func=cmd_eval)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
