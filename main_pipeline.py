from __future__ import annotations


if time_keys:
key = time_keys[0]
df = truth.merge(pred, on=key, how="inner")
else:
n = min(len(truth), len(pred))
df = truth.iloc[:n].reset_index(drop=True)
pred = pred.iloc[:n].reset_index(drop=True)
df = pd.concat([df, pred], axis=1)


# Default: 2D correction columns; fallback to first numeric columns
if {"target_dx","target_dy"}.issubset(df.columns) and {"pred_dx","pred_dy"}.issubset(df.columns):
y = df[["target_dx","target_dy"]].to_numpy()
yhat = df[["pred_dx","pred_dy"]].to_numpy()
elif "target" in df.columns and "pred" in df.columns:
y = df[["target"]].to_numpy(); yhat = df[["pred"]].to_numpy()
else:
y = truth.select_dtypes(include=[float,int]).to_numpy()
yhat = pred.select_dtypes(include=[float,int]).to_numpy()
y = y[:, :yhat.shape[1]]


rmse = float(np.sqrt(mean_squared_error(y, yhat)))
mae = float(mean_absolute_error(y, yhat))
print(f"RMSE={rmse:.4f} MAE={mae:.4f} N={len(df)}")




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
