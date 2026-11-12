import numpy as np, pandas as pd, lightgbm as lgb
from pandas.tseries.holiday import USFederalHolidayCalendar
from config import *

df = pd.read_parquet(DATA / "hourly.parquet").reset_index()
w = pd.read_csv(DATA / "weather.csv", parse_dates=["ts"])
df = df.merge(w, on="ts", how="left").sort_values(["zone", "ts"]).reset_index(drop=True)

g = df.groupby("zone")["pickups"]
df["lag_24"] = g.shift(24)
df["lag_168"] = g.shift(168)
df["roll_168"] = g.transform(lambda s: s.shift(24).rolling(168).mean())

df["hour"], df["dow"], df["month"] = df.ts.dt.hour, df.ts.dt.day_of_week, df.ts.dt.month
df["is_weekend"] = (df["dow"] >= 5).astype(int)
hol = USFederalHolidayCalendar().holidays(df.ts.min(), df.ts.max())
df["is_holiday"] = df.ts.dt.normalize().isin(hol).astype(int)
df["zone"] = df["zone"].astype("category")

FEATURES = ["zone", "hour", "dow", "month", "is_weekend", "is_holiday",
            "temp", "rain", "wind", "humidity", "lag_24", "lag_168", "roll_168"]

end = pd.Timestamp(TEST_END) + pd.Timedelta(hours=23)
d = df.dropna(subset=["lag_168", "roll_168"])
d = d[d.ts <= end]

test_start = end.normalize() - pd.Timedelta(days=TEST_DAYS - 1)
val_start = test_start - pd.Timedelta(days=14)
train, val, test = d[d.ts < val_start], d[(d.ts >= val_start) & (d.ts < test_start)], d[d.ts >= test_start]
print("test window:", test.ts.min(), "to", test.ts.max())


def wape(y, p):
    return float(np.abs(y - p).sum() / y.sum() * 100)


off = lambda x: np.log(x["roll_168"].to_numpy() + 1)   # each zone's recent level as an offset

m = lgb.LGBMRegressor(objective="poisson", n_estimators=3000, learning_rate=0.05, num_leaves=127,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)
m.fit(train[FEATURES], train["pickups"], init_score=off(train),
      eval_set=[(val[FEATURES], val["pickups"])], eval_init_score=[off(val)],
      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])

pred = np.exp(m.predict(test[FEATURES], raw_score=True) + off(test))
y = test["pickups"].to_numpy()
print("LightGBM   WAPE %:", round(wape(y, pred), 2), " MAE:", round(float(np.abs(y - pred).mean()), 2))
print("Same hr -1d WAPE %:", round(wape(y, test["lag_24"].to_numpy()), 2))
print("Same hr -1w WAPE %:", round(wape(y, test["lag_168"].to_numpy()), 2))
print(pd.Series(m.feature_importances_, FEATURES).sort_values().tail(8))

out = test[["zone", "ts", "pickups"]].copy()
out["zone"] = out["zone"].astype(int)
out["pred"] = pred
out.to_parquet(DATA / "pred.parquet")