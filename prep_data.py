import pandas as pd, requests
import pyarrow.parquet as pq
from config import *


def month_counts(month):
    t = pq.read_table(RAW / f"yellow_tripdata_{month}.parquet",
                      columns=["tpep_pickup_datetime", "PULocationID"]).to_pandas()
    start = pd.Timestamp(month + "-01")
    end = start + pd.offsets.MonthBegin(1)
    ts = t["tpep_pickup_datetime"]
    t = t[(ts >= start) & (ts < end) & t["PULocationID"].between(1, 263)]
    t = t.assign(ts=t["tpep_pickup_datetime"].dt.floor("h"))
    out = t.groupby(["PULocationID", "ts"]).size().rename("pickups").reset_index()
    print(month, len(t), "trips")
    return out.rename(columns={"PULocationID": "zone"})


df = pd.concat(month_counts(m) for m in MONTHS)
top = df.groupby("zone")["pickups"].sum().nlargest(TOP_ZONES).index.sort_values()
df = df[df["zone"].isin(top)].set_index(["zone", "ts"])

first = pd.Timestamp(MONTHS[0] + "-01")
last = pd.Timestamp(MONTHS[-1] + "-01") + pd.offsets.MonthEnd(0)
idx = pd.date_range(first, last + pd.Timedelta(hours=23), freq="h")
df = df.reindex(pd.MultiIndex.from_product([top, idx], names=["zone", "ts"]), fill_value=0)
df.to_parquet(DATA / "hourly.parquet")

r = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=120, params={
    "latitude": LAT, "longitude": LON, "start_date": str(first.date()), "end_date": str(last.date()),
    "hourly": "temperature_2m,precipitation,wind_speed_10m,relative_humidity_2m",
    "timezone": "America/New_York"})
r.raise_for_status()
w = pd.DataFrame(r.json()["hourly"])
w.columns = ["ts", "temp", "rain", "wind", "humidity"]
w.to_csv(DATA / "weather.csv", index=False)
print("done:", df.shape)