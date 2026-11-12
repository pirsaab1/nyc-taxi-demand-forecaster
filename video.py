import geopandas as gpd, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.animation import FuncAnimation, FFMpegWriter
import imageio_ffmpeg
from config import *

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
BG, TXT, MUTED = "#0b1220", "#e2e8f0", "#94a3b8"
STEPS, HOLD = 4, 12

z = gpd.read_file(next(ZONES.rglob("*.shp"))).to_crs(4326)[["LocationID", "geometry"]].rename(columns={"LocationID": "zone"})
z["zone"] = z["zone"].astype(int)
pred = pd.read_parquet(DATA / "pred.parquet")

tot = pred.groupby(pred.ts.dt.normalize())["pickups"].sum()
DAY = tot.idxmax()                                   # busiest day in the test window
day = pred[pred.ts.dt.normalize() == DAY]
A = np.nan_to_num(day.pivot(index="ts", columns="zone", values="pickups").reindex(columns=z["zone"]).to_numpy())
P = np.nan_to_num(day.pivot(index="ts", columns="zone", values="pred").reindex(columns=z["zone"]).to_numpy())
wape = np.abs(A - P).sum() / A.sum() * 100
norm = PowerNorm(0.5, vmin=0, vmax=np.percentile(A, 99.5))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 8.4), dpi=100)
fig.patch.set_facecolor(BG)
fig.subplots_adjust(left=0.02, right=0.98, top=0.76, bottom=0.12, wspace=0.13)
colls = []
for ax, title in ((a1, "Actual"), (a2, "Predicted (LightGBM)")):
    z.assign(v=A[0]).plot(column="v", ax=ax, cmap="inferno", vmin=0, vmax=1, edgecolor="#26344d", linewidth=0.25)
    c = ax.collections[0]
    c.set_norm(norm)
    colls.append(c)
    ax.set_facecolor(BG); ax.axis("off")
    ax.set_title(title, color=TXT, fontsize=18, pad=10)

cax = fig.add_axes([0.32, 0.06, 0.36, 0.018])
cb = fig.colorbar(colls[0], cax=cax, orientation="horizontal")
cb.set_label("pickups per zone per hour", color=MUTED); cb.ax.tick_params(colors=MUTED)
cb.outline.set_visible(False)

fig.text(0.04, 0.94, "NYC Taxi Demand: Actual vs Forecast", color=TXT, fontsize=26, weight="bold")
fig.text(0.04, 0.90, f"Yellow taxi pickups by zone · one day from the 14-day backtest · day-level WAPE {wape:.1f}%",
         color=MUTED, fontsize=14)
clock = fig.text(0.5, 0.82, "", color="#fb923c", fontsize=30, weight="bold", ha="center")
n1 = fig.text(0.20, 0.07, "", color=MUTED, fontsize=14, ha="center")
n2 = fig.text(0.80, 0.07, "", color=MUTED, fontsize=14, ha="center")


def update(k):
    pos = min(k, 23 * STEPS) / STEPS
    h = int(pos); f = pos - h; h2 = min(h + 1, 23)
    va, vp = A[h] * (1 - f) + A[h2] * f, P[h] * (1 - f) + P[h2] * f
    colls[0].set_array(va); colls[1].set_array(vp)
    clock.set_text(f"{DAY + pd.Timedelta(hours=pos):%a %d %b  %H:%M}")
    n1.set_text(f"{va.sum():,.0f} pickups/h"); n2.set_text(f"{vp.sum():,.0f} pickups/h")


peak_h = int(A.sum(axis=1).argmax())
update(peak_h * STEPS)
fig.savefig(OUT / "poster.png", dpi=150, facecolor=BG)

anim = FuncAnimation(fig, update, frames=23 * STEPS + 1 + HOLD)
anim.save(OUT / "taxi_forecast.mp4", writer=FFMpegWriter(fps=12, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]),
          savefig_kwargs={"facecolor": BG})
print("Saved:", OUT / "taxi_forecast.mp4", "and poster.png")