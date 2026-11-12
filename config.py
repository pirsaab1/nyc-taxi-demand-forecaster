from pathlib import Path

ROOT = Path(r"F:\Github5")
DATA, OUT = ROOT / "data", ROOT / "outputs"
RAW, ZONES = DATA / "raw", DATA / "zones"
OUT.mkdir(exist_ok=True)

MONTHS = [f"2025-{m:02d}" for m in range(1, 13)]
LAT, LON = 40.78, -73.97      # Manhattan
TOP_ZONES = 120
TEST_DAYS = 14
TEST_END = "2025-11-16"