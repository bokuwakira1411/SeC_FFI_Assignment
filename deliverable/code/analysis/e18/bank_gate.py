"""E18 gate before E18b (CPU): bank part = J&F(full = E17a) - J&F(pixel) on rest120.
Rule (set by the user before step 2): stop E18b if the bank part is ~0, i.e. point estimate < 0.5 and the CI contains 0
(both object CI and video CI). Exit 0 = run E18b, 3 = stop.

Usage (from SeC/): python3 experiments/e18/bank_gate.py
"""
import os
import sys

sys.path.insert(0, "experiments/e17")
import report as R  # noqa: E402

D = "experiments/e18"
R.RUNS["pixel"] = f"{D}/E18a_pixel"
os.makedirs(f"{D}/eval", exist_ok=True)
for s in R.SETS:
    link = f"{D}/eval/E17a_{s}"
    if not os.path.islink(link):
        os.symlink(os.path.abspath(f"experiments/e17/eval/E17a_{s}"), link)
R.D = D
_, o_full = R.evaluate("E17a", "rest120")
_, o_pix = R.evaluate("pixel", "rest120")
m, lo, hi, up, dn, _ = R.paired_delta(o_pix, o_full)
_, clo, chi = R.cluster_delta(o_pix, o_full)
stop = m < 0.5 and lo <= 0 <= hi and clo <= 0 <= chi
print(f"bank part (full - pixel, rest120): {m:+.2f} object CI [{lo:+.2f}, {hi:+.2f}] video CI [{clo:+.2f}, {chi:+.2f}]"
      f" -> {'STOP E18b' if stop else 'run E18b'}")
sys.exit(3 if stop else 0)
