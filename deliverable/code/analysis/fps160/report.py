"""fps160 report: J&F checks, fps / throughput / ms per frame, slowdown, GPU sharing, E14 on 160 videos (CPU).

Usage (from SeC/): python experiments/fps160/report.py
Inputs: experiments/fps160/<run>/{masks,frames,videos.csv}, gpu_apps.csv (run_interleaved.py).
Evaluation: vos_evaluation/sav_evaluator.py --scene_info --strict on symlinked pred dirs
(experiments/fps160/eval/<run>_<set>/masks). Bootstrap as experiments/final_report.py (10,000, seed 0).
Writes experiments/fps160/report.md.
"""
import csv
import filecmp
import glob
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, "experiments")
from final_report import cluster_delta, paired_delta, parse_eval, per_object  # noqa: E402

D = "experiments/fps160"
RUNS = ["E0", "E10a", "E14"]
S40 = [l.strip() for l in open("experiments/subset40.txt") if l.strip()]
R120 = [l.strip() for l in open("experiments/full160/rest120.txt") if l.strip()]
SETS = {"all160": S40 + R120, "rest120": R120, "subset40": S40}
GT = {"all160": "experiments/full160/gt_all160", "rest120": "experiments/full160/gt_rest120",
      "subset40": "experiments/gt_subset40"}
OLD = {  # earlier masks the new ones must equal bit for bit (deterministic pipeline)
    "E0": {v: "experiments/E0/masks" for v in S40},
    "E10a": {**{v: "experiments/E10a/masks" for v in S40},
             **{v: "experiments/full160/E10a_rest120/masks" for v in R120}},
    "E14": {v: "experiments/bank/E14/masks" for v in S40},
}
EXPECT = {("E0", "all160"): 69.9, ("E10a", "all160"): 72.6, ("E14", "subset40"): 75.0}
N_BOOT = 10000


def evaluate(run, s):
    d = f"{D}/eval/{run}_{s}"
    os.makedirs(f"{d}/masks", exist_ok=True)
    for v in SETS[s]:
        link = f"{d}/masks/{v}"
        if not os.path.islink(link):
            os.symlink(os.path.abspath(f"{D}/{run}/masks/{v}"), link)
    if not os.path.exists(f"{d}/eval.txt"):
        with open(f"{d}/eval.tmp", "w") as fh:
            subprocess.run([sys.executable, "vos_evaluation/sav_evaluator.py", "--gt_root", GT[s],
                            "--pred_root", f"{d}/masks", "--scene_info", "data/SeCVOS/scene_info.json",
                            "--strict"], stdout=fh, stderr=subprocess.STDOUT, check=True,
                           env={**os.environ, "PYTHONPATH": os.getcwd()})
        os.replace(f"{d}/eval.tmp", f"{d}/eval.txt")
    return parse_eval(f"{d}/eval.txt"), per_object(f"{d}/masks")


def same_masks(run):
    """(videos identical, videos compared, list of differing videos) vs the earlier run's masks."""
    diff = []
    for v, old in OLD[run].items():
        new_files = sorted(os.path.relpath(p, f"{D}/{run}/masks/{v}")
                           for p in glob.glob(f"{D}/{run}/masks/{v}/*/*.png"))
        old_files = sorted(os.path.relpath(p, f"{old}/{v}") for p in glob.glob(f"{old}/{v}/*/*.png"))
        if new_files != old_files or not all(
                filecmp.cmp(f"{D}/{run}/masks/{v}/{f}", f"{old}/{v}/{f}", shallow=False) for f in new_files):
            diff.append(v)
    return len(OLD[run]) - len(diff), len(OLD[run]), diff


def shared_videos():
    """Videos where any run saw a foreign GPU process (2 s sampler or snapshot after the video)."""
    out = {}
    for r in csv.DictReader(open(f"{D}/gpu_apps.csv")):
        apps = (r["foreign_seen"] + " " + r["foreign_after_video"]).split()
        if apps:
            out.setdefault(r["video"], set()).update(apps)
    return out


def load_logs(run):
    vids = {v["video"]: v for v in csv.DictReader(open(f"{D}/{run}/videos.csv"))}
    rows = [r for p in glob.glob(f"{D}/{run}/frames/*.csv") for r in csv.DictReader(open(p))]
    return vids, rows


def speed(run, videos, logs):
    vids, rows = logs[run]
    vs = [vids[v] for v in videos]
    fps = np.array([int(v["n_frames"]) / float(v["video_sec"]) for v in vs])
    sec = sum(float(v["video_sec"]) for v in vs)
    keep = set(videos)
    nc = [r for r in rows if r["video"] in keep and r["is_cond"] == "0"]
    call = [float(r["frame_ms"]) for r in nc if r["triggered"] == "1"]
    nocall = [float(r["frame_ms"]) for r in nc if r["triggered"] == "0"]
    frame_sec = sum(float(r["frame_ms"]) for r in rows if r["video"] in keep) / 1000
    return dict(fps_med=np.median(fps), fps_mean=fps.mean(), thr=sum(int(v["n_obj_frames"]) for v in vs) / sec,
                nocall_ms=np.median(nocall), call_ms=np.median(call) if call else float("nan"),
                call_pct=100 * len(call) / len(nc), n_call=len(call), call_sum=sum(call) / 1000,
                nocall_sum=sum(nocall) / 1000, sec=sec, overhead=100 * (sec - frame_sec) / sec,
                added=sum(r["bank_reason"].startswith("added") for r in nc),
                replaced=sum(r["bank_reason"] == "added_div_replace" for r in nc))


def slowdown(run, base, videos, logs):
    """Per-video time ratio run / base; median and 95% bootstrap CI over videos."""
    r = np.array([float(logs[run][0][v]["video_sec"]) / float(logs[base][0][v]["video_sec"]) for v in videos])
    rng = np.random.default_rng(0)
    boots = np.median(r[rng.integers(0, len(r), size=(N_BOOT, len(r)))], axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return np.median(r), lo, hi


def ci(lo, hi):
    return f"[{lo:+.1f}, {hi:+.1f}]{' *' if lo > 0 or hi < 0 else ''}"


def main():
    for s in ("all160", "rest120"):
        if not os.path.exists(GT[s]):
            sys.exit(f"missing {GT[s]}")
    res = {(r, s): evaluate(r, s) for r in RUNS for s in SETS}
    logs = {r: load_logs(r) for r in RUNS}
    shared = shared_videos()
    clean = [v for v in SETS["all160"] if v not in shared]
    md = ["# fps160: E0, E10a, E14 trên 160 video, cùng code, chạy xen kẽ\n",
          "Mỗi video chạy E0 → E10a → E14 liền nhau, trong một process (model load một lần). "
          "Một video khởi động (`_H00-2zCavQ`) chạy trước và không tính. Seed 0.\n"]

    md.append("## 1. Kiểm tra tính đúng\n")
    ok = True
    for (r, s), ref in EXPECT.items():
        jf = res[(r, s)][0]["overall"][0]
        good = abs(jf - ref) <= 0.1 + 1e-9
        ok &= good
        md.append(f"- {r} trên {s}: J&F {jf:.1f}, lần trước {ref:.1f}. {'Khớp.' if good else '**KHÔNG KHỚP.**'}")
    for r in RUNS:
        n_same, n, diff = same_masks(r)
        md.append(f"- Mask {r} trùng từng bit với lần chạy trước trên {n_same}/{n} video"
                  + (f". Khác: {', '.join(diff)}." if diff else "."))
        ok &= not diff
    md.append("- E0 trên rest120 chỉ so được qua J&F: lần chạy E0 160 video trước dùng code gốc.\n")

    md.append("## 2. J&F\n")
    for s in ("all160", "rest120", "subset40"):
        n_obj = len(res[("E0", s)][1])
        md.append(f"\n### {s} ({len(SETS[s])} video, {n_obj} object)\n")
        md.append("| Run | J&F | J | F | J&F first / second / remain | Δ vs E0 [object CI] | Δ vs E0 video CI "
                  "| up / down vs E0 | Δ vs E10a [object CI; video CI] | up / down vs E10a |")
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in RUNS:
            e, o = res[(r, s)]
            seg = " / ".join(f"{e[k][0]:.1f}" for k in ("first_segment", "second_segment", "remain_segments"))
            cells = [f"{e['overall'][0]:.1f}", f"{e['overall'][1]:.1f}", f"{e['overall'][2]:.1f}", seg]
            if r == "E0":
                cells += ["–"] * 5
            else:
                m, lo, hi, up, dn, _ = paired_delta(res[("E0", s)][1], o)
                _, clo, chi = cluster_delta(res[("E0", s)][1], o)
                cells += [f"{m:+.1f} {ci(lo, hi)}", ci(clo, chi), f"{up} / {dn}"]
                if r == "E14":
                    m, lo, hi, up, dn, _ = paired_delta(res[("E10a", s)][1], o)
                    _, clo, chi = cluster_delta(res[("E10a", s)][1], o)
                    cells += [f"{m:+.1f} {ci(lo, hi)}; {ci(clo, chi)}", f"{up} / {dn}"]
                else:
                    cells += ["–", "–"]
            md.append(f"| {r} | " + " | ".join(cells) + " |")

    md.append("\n## 3. Tốc độ\n")
    md.append("- **fps theo video** = số frame / thời gian xử lý video cho mọi object. "
              "Không tính load model. Có tính giải mã JPEG, frame đầu, đọc/ghi mask.")
    md.append("- **Thông lượng** = tổng (frame × object) / tổng thời gian.")
    md.append("- **ms/frame**: thời gian của một (object, frame), bỏ frame đầu.")
    md.append("- Bấm giờ bằng `perf_counter`, có `torch.cuda.synchronize()` ở hai đầu.\n")
    sp = {}
    for label, vids in ((f"cả 160 video", SETS["all160"]), (f"{len(clean)} video không bị chia GPU", clean)):
        md.append(f"\n### {label}\n")
        md.append("| Run | fps theo video (trung vị; trung bình) | thông lượng | ms/frame không gọi concept (trung vị) "
                  "| ms/frame có gọi concept (trung vị) | % frame gọi concept | frame thêm vào bank | overhead ngoài frame |")
        md.append("|---|---|---|---|---|---|---|---|")
        for r in RUNS:
            x = speed(r, vids, logs)
            sp[(r, label)] = x
            add = f"{x['added']}" + (f" ({x['replaced']} thay thế)" if x["replaced"] else "")
            md.append(f"| {r} | {x['fps_med']:.2f}; {x['fps_mean']:.2f} | {x['thr']:.2f} | {x['nocall_ms']:.0f} "
                      f"| {x['call_ms']:.0f} | {x['call_pct']:.1f} | {add} | {x['overhead']:.1f} % |")

    md.append("\n## 4. Tỷ lệ chậm đi (thời gian run / thời gian E0, theo từng video, trung vị, CI 95 % bootstrap theo video)\n")
    md.append("| So sánh | cả 160 video | video không bị chia GPU |")
    md.append("|---|---|---|")
    for r, b in (("E10a", "E0"), ("E14", "E0"), ("E14", "E10a")):
        cells = []
        for vids in (SETS["all160"], clean):
            m, lo, hi = slowdown(r, b, vids, logs)
            cells.append(f"×{m:.2f} [{lo:.2f}, {hi:.2f}]")
        md.append(f"| {r} / {b} | " + " | ".join(cells) + " |")

    md.append("\n## 5. Video bị chia GPU\n")
    md.append("Bỏ qua `gnome-remote-desktop-daemon` (225 MiB, chạy thường trực), như đã thống nhất.\n")
    if shared:
        md.append(f"{len(shared)} video:\n")
        for v in sorted(shared):
            md.append(f"- {v}: {', '.join(sorted(shared[v]))}")
    else:
        md.append("Không có video nào.")

    md.append("\n## 6. Phần chậm thêm đến từ đâu\n")
    lab = "cả 160 video"
    e0, e10, e14 = (sp[(r, lab)] for r in RUNS)
    md.append("Tách theo tổng thời gian (không dùng trung vị), trên cả 160 video:\n")
    md.append("| Run | tổng thời gian (s) | số lần gọi concept | ms TB / frame có gọi | ms TB / frame không gọi "
              "| số frame ngữ cảnh TB / lần gọi | predict_forward TB (ms) |")
    md.append("|---|---|---|---|---|---|---|")
    for r, x in zip(RUNS, (e0, e10, e14)):
        trig = [q for q in logs[r][1] if q["triggered"] == "1"]
        ctx = np.mean([int(q["n_ctx_frames"] or 0) for q in trig])
        pf = np.mean([float(q["predict_forward_ms"]) for q in trig if q["predict_forward_ms"]])
        n_nocall = round(x["n_call"] * (100 - x["call_pct"]) / x["call_pct"])
        md.append(f"| {r} | {x['sec']:.0f} | {x['n_call']} | {1000 * x['call_sum'] / x['n_call']:.0f} "
                  f"| {1000 * x['nocall_sum'] / n_nocall:.0f} | {ctx:.1f} | {pf:.0f} |")
    extra = e10["n_call"] - e0["n_call"]
    per = 1000 * e10["call_sum"] / e10["n_call"] - 1000 * e0["nocall_sum"] / round(
        e0["n_call"] * (100 - e0["call_pct"]) / e0["call_pct"])
    d_call = 1000 * (e14["call_sum"] / e14["n_call"] - e10["call_sum"] / e10["n_call"])
    md.append(f"\n- **E10a so với E0**: thêm {extra} lần gọi concept. Mỗi lần biến một frame thường thành frame "
              f"có gọi, tốn thêm khoảng {per:.0f} ms. {extra} × {per:.0f} ms ≈ {extra * per / 1000:.0f} s. "
              f"Thời gian thực tăng {e10['sec'] - e0['sec']:.0f} s.")
    md.append(f"- **E14 so với E10a**: cùng số lần gọi ({e14['n_call']}). Mỗi lần gọi đắt hơn {d_call:.0f} ms, "
              f"vì bank lớn hơn nên LLM nhận nhiều frame ngữ cảnh hơn. {e14['n_call']} × {d_call:.0f} ms ≈ "
              f"{e14['n_call'] * d_call / 1000:.0f} s. Thời gian thực tăng {e14['sec'] - e10['sec']:.0f} s.")
    md.append("- Phần ngoài frame (giải mã, đọc/ghi mask) gần như bằng nhau ở ba run.")
    md.append("")
    md.append("Kiểm tra tính đúng: " + ("đạt." if ok else "**KHÔNG ĐẠT, cần xem lại trước khi dùng số tốc độ.**"))
    open(f"{D}/report.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
