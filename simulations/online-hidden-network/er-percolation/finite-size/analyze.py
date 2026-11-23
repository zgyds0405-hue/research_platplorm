"""Locate L2 peaks, bootstrap graph pairs and extrapolate finite-size estimates."""
import csv
from pathlib import Path

import numpy as np
import xlsxwriter

ROOT = Path(__file__).resolve().parent
BOOT_COUNT = 5000
BOOT_SEED = 590273


def interval(values):
    return tuple(np.quantile(values, [.025, .975]))


def peak_window(mean_l2):
    """Start with the two coarse peaks; enlarge until the fine-grid maximum is enclosed."""
    coarse = np.arange(0, len(mean_l2), 10)
    centers = coarse[mean_l2[coarse].argmax(axis=0)]
    left = max(0, centers.min() - 30)
    right = min(len(mean_l2)-1, centers.max() + 30)
    full_max = mean_l2.max(axis=0)
    while True:
        local = left + mean_l2[left:right+1].argmax(axis=0)
        if ((local > left) & (local < right)).all() and np.array_equal(
                mean_l2[local, np.arange(2)], full_max):
            return left, right, local
        if left == 0 and right == len(mean_l2)-1:
            raise ValueError("A peak lies on the full scan boundary")
        left = max(0, left - 30)
        right = min(len(mean_l2)-1, right + 30)


def fit_weights(sizes):
    # Multiplying the first or second row of the pseudoinverse by observations
    # returns the intercept or slope of the unweighted least-squares regression.
    predictors = sizes.astype(float) ** (-1/3)
    return np.linalg.pinv(np.column_stack((np.ones(len(sizes)), predictors)))[0]


def analyze():
    data = np.load(ROOT / "paired_runs.npz")
    probs, sizes = data["probabilities"], data["sizes"]
    runs = data["component_sums"] / int(data["snapshots"])
    peak_rows, fit_rows, l1_rows, curves = [], [], [], []
    rng = np.random.default_rng(BOOT_SEED)
    for ki, k in enumerate(data["degrees"]):
        peaks, boot_peaks, critical_sizes, boot_sizes = [], [], [], []
        threshold = 1 / k
        critical_idx, = np.flatnonzero(probs == threshold)
        for mi, n in enumerate(sizes):
            values = runs[ki, mi]  # [paired graph, probability, observable]
            mean = values.mean(axis=0)
            left, right, loc = peak_window(mean[:, [1, 3]])
            pair = probs[loc]
            reps = len(values)
            # One vector of multiplicities resamples both mechanisms and the full
            # probability grid. Graphs at different sizes are generated independently.
            weights = rng.multinomial(reps, np.full(reps, 1/reps), size=BOOT_COUNT)
            boot_pair = np.column_stack([
                probs[np.argmax(weights @ values[:, :, col], axis=1)] for col in (1, 3)])
            peaks.append(pair)
            boot_peaks.append(boot_pair)
            peak_rows.append((k, n, threshold, probs[left], probs[right],
                              pair[0], *interval(boot_pair[:, 0]), pair[1], *interval(boot_pair[:, 1]),
                              pair[1]-pair[0], *interval(boot_pair[:, 1]-boot_pair[:, 0])))
            graph_l1 = values[:, critical_idx, :][:, [0, 2]]
            l1 = graph_l1.mean(axis=0)
            critical_sizes.append(l1)
            boot_sizes.append(weights @ graph_l1 / reps)
            s, d = l1/n
            l1_rows.append((k, n, threshold, *l1, s, d, abs(d-s), abs(d-s)/s*100))
            curves.extend((k, n, x, *v) for x, v in zip(probs, mean))
        peaks = np.array(peaks)
        intercept = fit_weights(sizes)
        fit = intercept @ peaks
        boot = np.einsum("m,mba->ba", intercept, boot_peaks)
        trimmed = fit_weights(sizes[1:]) @ peaks[1:]
        slope_weights = np.linalg.pinv(np.column_stack((np.ones(len(sizes)),np.log(sizes))))[1]
        slopes = slope_weights @ np.log(critical_sizes)
        slope_boot = np.einsum("m,mba->ba", slope_weights, np.log(boot_sizes))
        fit_rows.append((k, threshold, fit[0], *interval(boot[:, 0]), fit[1], *interval(boot[:, 1]),
                         fit[1]-fit[0], *interval(boot[:, 1]-boot[:, 0]), *trimmed,
                         slopes[0], *interval(slope_boot[:, 0]), slopes[1], *interval(slope_boot[:, 1])))
    return peak_rows, fit_rows, l1_rows, curves


def export():
    peaks, fits, l1, curves = analyze()
    tables = [
        ("表1 有限规模峰位", "peaks",
         "k M 理论阈值 粗扫后窗口下界 粗扫后窗口上界 传统峰位 传统下限 传统上限 动态峰位 动态下限 动态上限 动态减传统 差值下限 差值上限", peaks),
        ("表2 临界点外推", "extrapolation",
         "k 理论阈值 传统外推 传统下限 传统上限 动态外推 动态下限 动态上限 动态减传统 差值下限 差值上限 去除最小规模_传统 去除最小规模_动态 传统L1指数 指数下限_传统 指数上限_传统 动态L1指数 指数下限_动态 指数上限_动态", fits),
        ("表3 临界处最大分量", "critical_l1",
         "k M 理论阈值 传统平均L1 动态平均L1 传统平均L1/M 动态平均L1/M 均值差绝对值 相对误差(%)", l1),
    ]
    with xlsxwriter.Workbook(ROOT / "finite_size.xlsx") as book:
        head = book.add_format({"bold": True, "bg_color":"#204665", "font_color":"white"})
        numeric = book.add_format({"num_format":"0.00000"})
        for title, stem, columns, rows in tables:
            headers = columns.split()
            with (ROOT / f"{stem}.csv").open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(headers)
                writer.writerows(rows)
            sheet = book.add_worksheet(title)
            sheet.freeze_panes(1, 2)
            sheet.write_row(0, 0, headers, head)
            sheet.set_column(0, 0, 8)
            for j, label in enumerate(headers[1:], start=1):
                sheet.set_column(j, j, max(15, len(label)*2+4), None if label=="M" else numeric)
            for i, row in enumerate(rows, 1):
                sheet.write_row(i, 0, row)
            sheet.autofilter(0, 0, len(rows), len(headers)-1)
        sheet = book.add_worksheet("统计说明")
        sheet.set_column(0, 0, 18)
        sheet.set_column(1, 1, 116)
        for i, row in enumerate([
            ("项目","口径"),
            ("仿真","每种 k、M 生成 10 张独立 ER 底图，两组配对；每张底图、每个概率、每组 8 个快照。"),
            ("峰位","先对快照、再对 10 张底图取平均，寻找 L₂ 的峰。并列最大值取最左端。"),
            ("概率网格","先以 0.01 粗扫，再用 0.001 细扫；使用全区间细网格复核。"),
            ("动态组","全在线起步，μ=0.01，λ=μp/(1−p)。首个快照在 10τ，之后间隔 3τ，τ=1/(λ+μ)。"),
            ("区间","95% 配对底图 bootstrap 百分位区间，5,000 次。每个规模独立重抽，并在全网格重新找峰。"),
            ("外推","五规模无权最小二乘拟合 p_peak=b+aM^(−1/3)，b 为自由截距；另列去除 M=2000 后的截距。"),
            ("指数","在 p=1/k 处拟合 log(平均L₁) 对 log(M) 的斜率，理论值为 2/3。"),
            ("误差","表3先对 10 次结果取均值，再取两组均值差的绝对值，相对误差以传统组为分母。"),
            ("局限","区间反映有限底图的抽样波动，不包含外推模型或更高阶规模修正产生的系统误差。")
        ]):
            sheet.write_row(i,0,row,head if i==0 else None)
    with (ROOT / "mean_scan.csv").open("w",newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["k","M","p","static_L1","static_L2","dynamic_L1","dynamic_L2"])
        writer.writerows(curves)


if __name__ == "__main__":
    export()
