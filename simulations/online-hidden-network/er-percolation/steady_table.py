"""Run a fresh paired simulation and export the largest-component tables."""
import csv

import numpy as np
import xlsxwriter

from percolation import GRAPHS, M, MU, ROOT, SAMPLES, simulate, summarize

TABLE_SEED = 813402
HEADERS = ("平均度 k", "传统保留概率 p", "动态稳态概率 p*", "上线率 λ", "下线率 μ",
           "传统 平均 L₁（节点）", "动态 平均 L₁（节点）",
           "传统 平均 L₁/M", "动态 平均 L₁/M", "相对误差（%）")


def export_tables(observations):
    destination = ROOT / "tables"
    destination.mkdir(exist_ok=True)
    with (destination / "steady_components.csv").open("w", encoding="utf-8-sig", newline="") as file, \
            xlsxwriter.Workbook(destination / "steady_components.xlsx") as book:
        csv_out = csv.writer(file)
        csv_out.writerow(HEADERS)
        sheet = book.add_worksheet("稳态连通分量")
        header = book.add_format({"bold": True, "bg_color": "#204665", "font_color": "white"})
        sheet.write_row(0, 0, HEADERS, header)
        sheet.set_row(0, 30)
        sheet.freeze_panes(1, 3)
        formats = {}
        for first, last, width, spec in [(0, 0, 14, "0"), (1, 2, 21, "0.000"),
                                         (3, 4, 16, "0.000000"), (5, 6, 25, "#,##0.000"),
                                         (7, 8, 22, "0.000000"), (9, 9, 20, "0.000%")]:
            fmt = book.add_format({"num_format": spec})
            sheet.set_column(first, last, width, fmt)
            formats[first] = fmt
        summary = summarize(observations)
        for i, entry in enumerate(summary, start=1):
            k, x, activation, static, _, _, dynamic = entry[:7]
            if static <= 0:
                raise ValueError("The relative difference requires a positive static reference.")
            rel = abs(dynamic - static) / static
            values = [int(k), x, activation / (activation + MU), activation, MU,
                      static * M, dynamic * M, static, dynamic]
            csv_out.writerow([*values, rel * 100])
            sheet.write_row(i, 0, values)
            r = i + 1
            sheet.write_formula(i, 9, f"=ABS(I{r}-H{r})/H{r}", formats[9], rel)
        sheet.autofilter(0, 0, len(summary), len(HEADERS) - 1)
        notes = book.add_worksheet("统计口径")
        notes.set_column(0, 0, 22)
        notes.set_column(1, 1, 105)
        notes.write_row(0, 0, ["项目", "说明"], header)
        for i, row in enumerate([
            ("统计量", "L₁ 为一次快照中最大连通分量的节点数；比例 L₁/M 的分母为全部节点。"),
            ("相对误差", "以传统组均值为参照，先求两组均值，再计算两者差的绝对值除以传统组均值。"),
            ("网络", f"每张图 {M} 个节点；每个 k 生成 {GRAPHS} 张独立 ER 底图，组间逐张配对。"),
            ("采样", f"每个概率、每张底图、每个机制采样 {SAMPLES} 次。先在图内取均值，再对底图取均值。"),
            ("概率配对", "传统 p 与动态 p* 对齐，动态组固定 μ=0.01，λ=μp/(1−p)。"),
            ("在线机制", "初始全在线，边仅在底图的两个原有端点都在线时可见。"),
            ("动态取样时间", "τ=1/(λ+μ)，首个快照在 10τ，随后每隔 3τ 取样。"),
            ("随机种子", str(TABLE_SEED)),
        ], start=1):
            notes.write_row(i, 0, row)


if __name__ == "__main__":
    export_tables(simulate(seed=TABLE_SEED))
