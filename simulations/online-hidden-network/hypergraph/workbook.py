"""Create a Chinese-language workbook, including the full probability scan."""
import csv
import gzip
import xlsxwriter
from simulate import ROOT

SHEETS=[
 ("稳态代表点","steady_selected.csv"),("外推临界点","thresholds.csv"),
 ("有限规模峰位","finite_peaks.csv"),("超边与时间","clock.csv"),
 ("单节点上线事件","events.csv"),("底图结构核查","topology.csv"),
 ("全部稳态结果","full_scan.csv.gz")]
NAMES={
 "h":"超边大小 h","z":"目标平均超度 z","N":"总节点数 N","p":"占据概率 p = p*","lambda":"上线率 λ","mu":"下线率 μ",
 "static_S":"传统 L₁/N","dynamic_S":"动态 L₁/N","static_CI_half":"传统95%半宽","dynamic_CI_half":"动态95%半宽",
 "relative_error_pct":"相对误差 (%)","diff_lo":"配对差95%下限","diff_hi":"配对差95%上限","diff":"动态减传统",
 "theory_S":"理论 L₁/N","static_L2":"传统平均 L₂","dynamic_L2":"动态平均 L₂","static_edge_fraction":"传统有效超边比例",
 "dynamic_edge_fraction":"动态有效超边比例","theory_edge_fraction":"理论有效超边比例","dynamic_online_fraction":"动态在线比例",
 "dynamic_component_count":"动态分量数","pc_theory":"理论临界概率","static_peak":"传统峰位","dynamic_peak":"动态峰位",
 "static_fit":"传统外推","dynamic_fit":"动态外推","static_lo":"传统95%下限","static_hi":"传统95%上限",
 "dynamic_lo":"动态95%下限","dynamic_hi":"动态95%上限","static_trim":"传统去掉最小规模","dynamic_trim":"动态去掉最小规模",
 "static_L1_exponent":"传统临界 L₁ 指数","dynamic_L1_exponent":"动态临界 L₁ 指数","S_static_at_pc":"临界传统 L₁/N",
 "S_dynamic_at_pc":"临界动态 L₁/N","speed":"速率倍数","time":"时间/时滞","S":"全在线起步 L₁/N",
 "active_edges":"全在线起步 有效超边比例","online":"全在线起步 在线比例","node_corr":"稳态节点自相关","edge_corr":"稳态超边自相关",
 "theory_active":"理论有效超边比例","theory_online":"理论在线比例","theory_nodecorr":"理论节点自相关","theory_edgecorr":"理论超边自相关",
 "event_count":"上线样本数","restored_edges":"平均恢复超边数","theory_restored":"理论恢复超边数","mean_gain":"平均 L₁ 增量",
 "P_r0":"合并前涉及0个分量的概率","P_r1":"涉及1个分量的概率","P_r_ge2":"涉及至少2个分量的概率","P_r_ge3":"涉及至少3个分量的概率",
 "P_gain_positive":"L₁ 增长概率","P_gain_ge10":"增量≥10 的概率","P_gain_ge100":"增量≥100 的概率","count_gain_ge100":"增量≥100 的事件数",
 "graph_id":"配对重复编号","edge_count":"超边数","mean_hyperdegree":"实际平均超度","excess_hyperdegree":"实际平均剩余超度",
 "incidence_four_cycles":"关联图四环数","four_cycle_vertex_fraction":"四环所涉原节点比例",
}
HALF=("S:L₁/N edges:有效超边 online:在线比例 nodecorr:节点自相关 edgecorr:超边自相关 "
      "restored:恢复数 r0:r=0 r1:r=1 r2:r≥2 r3:r≥3 gain:L₁增量 pos:增量>0 ge10:增量≥10 ge100:增量≥100")
NAMES.update({a+"_half":b+" 95%半宽" for a,b in (pair.split(":") for pair in HALF.split())})


def run():
    with xlsxwriter.Workbook(ROOT/"hypergraph_results.xlsx",{"constant_memory":True}) as book:
        head=book.add_format({"bold":True,"bg_color":"#19435F","font_color":"white","text_wrap":True,"valign":"vcenter"})
        fmt=book.add_format({"num_format":"0.000000"})
        notes=book.add_worksheet("实验说明")
        notes.set_column(0,0,20); notes.set_column(1,1,112)
        for i,row in enumerate([
            ("项目","统计口径"),
            ("模型","局部树状的均匀随机超图。所有成员在线时超边才有效；任一成员下线时整条超边失效。"),
            ("符号","h 是超边成员数；z 是每个节点参与的平均超边数。L₁、L₂ 按原节点数计，分母 N 为全部节点。"),
            ("主实验","4 种超边大小 × 3 种平均超度 × 5 种规模 × 10 张底图。每张图每种机制取 8 个快照。"),
            ("动态","μ=0.01，λ=μp/(1−p)。从全在线开始，在 10τ 首次观测，此后每隔 3τ。τ=1/(λ+μ)。"),
            ("误差","先在每张图内平均，再对 10 张图平均。相对误差为两组均值差的绝对值除以传统均值（百分数）。"),
            ("区间","稳态、事件与时间结果为图间逐点 95% t 区间。峰位和外推采用 2,500 次配对整图 bootstrap 百分位区间。"),
            ("有限尺度","取平均 L₂ 的峰位。以 N^(−1/3) 作线性外推，并另给去除 N=2000 后的结果。"),
            ("附加仿真","时间实验固定 p*=0.65；单次上线实验按稳态上线事件的条件分布直接采样。两个附加实验固定 N=4000、z=4。"),
        ]): notes.write_row(i,0,row,head if i==0 else None)
        for title,name in SHEETS:
            path=ROOT/name
            op=gzip.open if name.endswith(".gz") else open
            with op(path,"rt",encoding="utf-8-sig",newline="") as inp:
                rows=csv.reader(inp); cols=next(rows)
                sh=book.add_worksheet(title)
                sh.freeze_panes(1,min(3,len(cols)))
                sh.set_row(0,34)
                sh.write_row(0,0,[NAMES[x] for x in cols],head)
                for j,c in enumerate(cols):
                    sh.set_column(j,j,max(13,min(32,len(NAMES[c])*2+4)),None if c in ("h","z","N","graph_id","event_count","edge_count","speed","time","count_gain_ge100","incidence_four_cycles") else fmt)
                for i,row in enumerate(rows,1): sh.write_row(i,0,[float(v) for v in row])
                sh.autofilter(0,0,i,len(cols)-1)


if __name__=="__main__":
    run()
