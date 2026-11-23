# 网络节点上下线仿真 · 图 5–8

使用 Python 3.12。解压后，在本目录运行以下两条命令，即可使用包内数据重绘四张图。输出在 figures/，包括 PDF、SVG 和 600 dpi PNG。

```sh
python -m pip install -r requirements.txt
python plots.py
```

## 从头生成数据

在 Linux 上使用支持 C++17 的 g++ 编译连续时间生灭过程采样器，然后运行仿真。随机种子写在各脚本中。

```sh
g++ -O3 -std=c++17 -shared -fPIC events.cpp -o events.so
python size.py
python topology.py
python plots.py
```

## 图 5 的初始在线比例对照

运行 `python fig5.py` 可单独生成图 5。图中 M 是固定的总节点数。六个子图沿用原论文的速率组合。每幅保留 2,000、4,000、6,000 个节点全在线起步的仿真和理论曲线，同时叠加两条低初值曲线：总数 4,000、初始在线 400 的紫红色仿真轨迹和深紫色点线理论均值。橙红色和紫红色实线可用于比较同一网络规模的两种初值。每条仿真曲线展示固定种子下 20 次独立运行的第一条轨迹。给定初始在线数 n 和稳态在线比例 p = λ/(λ+μ)，理论均值为 Mp + (n − Mp) exp[−(λ+μ)t]。图中使用对数时间轴，从 t = 0.1 显示。

## 图 6–8 的数据

size.py 生成图 6 的占用时间概率和图 7 的稳态窗口平均值，保存为 six.npy 与 seven.npy。topology.py 生成图 8 的 BA 和 WS 网络样本及平均度分布，保存为 eight.npz。
