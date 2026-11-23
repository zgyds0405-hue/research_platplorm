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

图 5 的 20 条规模轨迹在 plots.py 中按连续时间马尔可夫链的精确转移概率采样，显示第一条及理论均值。size.py 生成图 6 的占用时间概率和图 7 的稳态窗口平均值，保存为 six.npy 与 seven.npy。topology.py 生成图 8 的 BA 和 WS 网络样本及平均度分布，保存为 eight.npz。
