# 文档索引

| 文档 | 内容 |
|---|---|
| [array_computing_history.md](array_computing_history.md) | 数组计算史：Fortran 数值库 → APL → MATLAB / IDL / Mathematica → Numeric / Numarray → NumPy；NumPy 函数名、设计约定、图像函数的出处；本仓库天文算法的原始论文 |
| [stacking_deconvolution_history.md](stacking_deconvolution_history.md) | 叠加与反卷积的来历：为什么平均能降噪（√N）、离群值剔除的起源、反卷积最早用在哪里、各方法的公式与代码、软件实现史 |
| [calibration_frames_history.md](calibration_frames_history.md) | 暗场 / 平场 / 偏置场的来历：探测器信号模型、为什么减与除、从底片 → 光电测光 → 电视管 → CCD（IRAF ccdproc）→ 业余 CMOS 的演变、专业与业余的差异、常见误区 |
| [deepsky_postprocessing_pixinsight.md](deepsky_postprocessing_pixinsight.md) | 深空后期流程篇：为什么要 DBE、PixInsight 全流程（预处理 → 线性 → 拉伸 → 非线性）、发射星云 / SHO·HOO / 星系分别怎么处理、工具演变时间线、常见错误 |
| [matlab_numpy_wolfram_image_processing.md](matlab_numpy_wolfram_image_processing.md) | MATLAB vs NumPy vs Wolfram 图像处理详细对比：类型约定、坐标系、默认参数差异、13 类操作的三语言代码 |
| [numpy_vs_wolfram_image_processing.md](numpy_vs_wolfram_image_processing.md) | NumPy vs Wolfram 语言基础对比：数据模型、索引、广播与线程化、归约、速查表 |
| [demos/stack_deconv_history_demo.py](demos/stack_deconv_history_demo.py) | 可运行的数值实验（只依赖 numpy）：√N 定律、中值效率、σ-clip；逆滤波、van Cittert、维纳、Tikhonov、RL、CLEAN 对比 |

建议阅读顺序：校准帧 → 后期流程 → 历史篇 → 叠加与反卷积 → 三方对比 → NumPy/Wolfram 细节。
