# 叠加与反卷积：算法的来历、原理与最早的实现

本仓库后期流程中最核心的两步是**叠加**（`stack.py`、`sigma_clip_mean`）和**反卷积**（`deconvolve.py`）。
本文回答几个问题：为什么要叠加？为什么平均能降噪？剔除离群值的想法从哪里来？反卷积最早用在什么领域，用什么方法、什么函数实现？

文中的数值结论都可以用演示脚本复现（只依赖 numpy）：

```bash
python docs/demos/stack_deconv_history_demo.py
```

相关文档：[数组计算历史](array_computing_history.md) · [MATLAB/NumPy/Wolfram 图像处理对比](matlab_numpy_wolfram_image_processing.md)

- [第一部分：叠加](#第一部分叠加)
  - [1. 平均的思想来自天文观测](#1-平均的思想来自天文观测)
  - [2. 为什么平均能降噪：√N 定律](#2-为什么平均能降噪n-定律)
  - [3. 离群值剔除：从 Peirce 准则到 σ-clip](#3-离群值剔除从-peirce-准则到-σ-clip)
  - [4. 从照相底片到 CCD 再到业余软件](#4-从照相底片到-ccd-再到业余软件)
  - [5. 实验结果](#5-实验结果)
- [第二部分：反卷积](#第二部分反卷积)
  - [6. 问题是什么：卷积模型与病态性](#6-问题是什么卷积模型与病态性)
  - [7. 最早的反卷积：光谱学和统计修正](#7-最早的反卷积光谱学和统计修正)
  - [8. 年表：从手算到神经网络](#8-年表从手算到神经网络)
  - [9. 各方法的公式与代码](#9-各方法的公式与代码)
  - [10. 实验结果](#10-实验结果)
  - [11. 软件实现史：函数是怎么来的](#11-软件实现史函数是怎么来的)
- [参考文献](#参考文献)

---

# 第一部分：叠加

## 1. 平均的思想来自天文观测

「多次测量取平均」在今天看来理所当然，但它是 18 世纪天文学家争论出来的。早期天文学家倾向于挑出「最好的一次观测」，
而不是取平均，理由是混入差的观测会拖累结果。

| 年代 | 人物 | 贡献 |
|---|---|---|
| 1722 | Roger Cotes | 提出按观测质量对多次观测**加权平均** |
| 1750 | Tobias Mayer | 研究月球天平动时，把 27 个方程分组相加求解（「平均法」），比挑选少量观测得到的结果更准 |
| 1755 | **Thomas Simpson** | 论文《On the advantage of taking the mean of a number of observations in practical astronomy》，**首次用概率论论证取平均比单次观测更好**。题目里就写着「实用天文学」 |
| 1757 | Roger Boscovich | 最小绝对偏差法，与中值相关 |
| 1774 | Laplace | 研究误差分布，提出中值是某些误差分布下的最优估计 |
| 1805 | Legendre | 发表**最小二乘法**（用于彗星轨道） |
| 1809 | **Gauss** | 《天体运动论》：用正态误差分布论证最小二乘和算术平均；1801 年他据此找回了失踪的谷神星 |
| 1810 | Laplace | 中心极限定理：很多独立误差之和趋于正态分布 |
| 1861 | George Airy（英国皇家天文学家） | 《On the algebraical and numerical theory of errors of observations》，系统阐述误差理论和**按方差倒数加权** |

所以，**「叠加取平均」从诞生之日起就是天文学的工具**。今天的图像叠加，是把同一思想用到了每个像素上：每个像素都是对同一片天空的 N 次独立测量。

## 2. 为什么平均能降噪：√N 定律

设每帧在某个像素测得 `xᵢ = S + nᵢ`，其中 S 是真实信号，nᵢ 是独立的噪声，方差为 σ²。

```
求和： Σxᵢ = N·S + Σnᵢ
信号按 N 线性增长；独立噪声的方差相加：Var(Σnᵢ) = N·σ²，所以噪声标准差只按 √N 增长

平均： x̄ = S + (Σnᵢ)/N,   std(x̄) = σ/√N

信噪比： SNR_N = N·S / (√N·σ) = √N · SNR_1
```

**关键在于「独立」**：信号在每帧都相同（相关），所以线性叠加；噪声每帧不同（不相关），所以只按平方和开根号叠加。

天文图像中的噪声来源：

| 噪声 | 规律 | 叠加时的表现 |
|---|---|---|
| **光子噪声（散粒噪声）** | 泊松分布（Poisson 1837），方差 = 期望光子数 S，即 σ = √S | 叠加 N 帧等于收集 N 倍光子：SNR = √(N·S)，完全等价于一张 N 倍时长的曝光 |
| 天空背景噪声 | 也是泊松噪声，σ = √(天光) | 同上，光污染越重，需要的总曝光越长 |
| **读出噪声** | 每读出一次固定一份，σ_read | N 帧带来 N 份：总方差 N·σ_read²。这就是「单张曝光不能太短」的原因 |
| 暗电流 | 泊松噪声，随时间和温度增长 | 靠制冷和暗场减除 |
| 固定图案噪声（热像素、条纹） | **不独立**，每帧相同 | 平均**不能**消除，需要暗场/偏置场，或抖动拍摄（dither）后剔除 |

叠加 N 帧与单张 N 倍曝光的总噪声对比（S 为每帧信号光子数，B 为每帧天光光子数）：

```
单张 N·t：  σ² = N·S + N·B + σ_read²
叠加 N 张： σ² = N·S + N·B + N·σ_read²
```

只要 `B ≫ σ_read²`（「天光主导」），两者几乎相同。叠加还有单张长曝光没有的好处：不怕饱和、可以剔除卫星和宇宙线、可以丢掉坏帧、
对跟踪精度要求低。这就是现代深空摄影都采用「多张短曝光再叠加」的原因。

**为什么用均值而不用中值**：对正态噪声，中值的效率只有均值的 2/π ≈ 64%，噪声大约是均值的 √(π/2) ≈ **1.25 倍**。
相当于浪费了 36% 的曝光时间。所以现代软件使用「剔除离群值后求均值」，兼顾两者的优点。

**加权平均**：各帧质量不同（视宁度、透明度、天光）时，按方差倒数加权可以得到最优的信噪比（Gauss、Airy）。
本仓库 `stack.py` 用 1/FWHM² 作权重，`compose.py` 合成亮度时用 `L/σ_L² + RGB/σ_RGB²`，都是这个思想。

## 3. 离群值剔除：从 Peirce 准则到 σ-clip

取平均的最大弱点是一个极端值就能把结果带偏。卫星划过一帧时亮度 +2000 ADU，10 帧平均后仍有 +200。所以需要先剔除离群值。

| 年代 | 方法 | 出处 |
|---|---|---|
| 1852 | **Peirce 准则** | Benjamin Peirce（哈佛天文学家、数学家），在《天文学杂志》（*Astronomical Journal*）上提出第一个剔除可疑观测的严格统计判据 |
| 1863 | **Chauvenet 准则** | William Chauvenet 在《球面与实用天文学手册》中提出：偏离均值的概率乘以 N 小于 1/2 的数据点予以剔除 |
| 1950s–60s | Winsorizing | Charles Winsor 提出、Tukey 推广：把极端值替换为分位数处的值，而不是直接丢弃（PixInsight 的 Winsorized Sigma Clipping 由此得名） |
| 1964 | M 估计 | Peter Huber，鲁棒统计学的开端 |
| 1974 | **MAD 与影响函数** | Frank Hampel：用中值和 MAD 估计尺度，MAD × 1.4826 是正态分布 σ 的无偏估计 |
| 1980s | **σ-clip 叠加** | IRAF `imcombine`（NOAO）提供 `avsigclip`、`ccdclip`（利用 CCD 的增益和读出噪声建立噪声模型）、`minmax`、`crreject` 等剔除算法 |
| 1990 年代起 | Hubble `CR-SPLIT` | 哈勃望远镜把一次观测拆成至少两次曝光，专门用于剔除宇宙线 |
| 2000 年代起 | Linear Fit Clipping、ESD 等 | PixInsight ImageIntegration 等软件提供更多剔除算法 |

σ-clip 的流程（`astrotools.sigma_clip_mean`）：

```
对每个像素的 N 个值：
  1. 中心 m = 中值（不受离群值影响）
  2. 尺度 s = 1.4826 × MAD（同样鲁棒）
  3. 剔除 |x − m| > k·s 的值（k 通常取 2.5–3）
  4. 重复 1–3 若干次，对剩余值求（加权）平均
```

## 4. 从照相底片到 CCD 再到业余软件

| 年代 | 事件 |
|---|---|
| 1880 | Henry Draper 拍摄猎户座大星云 M42，第一张星云照片。**照相底片本身就是一个积分器**：长时间曝光让光子在底片上累积 |
| 1883 | Andrew Ainslie Common 用 60 分钟曝光拍出 M42 的细节，证明照相比人眼看得更深 |
| 1970s–80s | **David Malin**（英澳天文台）把多张底片对齐后叠加印放，再配合非锐化遮罩，揭示了极暗的星系潮汐结构。这是模拟时代的「叠加」 |
| 1969 | **CCD** 发明（Boyle & Smith，贝尔实验室；2009 年获诺贝尔物理学奖） |
| 1976 | 第一批天文 CCD 图像（JPL 与亚利桑那大学合作，用 61 英寸望远镜拍摄天王星等） |
| 1970 | Antoine Labeyrie 提出**散斑干涉**：用大量极短曝光冻结大气湍流 |
| 1980 | Bates & Cady 提出 **shift-and-add**：每帧按最亮散斑对齐后叠加，是「幸运成像」和行星视频叠加的前身 |
| 1978 | David Fried 计算出短曝光恰好遇到良好视宁度的概率，奠定了**幸运成像**的理论基础 |
| 1980s–90s | IRAF `imcombine` 成为专业天文叠加的标准工具 |
| 1995 | **哈勃深场**：342 张曝光合成。为此开发了 **Drizzle** 算法（Fruchter & Hook 2002），在叠加的同时提高分辨率 |
| 约 2002 | RegiStax（Cor Berrevoets）：业余行星视频叠加 |
| 2000 年代中期 | DeepSkyStacker（Luc Coiffier）、PixInsight ImageIntegration：业余深空叠加 |
| 约 2010 | AutoStakkert!（Emil Kraaikamp）：行星幸运成像 |
| 2010s | Siril（开源）、Sequator（银河星野），以及本仓库 `milkyway/` 模仿的思路 |

## 5. 实验结果

`python docs/demos/stack_deconv_history_demo.py` 第一部分的输出（σ = 10）：

```
   N  mean noise  sigma/sqrtN  median noise  median/mean
   1      10.028       10.000        10.028        1.000
   4       5.002        5.000         5.461        1.092
  16       2.498        2.500         3.003        1.202
  64       1.251        1.250         1.550        1.239
  theory for large N: median/mean = sqrt(pi/2) = 1.2533
  Poisson: 1 frame SNR 7.09 (sqrt(S)=7.07); 16 frames SNR 28.29 (x3.99)
  N=10, 5% pixels hit by a 2000-ADU cosmic ray in one frame:
    mean  rms error   44.94
    median rms error    3.74
    sigma-clip mean    3.47   (ideal sigma/sqrtN = 3.16)
```

结论：
1. 均值叠加的噪声严格按 σ/√N 下降；
2. 中值叠加的噪声比均值大，N 越大越接近 1.25 倍；
3. 泊松噪声：叠加 16 帧，信噪比提高 √16 = 4 倍；
4. 有离群值时，纯均值完全失效（误差 44.9），中值稳健但噪声偏大（3.74），**σ-clip 均值最好**（3.47，接近理论极限 3.16）。

---

# 第二部分：反卷积

## 6. 问题是什么：卷积模型与病态性

望远镜加大气让每颗星都变成一个模糊的光斑，这个光斑就是**点扩散函数（PSF）**。观测图像是真实天空与 PSF 的卷积，再加上噪声：

```
d(x) = (h ⊗ u)(x) + n(x)          d：观测   u：真实天空   h：PSF   n：噪声
傅里叶域：  D(f) = H(f)·U(f) + N(f)          （卷积定理）
```

最直接的解法是除回去，即「逆滤波」：`U = D / H`。但高斯 PSF 的 H(f) 在高频处趋近于 0，于是 `N(f)/H(f)` 被放大到天文数字。
本文实验中，逆滤波结果的 rms 误差达到 **2×10¹⁰**。

这类「解存在但对噪声极其敏感」的问题称为**病态问题**（Hadamard 1902/1923 年定义的 ill-posed problem）。
反卷积一百年来的历史，就是**如何在恢复细节和放大噪声之间找平衡**的历史：
加先验（维纳的信噪比、Tikhonov 的平滑性、RL 的非负性与泊松统计、最大熵、CLEAN 的点源假设），或者提前停止迭代。

## 7. 最早的反卷积：光谱学和统计修正

**反卷积最早并不是用在图像上，而是用在一维的光谱和统计分布上**：

1. **Eddington（1913）**：在研究恒星统计时，测量误差会把真实分布「抹平」。Eddington 在《皇家天文学会月刊》（*MNRAS*）上给出了修正公式：
   `f_true ≈ f_obs − (σ²/2)·f_obs″`，即减去观测分布的二阶导数。本质上是**一阶近似的反卷积**，「Eddington 偏差」一词即源于此。
2. **van Cittert（1931）**：光谱仪的狭缝和光学系统让谱线变宽，即真实谱线与「仪器轮廓」的卷积。
   P. H. van Cittert 在《物理学杂志》（*Zeitschrift für Physik*）上提出迭代修正：
   `u_{k+1} = u_k + (d − h ⊗ u_k)`。当时是**用手算和计算尺**完成的，这是第一个迭代反卷积算法。
3. **Bracewell & Roberts（1954）**：射电望远镜的天线方向图很宽，观测到的射电天图是真实天图与天线方向图的卷积。
   他们在《Aerial smoothing in radio astronomy》中分析了哪些频率成分可以恢复（「主解」），提出了**可恢复性的频率极限**。
4. **反射地震学（1950s）**：MIT 地球物理分析组的 Enders Robinson 在 1954 年的博士论文中提出**预测反卷积**，
   用来去掉地震波在地层中的多次反射。石油勘探工业是最早**大规模用数字计算机做反卷积**的领域，
   「deconvolution」一词正是在地球物理界流行起来的。

## 8. 年表：从手算到神经网络

| 年代 | 方法 / 事件 | 领域 | 要点 |
|---|---|---|---|
| 1807/1822 | Fourier 级数与变换 | 热传导 | 卷积定理的数学基础 |
| 1913 | **Eddington 修正** | 恒星统计 | 一阶近似的反卷积 |
| 1931 | **van Cittert 迭代** | 光谱学 | 第一个迭代反卷积，手算 |
| 1941–49 | **维纳滤波**（Kolmogorov 1941，Wiener 1942/1949） | 防空火控 | 二战高射炮预测的保密报告（因封面颜色被称为「黄祸」），1949 年公开出版；最小均方误差意义下的最优线性滤波 |
| 1954 | Bracewell & Roberts | 射电天文 | 天线平滑与可恢复性 |
| 1954 | Robinson 预测反卷积 | 地震勘探 | 数字计算机上的大规模反卷积 |
| 1962/1963 | **Tikhonov 正则化**（Phillips 1962，Tikhonov 1963） | 数学物理 | 求解病态问题的通用方法：加上平滑惩罚项 |
| 1965 | Cooley–Tukey FFT | 计算 | 让频域反卷积在计算机上可行 |
| 1966 起 | **JPL 图像处理实验室**（Robert Nathan 等） | 空间探测 | 为徘徊者号、勘测者号、水手号的图像做去模糊和去噪；VICAR 软件（Fortran）。**数字图像复原诞生于此** |
| 1966–67 | Harris、Helstrom | 光学 | 维纳滤波用于二维图像复原 |
| 1972 | **Richardson** | 光学 | 《Bayesian-based iterative method of image restoration》（*JOSA*） |
| 1972 | Frieden | 光学 | **最大熵**图像复原 |
| 1974 | **Lucy** | 天文 | 《An iterative technique for the rectification of observed distributions》（*AJ*），与 Richardson 的方法相同，两人独立提出 |
| 1974 | **CLEAN**（Högbom） | 射电干涉 | 反复在残差图峰值处减去 PSF（「脏束」），记录点源分量；至今仍是射电成像的标准算法（AIPS → CASA `tclean`） |
| 1975 | **盲反卷积**（Stockham、Cannon & Ingebretsen） | 音频 | 同时估计模糊核与信号，修复了 1900 年代**卡鲁索**的唱片录音 |
| 1977 | EM 算法（Dempster、Laird & Rubin） | 统计 | |
| 1977 | 《Digital Image Restoration》（Andrews & Hunt） | 图像处理 | 第一本图像复原教科书 |
| 1978 | Gull & Daniell（*Nature*） | 射电天文 | 最大熵用于射电图像重建 |
| 1982 | Shepp & Vardi | 医学 PET | 从 EM 推导出 **ML-EM**，与 RL 是**同一个算法**；医学成像中称为 MLEM/OSEM |
| 1984 | Skilling & Bryan | 天文 | 实用的最大熵算法 MEMSYS |
| 1988 | Ayers & Dainty | 光学 | 迭代盲反卷积 |
| **1990** | **哈勃球差** | 天文 | 主镜磨错了约 2.2 μm，图像严重模糊。在 1993 年 12 月 COSTAR 修复之前，**RL 成为处理 HST 图像的主力**；STScI 召开了「HST 图像与光谱复原」研讨会（1990、1993） |
| 1994 | 阻尼 RL（Richard White，STScI） | 天文 | 在噪声水平附近抑制迭代，减少噪声放大 |
| 1997 | 加速 RL（Biggs & Andrews） | 光学 | 向量外推加速收敛；**MATLAB 的 `deconvlucy` 实现的就是 Biggs–Andrews 加速与 White 阻尼** |
| 1990s–2000s | 多尺度 / 小波正则化（Starck & Murtagh 等） | 天文 | 在小波域约束噪声；PixInsight Deconvolution 的小波正则化和去振铃由此而来 |
| 2022 | BlurXTerminator（Russell Croman） | 业余天文 | 神经网络反卷积，可以校正随视场变化的 PSF |

## 9. 各方法的公式与代码

以下代码均为 numpy 实现，摘自 `docs/demos/stack_deconv_history_demo.py`。`H` 是 PSF 的傅里叶变换，`conv(x, H)` 是频域卷积。

### 9.1 逆滤波：U = D / H

```python
u = np.fft.irfft(np.fft.rfft(data) / np.maximum(H, 1e-12), n=n)
```

在高频处被 H ≈ 0 除，噪声爆炸。这是所有反卷积方法要解决的问题。

### 9.2 van Cittert（1931）

```
u_{k+1} = u_k + (d − h ⊗ u_k)
```

```python
u = data.copy()
for _ in range(iters):
    u = u + (data - conv(u, H))
```

频域中，迭代 k 次等价于用 `1 − (1 − H)^k` 去逼近 `1/H`。k 小时只恢复中低频，k → ∞ 时就退化为逆滤波。
所以它**必须提前停止**，而且结果会出现负值。PixInsight 的 Deconvolution 中至今还保留着 van Cittert 选项。

### 9.3 维纳滤波（1942/1949）

```
U = D · H* / (|H|² + N/S)
```

```python
u = np.fft.irfft(np.fft.rfft(data) * H / (H**2 + nsr), n=n)
```

在信噪比高的频率上接近逆滤波，在信噪比低的频率上趋于 0。一步算完，不需要迭代，但需要估计噪信比 N/S，结果可能为负。

### 9.4 Tikhonov 正则化（1963）

```
u = argmin ‖h ⊗ u − d‖² + λ‖∇u‖²    →    U = D · H* / (|H|² + λ|∇|²)
```

```python
lap = (2*np.sin(np.pi*np.fft.rfftfreq(n)))**2          # 离散梯度算子的频率响应
u = np.fft.irfft(np.fft.rfft(data) * H / (H**2 + lam*lap), n=n)
```

与维纳滤波形式相同，只是用「平滑性」代替「信噪比」作为先验。现代的全变分（TV）正则化是它的推广。

### 9.5 Richardson–Lucy（1972/1974）

```
u_{k+1} = u_k · [ hᵀ ⊗ ( d / (h ⊗ u_k) ) ]
```

推导思路：光子计数服从泊松分布，在泊松似然下求最大似然解（即 EM 算法）。性质：
- **始终非负**：乘法更新，只要初值为正，结果永远为正；
- **总通量守恒**；
- **天然适合光子计数数据**，这正是天文 CCD 数据的统计特性；
- 收敛慢，迭代太多时会放大噪声并产生振铃，需要提前停止、加阻尼或正则化。

```python
u = np.full(n, data.mean())
for _ in range(iters):
    u *= conv(data / np.maximum(conv(u, H), 1e-9), H)     # 高斯 PSF 对称，hᵀ = h
```

本仓库的 `scripts/deconvolve.py`（二维 FFT 版加亮度遮罩和去振铃）和 `wolfram/AstroTools.wl` 中的 `Deconvolve`（内置 `ImageDeconvolve`）都使用 RL。

### 9.6 CLEAN（Högbom 1974）

```
重复：
  1. 找到残差图的峰值位置 k 和峰值 r_k
  2. 记录点源分量 comps[k] += g·r_k        （g 是增益，通常为 0.1）
  3. 残差图减去 g·r_k × PSF（以 k 为中心）
直到残差低于噪声水平；最后把点源分量与一个较窄的「clean beam」卷积，再加回残差
```

```python
for _ in range(max_iter):
    k = np.argmax(res)
    if res[k] < threshold:
        break
    comps[k] += gain * res[k]
    res -= gain * res[k] * np.roll(psf, k - n//2)
```

它假设天空由点源组成，这一点特别适合射电干涉：干涉仪的 PSF（脏束）有很强的旁瓣。本质上它是一种贪心的稀疏重建，
比压缩感知理论（2006 年）早了三十多年。

### 9.7 最大熵（1972/1978/1984）

```
u = argmax  −Σ u·log(u/m)    满足  χ²(d, h⊗u) ≈ N
```

在所有与数据吻合的解中，选「信息最少」（最平滑、最不武断）的那个，并且自动保证非负。
1980–90 年代在射电天文（AIPS 的 VM 任务）和 HST 图像处理中广泛使用。

## 10. 实验结果

一维测试信号：两颗相距 8 px 的星（亮度 100 和 70）加一片暗弱星云。用 FWHM = 10 px 的高斯 PSF 模糊，加上泊松噪声。
模糊后两颗星完全融为一团。表中的「谷/峰」是两峰之间的最低值除以较弱的峰值，小于 0.8 视为分开；「最小值」为负说明出现了振铃。

```
  observed (blurred)           rms       5.05   valley/peak         1   min      0.92  blended / broken
  inverse filter 1/H           rms   1.95e+10   valley/peak     -1.69   min -8.56e+10  blended / broken
  van Cittert (10 it)          rms       4.98   valley/peak     0.696   min     -1.64  RESOLVED
  van Cittert (200 it)         rms        7.8   valley/peak    -0.781   min     -28.9  blended / broken
  Wiener (NSR=1e-4)            rms       4.78   valley/peak     0.399   min     -4.82  RESOLVED
  Tikhonov (gradient, 1e-4)    rms       4.76   valley/peak     0.311   min     -5.18  RESOLVED
  Richardson-Lucy (50 it)      rms       4.92   valley/peak         1   min     0.396  blended / broken
  Richardson-Lucy (1000 it)    rms       4.54   valley/peak     0.393   min    0.0639  RESOLVED
  CLEAN (gain 0.1, beam 3px)   rms       5.18   valley/peak     0.384   min     0.792  RESOLVED
```

这张表几乎就是一部浓缩的反卷积史：

| 方法 | 结果 | 对应的历史教训 |
|---|---|---|
| 逆滤波 | 误差 10¹⁰，彻底失败 | 病态问题必须加约束 |
| van Cittert | 10 次迭代能分开双星；200 次发散，出现大负值 | 迭代法要**提前停止** |
| 维纳 / Tikhonov | 能分开，但有负值（振铃） | 线性方法无法保证非负 |
| **Richardson–Lucy** | 误差最小、**始终非负**，但需要很多次迭代 | 这就是天文界 50 年来偏爱 RL 的原因 |
| CLEAN | 能分开，非负 | 点源假设对恒星非常有效（射电界的标准方法） |

## 11. 软件实现史：函数是怎么来的

| 年代 | 软件 | 函数 / 任务 | 语言 | 说明 |
|---|---|---|---|---|
| 1931 | — | 手算 | 计算尺 | van Cittert |
| 1950s | 石油公司地震处理系统 | 预测反卷积 | 汇编 / Fortran | 早期数字计算机的大规模应用 |
| 1966 起 | JPL **VICAR** | 维纳滤波、去模糊 | Fortran（IBM 360 等） | 空间探测图像 |
| 1978 起 | NRAO **AIPS** | `APCLN`（CLEAN）、`VM`（最大熵） | Fortran（POPS 命令语言） | 射电天文 |
| 1980s–90s | IRAF / **STSDAS** | `lucy`、`mem0`、`wiener` | SPP / Fortran | HST 球差时期的主力 |
| 1990s | **IDL Astronomy User's Library** | `max_likelihood.pro`、`max_entropy.pro`（Frank Varosi 等） | IDL | |
| 约 2000 年前后 | **MATLAB IPT** | `deconvlucy`、`deconvwnr`、`deconvreg`、`deconvblind` | MATLAB / C | `deconvlucy` 实现了 Biggs–Andrews 加速与 White 阻尼，带 `WEIGHT`（屏蔽坏像素）和 `DAMPAR`（阻尼）参数 |
| 2000s | **PixInsight** | Deconvolution（RL、van Cittert + 小波正则化 + 去振铃）、DynamicPSF | C++ | 业余天文 |
| 2008–2010 | **Mathematica** 7/8 | `ImageDeconvolve`（`"RichardsonLucy"`、`"Wiener"`、`"Tikhonov"`、`"TotalVariation"` 等方法） | Wolfram | |
| 2010s | **scikit-image** | `restoration.richardson_lucy`、`wiener`、`unsupervised_wiener` | Python / NumPy | |
| 2010s | **CASA** | `tclean`（多尺度 CLEAN 等） | C++ / Python | 射电天文 |
| 2022 | **BlurXTerminator** | 神经网络 | PixInsight 插件 | 能处理随视场变化的 PSF |
| 本仓库 | `scripts/deconvolve.py` | FFT 实现的 RL + 亮度遮罩 + 去振铃 | NumPy | 原理与上面的实现相同 |

同一个算法在不同时代的写法对照：

```fortran
C     1960s-80s 风格：Fortran 77，一维 RL 的一次迭代（显式循环卷积）
      DO 20 I = 1, N
         S = 0.0
         DO 10 J = 1, M
            S = S + H(J) * U(I - J + M/2)
   10    CONTINUE
         R(I) = D(I) / MAX(S, 1.0E-9)
   20 CONTINUE
C     ... 再把 R 与 H 卷积，乘回 U
```

```matlab
% 2000s：MATLAB，一行调用
J = deconvlucy(I, fspecial('gaussian', 25, fwhm/2.3548), 20);
```

```python
# 2010s：scikit-image
J = skimage.restoration.richardson_lucy(I, psf, num_iter=20)
```

```mathematica
(* Wolfram *)
ImageDeconvolve[img, GaussianMatrix[{12, fwhm/2.3548}], Method -> "RichardsonLucy", MaxIterations -> 20]
```

---

## 参考文献

**叠加与误差理论**
- Simpson, T. 1755, *Phil. Trans. R. Soc.* 49, 82：On the advantage of taking the mean of a number of observations in practical astronomy
- Gauss, C. F. 1809, *Theoria Motus Corporum Coelestium*
- Airy, G. B. 1861, *On the Algebraical and Numerical Theory of Errors of Observations*
- Peirce, B. 1852, *Astronomical Journal* 2, 161；Chauvenet, W. 1863, *A Manual of Spherical and Practical Astronomy*
- Huber, P. J. 1964, *Ann. Math. Stat.* 35, 73；Hampel, F. R. 1974, *JASA* 69, 383
- Stigler, S. M. 1986, *The History of Statistics: The Measurement of Uncertainty before 1900*（平均法历史的权威著作）
- Labeyrie, A. 1970, *A&A* 6, 85；Bates, R. H. T. & Cady, F. M. 1980, *Optics Communications* 32, 365；Fried, D. L. 1978, *JOSA* 68, 1651
- Fruchter, A. S. & Hook, R. N. 2002, *PASP* 114, 144（Drizzle）；Williams, R. E. et al. 1996, *AJ* 112, 1335（哈勃深场）

**反卷积**
- Eddington, A. S. 1913, *MNRAS* 73, 359
- van Cittert, P. H. 1931, *Zeitschrift für Physik* 69, 298
- Wiener, N. 1949, *Extrapolation, Interpolation, and Smoothing of Stationary Time Series*（1942 年保密报告）
- Bracewell, R. N. & Roberts, J. A. 1954, *Australian J. Phys.* 7, 615
- Robinson, E. A. 1954, MIT 博士论文（预测反卷积）
- Phillips, D. L. 1962, *J. ACM* 9, 84；Tikhonov, A. N. 1963, *Soviet Math. Dokl.* 4, 1035
- Richardson, W. H. 1972, *JOSA* 62, 55；Lucy, L. B. 1974, *AJ* 79, 745
- Frieden, B. R. 1972, *JOSA* 62, 511；Gull, S. F. & Daniell, G. J. 1978, *Nature* 272, 686；Skilling, J. & Bryan, R. K. 1984, *MNRAS* 211, 111
- Högbom, J. A. 1974, *A&AS* 15, 417（CLEAN）
- Stockham, T. G., Cannon, T. M. & Ingebretsen, R. B. 1975, *Proc. IEEE* 63, 678（盲反卷积，卡鲁索录音）
- Andrews, H. C. & Hunt, B. R. 1977, *Digital Image Restoration*
- Shepp, L. A. & Vardi, Y. 1982, *IEEE Trans. Med. Imaging* 1, 113
- Ayers, G. R. & Dainty, J. C. 1988, *Optics Letters* 13, 547
- White, R. L. 1994, in *The Restoration of HST Images and Spectra II*（阻尼 RL）
- Biggs, D. S. C. & Andrews, M. 1997, *Applied Optics* 36, 1766（加速 RL）
- Starck, J.-L., Pantin, E. & Murtagh, F. 2002, *PASP* 114, 1051（天文图像反卷积综述）
