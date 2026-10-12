# NumPy 与 Wolfram Language：图像处理对比学习

本文以本仓库的天文后期代码（`scripts/`、`process/` 用 Python，`wolfram/` 用 Wolfram）为例，逐项对比 NumPy 和
Wolfram Language（Mathematica）在数组与图像处理上的思路和写法，方便会其中一种的人快速掌握另一种。

- [1. 历史：数组编程的家谱](#1-历史数组编程的家谱)
- [2. 核心数据模型](#2-核心数据模型)
- [3. 索引与切片](#3-索引与切片)
- [4. 向量化：ufunc / 广播 vs Listable / 线程化](#4-向量化ufunc--广播-vs-listable--线程化)
- [5. 归约与轴](#5-归约与轴)
- [6. 图像的表示](#6-图像的表示)
- [7. 常用操作速查表](#7-常用操作速查表)
- [8. 实战对照：本仓库的算法](#8-实战对照本仓库的算法)
- [9. 性能与内存](#9-性能与内存)
- [10. 易错点清单](#10-易错点清单)

---

## 1. 历史：数组编程的家谱

> 更完整的历史（各数学软件简史、NumPy 每个函数名的出处）见 [array_computing_history.md](array_computing_history.md)；
> 加入 MATLAB 的三方图像处理对比见 [matlab_numpy_wolfram_image_processing.md](matlab_numpy_wolfram_image_processing.md)。

### 1.1 时间线

| 年份 | 事件 | 意义 |
|---|---|---|
| 1962 | Kenneth Iverson 出版《A Programming Language》（APL，1966 年实现） | **数组编程的鼻祖**：整个数组作为运算对象、归约、秩（rank）的概念 |
| 1977 | IDL（Interactive Data Language） | 长期是**天文界的标准工具**，哈勃、NASA 大量使用 |
| 1970s 末 / 1984 | MATLAB（Cleve Moler，最初是 Fortran 的 LINPACK/EISPACK 前端；1984 年 MathWorks 商业化） | 矩阵为中心、交互式；`zeros`、`ones`、`linspace`、`eye`、冒号切片等命名后来被 NumPy 大量沿用 |
| 1988 | Mathematica 1.0（Stephen Wolfram） | 符号计算 + 列表；`Listable` 属性让函数自动作用到列表的每个元素 |
| 1995 | Python **Numeric**（Jim Hugunin 等，Python matrix-sig） | Python 的第一个数组库，从一开始就有广播 |
| 1995 | PIL（Fredrik Lundh） | Python 的图像读写库 |
| 1999 | Mathematica 4 引入**打包数组（Packed Array）** | 同类型的数值连续存储在内存里，性能接近 C |
| 2000 | OpenCV（Intel） | C/C++ 计算机视觉库，后来提供 Python 绑定 |
| 2001 | **Numarray**（STScI：Perry Greenfield、Todd Miller、Rick White） | **空间望远镜研究所为处理哈勃大图开发**，目标之一就是替代 IDL；同时诞生了 PyFITS |
| 2001 | SciPy | `scipy.ndimage`、`scipy.fft`、`scipy.signal` 等 |
| 2006 | **NumPy 1.0**（Travis Oliphant 合并 Numeric 与 Numarray） | 统一了 Python 的数组生态 |
| 2008 | Mathematica 7 引入 `Image` 对象和成体系的图像处理函数 | Wolfram 的图像处理从这里开始成形 |
| 2009 | scikit-image | 基于 NumPy 的图像算法库 |
| 2011–2013 | Astropy | 整合 PyFITS 等天文工具，Python 成为天文主流语言 |

### 1.2 谁影响了谁

```
              APL (1962)
             /    |     \
         J/K   MATLAB   IDL ──────────┐  (天文界主力)
                  |  \                 │
                  |   \                │
       Mathematica │    Numeric (1995)  │
       (1988)      │        \           │
       Listable ···│········ Numarray (2001, STScI，替代 IDL 处理哈勃数据)
       Packed      │          /
       Array       └──► NumPy (2006) ◄──┘
```

- **有文献记载的直接来源**：MATLAB（函数命名、交互方式）、APL/J（数组整体运算、广播与秩的思想）、
  IDL/Yorick（科学数据处理习惯，尤其在天文领域）。NumPy 官方文档也有专门的《NumPy for MATLAB users》。
- **Mathematica 的位置**：同属数组编程家族，`Listable` 自动线程化与 ufunc 逐元素运算、打包数组与 ndarray
  连续内存，在思想上高度相似，彼此之间有交流和借鉴。但 NumPy 的设计文档和历史资料里，Mathematica 并不是主要的直接来源，
  它更像是「平行发展的近亲」。
- **天文的特殊作用**：Numarray 和 PyFITS 是 STScI 为哈勃数据做的。天文界从 IDL 迁移到 Python，
  是推动 NumPy 生态成熟的重要力量之一。本仓库用 Python 处理 FITS，也算是这条历史线的延续。

所以学习时可以这样理解：**NumPy ≈ MATLAB 的语义 + APL 的广播 + Python 的语言**；
**Wolfram ≈ 一切皆表达式 + Listable 自动线程化 + 打包数组 + 大量内置高阶函数**。

---

## 2. 核心数据模型

| | NumPy | Wolfram Language |
|---|---|---|
| 基本容器 | `ndarray`：固定 dtype、固定形状、连续或跨步内存 | `List`（嵌套）；全是同类机器数时自动成为**打包数组** |
| 元素类型 | `uint8/uint16/int32/float32/float64/complex64…`，显式指定 | 机器整数、机器实数（Real64）、复数；还有**精确数**（`1/2`）和任意精度数 |
| 形状 | `a.shape`、`a.ndim`、`a.size` | `Dimensions[a]`、`ArrayDepth[a]`、`Length[Flatten[a]]` |
| 32 位数组 | 原生支持 `float32` | 计算中一律是 Real64；`NumericArray[a, "Real32"]` 可以紧凑存储，但大多数运算会转回 Real64 |
| 赋值语义 | **引用**；切片返回**视图**（共享内存） | **值语义**（不可变 + 写时复制）；`b = a; b[[1]] = 0` 不影响 `a` |
| 缺失值 | `np.nan` + `nanmedian` / `nanmean` | `Missing[]`、`Indeterminate`，但会**破坏打包**、变慢；实践中改用 0/1 掩膜 |
| 图像 | 就是 ndarray（H×W 或 H×W×C） | 独立的 `Image` 对象（带类型、色彩空间、元数据），用 `ImageData` 与数组互转 |

```python
a = np.arange(10)
b = a[::2]       # 视图
b[0] = 99
a[0]             # 99 —— 原数组被改了！
```

```mathematica
a = Range[0, 9];
b = a[[;; ;; 2]];
b[[1]] = 99;
a[[1]]           (* 0 —— Wolfram 的值语义，a 不变 *)
```

**精确数陷阱（Wolfram 特有）**：`a/2` 结果是精确有理数，数组会因此解包、变成符号运算，慢上百倍。
写数值代码时用 `0.5`、`N[...]`，或者让数据一开始就是实数。本仓库 `ReadFITS` 中的 `N[raw]` 就是为此。

---

## 3. 索引与切片

| 操作 | NumPy（0 起始） | Wolfram（1 起始） | MATLAB（1 起始，参考） |
|---|---|---|---|
| 第一个元素 | `a[0]` | `a[[1]]` | `a(1)` |
| 最后一个 | `a[-1]` | `a[[-1]]` | `a(end)` |
| 第 i 行 j 列 | `a[i, j]` | `a[[i, j]]` | `a(i, j)` |
| 一行 / 一列 | `a[i]`、`a[:, j]` | `a[[i]]`、`a[[All, j]]` | `a(i,:)`、`a(:,j)` |
| 去掉首尾一圈 | `a[1:-1, 1:-1]` | `a[[2 ;; -2, 2 ;; -2]]` | `a(2:end-1, 2:end-1)` |
| 步长 2 | `a[::2, ::2]` | `a[[;; ;; 2, ;; ;; 2]]` | `a(1:2:end, 1:2:end)` |
| 区间 | `a[10:20]`（**不含** 20） | `a[[11 ;; 20]]`（**含**两端） | `a(11:20)` |
| 取若干下标 | `a[[3, 5, 7]]` | `a[[{4, 6, 8}]]` | `a([4 6 8])` |
| 布尔掩膜取值 | `a[mask]` | `Pick[a, mask]`（mask 为 True/False 或配合 0/1 用 `Pick[a, m, 1]`） | `a(mask)` |
| 掩膜赋值 | `a[mask] = 0` | `a = a (1 - m)` 或 `a + m (v - a)`（向量化） | `a(mask) = 0` |
| RGB 通道 | `img[..., 0]` | `img[[All, All, 1]]` | `img(:,:,1)` |
| 加新轴 | `a[:, None]` | 无需：靠线程化规则（见第 4 节） | — |
| 条件下标 | `np.argwhere(a > t)` | `Position[a, x_ /; x > t]`（慢）或 `SparseArray[UnitStep[a - t]]["NonzeroPositions"]` | `find(a > t)` |
| 排序下标 | `np.argsort(a)` | `Ordering[a]` | `[~, i] = sort(a)` |

**换算技巧**：NumPy 的 `a[i:j]` 等于 Wolfram 的 `a[[i+1 ;; j]]`；Wolfram 的 `Range[n]` 是 1..n，NumPy 的 `np.arange(n)` 是 0..n−1。

---

## 4. 向量化：ufunc / 广播 vs Listable / 线程化

两者的哲学相同：**不写循环，让整个数组参与运算**。区别在于形状不一致时如何对齐。

### 4.1 逐元素运算

```python
y = np.sqrt(x) * 2 + 1          # ufunc
```

```mathematica
y = Sqrt[x] 2 + 1               (* Sqrt、Plus、Times 都有 Listable 属性 *)
```

### 4.2 形状对齐规则（最大的差别）

| | NumPy 广播 | Wolfram 线程化 |
|---|---|---|
| 对齐方向 | 从**最后一维**开始对齐，长度为 1 的维度可扩展 | 从**第一维**开始：两个列表长度相同时逐个配对 |
| 标量 | 广播到任意形状 | 线程到任意形状 |
| 矩阵 − 行向量（长度 W） | `a - v` 直接成立（减到每一行） | `a - v` **报错或语义错误**（要求长度 H）；需写 `(# - v) & /@ a` |
| 矩阵 − 列向量（长度 H） | 需写 `a - v[:, None]` | `a - v` 直接成立（第 i 行减 v[[i]]） |
| 外积 | `a[:, None] * b[None, :]` | `Outer[Times, a, b]` |

本仓库 `Deband` 的两种写法正好体现了这一点：

```python
img = img - prof[:, None]   # 按行减：NumPy 要加轴
img = img - prof[None, :]   # 按列减
```

```mathematica
x = x - prof               (* 按行减：Wolfram 天然按第一维配对 *)
x = (# - prof) & /@ x      (* 按列减：对每一行减去同一向量 *)
```

### 4.3 逐元素的 max / min / where

| 需求 | NumPy | Wolfram | 说明 |
|---|---|---|---|
| 逐元素最大 | `np.maximum(a, b)` | `a + Ramp[b - a]` | ⚠️ `Max[a, b]` 返回**整体最大值**（一个标量） |
| 逐元素最小 | `np.minimum(a, b)` | `a - Ramp[a - b]` | 本仓库 SCNR 用的就是这个 |
| 截断 | `np.clip(a, lo, hi)` | `Clip[a, {lo, hi}]` | 两者都只能截到常数 |
| 阶跃 | `(a >= 0).astype(float)` | `UnitStep[a]` | `UnitStep[0] = 1` |
| ReLU | `np.maximum(a, 0)` | `Ramp[a]` | |
| 非零 | `(a != 0)` | `Unitize[a]` | |
| 条件选择 | `np.where(c, x, y)` | `y + UnitStep[c] (x - y)` 或 `MapThread[If[#1, #2, #3] &, …]`（慢） | 用算术代替 `If`，保持打包 |
| 符号 | `np.sign(a)` | `Sign[a]` | |

**原则**：Wolfram 里 `If`、`Which`、`Select` 都是逐元素解释执行，在大数组上很慢；
用 `UnitStep`、`Ramp`、`Clip`、`Unitize` 组合出等价的算术表达式，才能享受打包数组的速度，作用相当于 NumPy 的布尔掩膜。

---

## 5. 归约与轴

| 操作 | NumPy | Wolfram |
|---|---|---|
| 全部求和 | `a.sum()` | `Total[a, Infinity]` 或 `Total[Flatten[a]]` |
| 按列求和（沿行方向） | `a.sum(axis=0)` | `Total[a]`（默认作用于第一层） |
| 按行求和 | `a.sum(axis=1)` | `Total[a, {2}]` |
| 一叠图像逐像素中值 | `np.median(cube, axis=0)` | `Median[cube]`（对 3 维数组默认沿第一维） |
| 均值 / 标准差 | `mean`、`std`（默认总体，`ddof=0`） | `Mean`、`StandardDeviation`（**样本**标准差，相当于 `ddof=1`） |
| 分位数 | `np.percentile(a, 99.9)`（0–100） | `Quantile[a, 0.999]`（0–1，默认定义不同，结果可能略有差别） |
| MAD | `np.median(np.abs(a - np.median(a)))` | `MedianDeviation[a]`（内置） |
| 累积和 / 差分 | `np.cumsum`、`np.diff` | `Accumulate`、`Differences` |
| 滑动平均 | `np.convolve(x, np.ones(n)/n, 'valid')` | `MovingAverage[x, n]` |
| 分块归约 | `a.reshape(h//b, b, w//b, b).mean((1, 3))` | `BlockMap[Mean[Flatten[#]] &, a, {b, b}]` |

**转置的约定恰好相反**，这是最容易出错的地方：

```python
# NumPy：结果的第 i 个轴取自输入的第 axes[i] 个轴
rgb = np.transpose(np.stack([r, g, b]), (1, 2, 0))   # (3,H,W) -> (H,W,3)
```

```mathematica
(* Wolfram：输入的第 k 层放到结果的第 n_k 层 *)
rgb = Transpose[{r, g, b}, {3, 1, 2}]                 (* (3,H,W) -> (H,W,3) *)
```

两者的置换数组互为逆置换：`(1,2,0)` ↔ `{3,1,2}`。

---

## 6. 图像的表示

### 6.1 数组即图像 vs Image 对象

| | NumPy 生态 | Wolfram |
|---|---|---|
| 类型 | 图像就是 ndarray，元数据要自己管 | `Image[data, type, ColorSpace -> …]`，类型有 `"Bit"`、`"Byte"`、`"Bit16"`、`"Real32"`、`"Real64"` |
| 通道顺序 | H×W×C；**OpenCV 是 BGR**，PIL/skimage 是 RGB | H×W×C，RGB；`ColorSeparate` / `ColorCombine` 拆合通道 |
| 数值范围 | uint8 0–255、uint16 0–65535、float 任意 | Byte/Bit16 在函数内部按 [0,1] 解释；Real32 保持原值 |
| 互转 | `np.asarray(PIL.Image)`、`cv2.imread` | `Image[a, "Real32"]` ↔ `ImageData[img, "Real32"]` |
| 函数对象 | 函数作用于数组 | 很多函数**同时接受数组和 Image**（`GaussianFilter`、`MedianFilter`），也有只接受 Image 的（`Binarize`、`ImageDeconvolve`） |

### 6.2 坐标系

| | 数组下标 | 几何坐标 |
|---|---|---|
| NumPy / OpenCV / skimage | `a[row, col]`，原点左上 | OpenCV 的点是 `(x=col, y=row)`，y **向下** |
| Wolfram | `a[[row, col]]`，第 1 行在上 | `ComponentMeasurements`、`ImageValue`、`ImagePerspectiveTransformation[…, DataRange -> Full]` 用 `{x, y}`，原点**左下**，y **向上**；像素 (i, j) 的中心在 `{j − 1/2, h − i + 1/2}` |

本仓库 `stack.wls` 中，相位相关得到的数组平移是 `(dcol, drow)`，转换成图像坐标时 y 要取反：
`TranslationTransform[{-4 dcol, 4 drow}]`。

### 6.3 FITS 与天文数据

| | NumPy 生态 | Wolfram |
|---|---|---|
| 读 FITS | `astropy.io.fits.getdata`（源自 STScI 的 PyFITS） | `Import[f, "FITS"]`，或像本仓库那样用 `BinaryReadList` 自己解析 |
| WCS | `astropy.wcs` | 无内置；本仓库自己实现了 TAN 投影 |
| 星表 / 天体数据 | `astroquery`（联网查询 Simbad、Gaia） | **内置**：`StarData`、`NebulaData`、`GalaxyData`、`PlanetData` |

---

## 7. 常用操作速查表

NumPy 列给出纯 NumPy 写法；没有纯 NumPy 写法时，括号里给出常用库。

| 功能 | NumPy / SciPy / skimage / OpenCV | Wolfram |
|---|---|---|
| 读图 | `PIL.Image.open`、`cv2.imread`、`skimage.io.imread` | `Import["a.png"]` |
| 写 16 位 TIFF | `tifffile.imwrite`、`cv2.imwrite(uint16)` | `Export["a.tif", Image[x, "Bit16"]]` |
| 灰度 | `img.mean(-1)` 或加权 | `ColorConvert[img, "Grayscale"]` |
| 缩放 | `cv2.resize`、`skimage.transform.resize` | `ImageResize[img, w]`、`ArrayResample[a, {h, w}]` |
| 裁剪 | `a[y0:y1, x0:x1]` | `a[[y0 ;; y1, x0 ;; x1]]`、`ImageTake`、`ImageCrop` |
| 旋转 / 翻转 | `np.rot90`、`np.flipud`、`cv2.warpAffine` | `ImageRotate`、`ImageReflect`、`Reverse[a]` |
| 填充边界 | `np.pad(a, r, mode="reflect")` | `ArrayPad[a, r, "Reflected"]`（`"Fixed"` 对应 `mode="edge"`） |
| 卷积 | `scipy.ndimage.convolve`、`cv2.filter2D` | `ListConvolve`、`ImageConvolve` |
| 高斯模糊 | `scipy.ndimage.gaussian_filter(a, σ)` | `GaussianFilter[a, r]`（r 是**半径**，默认 σ = r/2）或 `{r, σ}` |
| 中值滤波 | `scipy.ndimage.median_filter(a, 3)` | `MedianFilter[a, 1]`（参数是**半径**：1 表示 3×3） |
| 最大 / 最小滤波 | `scipy.ndimage.maximum_filter` | `MaxFilter`、`MinFilter` |
| 形态学 | `skimage.morphology.binary_dilation`、`cv2.erode` | `Dilation`、`Erosion`、`Opening`、`Closing`、`TopHatTransform` |
| 阈值 | `a > t`、`skimage.filters.threshold_otsu` | `Binarize[img, t]`、`Binarize[img]`（自动阈值） |
| 连通域 | `scipy.ndimage.label`、`skimage.measure.label` | `MorphologicalComponents` |
| 区域测量 | `skimage.measure.regionprops` | `ComponentMeasurements`（质心、面积、偏心率、半轴等上百种属性） |
| 删除小目标 | `skimage.morphology.remove_small_objects` | `DeleteSmallComponents` |
| FFT | `np.fft.fft2`、`rfft2` | `Fourier`、`InverseFourier`（指数符号与 NumPy 相反，归一化可用 `FourierParameters` 调整） |
| 直方图 | `np.histogram` | `HistogramList`、`ImageHistogram` |
| 色彩空间 | `cv2.cvtColor`、`skimage.color.rgb2lab` | `ColorConvert[img, "LAB"]` |
| 几何变换 | `cv2.warpAffine`、`warpPerspective`、`skimage.transform.warp` | `ImagePerspectiveTransformation`、`ImageTransformation`、`ImageForwardTransformation` |
| 求变换 | `cv2.estimateAffinePartial2D`、`skimage.transform.estimate_transform` | `FindGeometricTransform` |
| 自动对齐 | `astroalign`、`cv2.findTransformECC` | `ImageAlign` |
| 特征点 | `cv2.ORB_create`、`skimage.feature.corner_harris` | `ImageKeypoints`、`ImageCorners` |
| 最近邻 | `scipy.spatial.cKDTree` | `Nearest` |
| 最小二乘 | `np.linalg.lstsq` | `LeastSquares` |
| SVD | `np.linalg.svd` → `(U, s, Vᵀ)` | `SingularValueDecomposition` → `{U, Σ矩阵, V}`（是 V 不是 Vᵀ） |
| 反卷积 | `skimage.restoration.richardson_lucy` | `ImageDeconvolve[img, psf, Method -> "RichardsonLucy"]` |
| 小波 | `pywt.swt2`、`pywt.wavedec2` | `StationaryWaveletTransform`、`DiscreteWaveletTransform`、`WaveletThreshold` |
| 降噪 | `cv2.bilateralFilter`、`skimage.restoration.denoise_nl_means`、`denoise_tv_chambolle` | `BilateralFilter`、`NonlocalMeansFilter`、`TotalVariationFilter`、`WienerFilter` |
| 拉伸 / 自动色阶 | 手写、`skimage.exposure.rescale_intensity` | `ImageAdjust`、`Rescale` |
| 随机数 | `rng.normal(0, 1, (h, w))` | `RandomVariate[NormalDistribution[0, 1], {h, w}]` |

可以看出：**NumPy 本身只提供数组和基本数学运算**，图像算法分散在 SciPy、scikit-image、OpenCV、astropy 等库里；
**Wolfram 是一体化的**，以上全是同一个系统的内置函数，命名风格统一。

---

## 8. 实战对照：本仓库的算法

### 8.1 MTF 拉伸（`astrotools.mtf` ↔ `AstroTools`MTF`）

```python
def mtf(m, x):
    return ((m - 1) * x) / ((2 * m - 1) * x - m)

def stf(img, shadows=-2.8, target=0.25):
    x = normalize(img)
    med = np.median(x); madn = 1.4826 * np.median(np.abs(x - med))
    c0 = np.clip(med + shadows * madn, 0, 1)
    m = mtf(target, (med - c0) / (1 - c0))
    return mtf(m, np.clip((x - c0) / (1 - c0), 0, 1))
```

```mathematica
MTF[m_, x_] := ((m - 1) x)/((2 m - 1) x - m);

AutoSTF[a_, shadows_ : -2.8, target_ : 0.25] := Module[{x = Rescale[a], med, c0},
  med = Median[Flatten[x]];
  c0 = Clip[med + shadows 1.4826 MedianDeviation[Flatten[x]], {0, 1}];
  MTF[MTF[target, (med - c0)/(1 - c0)], Clip[(x - c0)/(1 - c0), {0, 1}]]]
```

对比：几乎一一对应。Wolfram 有现成的 `Rescale` 和 `MedianDeviation`；同一个 `MTF` 定义既能作用于标量也能作用于矩阵（Listable 运算自动线程化）。

### 8.2 σ-clip 叠加

```python
med = np.nanmedian(cube, axis=0)
sd = 1.4826 * np.nanmedian(np.abs(cube - med), axis=0)    # cube - med：广播 (N,H,W) - (H,W)
bad = np.abs(cube - med) > k * np.maximum(sd, 1e-6)
cube[bad] = np.nan                                          # 用 NaN 标记剔除
w = weights[:, None, None]
out = np.nansum(cube * w, axis=0) / np.sum(np.where(np.isnan(cube), 0, w), axis=0)
```

```mathematica
med  = Median[cube];
sd   = Clip[1.4826 Median[Abs[(# - med) & /@ cube]], {10.^-6, Infinity}];  (* 不能直接 cube - med *)
keep = vm (UnitStep[k sd - Abs[# - med]] & /@ cube);                       (* 用 0/1 掩膜代替 NaN *)
out  = Total[weights keep cube]/Total[weights keep];                       (* weights 按第一维线程化 *)
```

对比：
- NumPy 的 `cube - med` 靠尾维广播；Wolfram 要 `Map` 到每一帧。
- 反过来，`weights * keep`（长度 N 乘以 N×H×W）在 Wolfram 里直接按第一维配对，NumPy 要写成 `weights[:, None, None]`。
- NumPy 用 NaN 和 `nan*` 系列函数；Wolfram 用掩膜，以保持打包数组。

### 8.3 背景网格（分块中值 + 插值）

```python
t = img[:gh*tile, :gw*tile].reshape(gh, tile, gw, tile).swapaxes(1, 2).reshape(gh, gw, -1)
grid = np.median(t, axis=-1)            # reshape 技巧：零拷贝分块
bkg = upsample_grid(grid, img.shape)    # 自己写的双线性插值
```

```mathematica
grid = BlockMap[Median[Flatten[#]] &, a, {tile, tile}];
bkg  = ArrayResample[grid, Dimensions[a]];
```

对比：NumPy 的 reshape/swapaxes 技巧很高效，但不直观；Wolfram 的 `BlockMap` 和 `ArrayResample` 一看就懂。

### 8.4 星点检测

```python
# 纯 NumPy（scripts/astrotools.py）：局部极大值 + 手写矩
sm = gaussian_blur(sub, 1.0)
peaks = (sm == max_filter(sm, 5)) & (sm > k * sigma / 2)
ys, xs = np.nonzero(peaks)
# ...逐个切出邮票图，计算一阶矩、二阶矩
```

```mathematica
lab = MorphologicalComponents[UnitStep[GaussianFilter[sub, 2] - k s/2]];
ComponentMeasurements[{Image[sub, "Real32"], lab},
  {"IntensityCentroid", "Total", "Max", "Count"}, #Count >= 3 &]
```

对比：纯 NumPy 需要自己实现最大值滤波和矩计算（也可以用 `scipy.ndimage.label` + `skimage.measure.regionprops`）；
Wolfram 一个 `ComponentMeasurements` 就能拿到上百种区域属性。

### 8.5 相位相关

```python
F = np.fft.rfft2(b) * np.conj(np.fft.rfft2(a))
r = np.fft.irfft2(F / np.maximum(np.abs(F), 1e-12), s=a.shape)
dy, dx = np.unravel_index(np.argmax(r), r.shape)
```

```mathematica
f = Fourier[b] Conjugate[Fourier[a]];
r = Re@InverseFourier[f/(Abs[f] + 10.^-12)];
{dy, dx} = First@Position[r, Max[r]] - 1
```

对比：几乎逐行对应。差别有三点：Wolfram 没有实数 FFT（`rfft2`）这种省一半内存的优化；下标从 1 开始，所以要减 1；
`Fourier` 的指数符号与 NumPy 相反，但互功率谱峰值的位置不受影响。

### 8.6 starlet 小波（带孔卷积）

```python
def _conv1d(img, k, axis, step):                # 移位求和实现带孔卷积
    p = np.pad(img, pad, mode="reflect")
    for i, w in enumerate(k):
        out += w * np.take(p, range(i*step, i*step + n), axis=axis)
```

```mathematica
holed[j_] := Module[{s = 2^j, k = ConstantArray[0., 4 2^j + 1]}, k[[1 ;; ;; s]] = b3; k];
sepSmooth[c_, k_] := With[{r = (Length[k] - 1)/2},
  ListConvolve[Transpose[{k}],
    ArrayPad[ListConvolve[{k}, ArrayPad[c, {{0, 0}, {r, r}}, "Reflected"]], {{r, r}, {0, 0}}, "Reflected"]]];
```

对比：Wolfram 把孔直接写进卷积核（插入零），用 `ListConvolve` 做可分离卷积，代码更短，但多算了很多零。
NumPy 版用移位求和只算非零抽头，更省计算。

---

## 9. 性能与内存

| | NumPy | Wolfram |
|---|---|---|
| 快的前提 | 避免 Python 循环，全部用 ufunc / 向量化 | 保持**打包数组**，避免 `Table` / `Do` 逐元素、`If`、精确数、`Indeterminate` |
| 检查 | `a.dtype`、`a.flags` | `Developer`PackedArrayQ[a]` |
| 加速循环 | Numba `@njit`、Cython | `Compile[…, CompilationTarget -> "C"]`、`FunctionCompile`（12.x+） |
| 内存 | float32 原生，视图零拷贝，可以 memmap（本仓库 `milkyway/` 用 memmap 做磁盘缓存） | 默认 Real64（同样图像占 2 倍内存）；值语义可能产生隐式拷贝 |
| 多线程 | BLAS/FFT 多线程；其余单线程（OpenCV 内部多线程） | 很多内置图像函数自动并行；`ParallelMap` 用于多核 |
| 大数据 | memmap、dask、分块处理 | 分块处理；`NumericArray` 紧凑存储 |

**经验**：同一个算法，「写得对」的 Wolfram 代码和 NumPy 速度通常在同一数量级。
「写得不对」时（解包、逐元素 `If`），Wolfram 可能慢 100 倍以上；NumPy 写 Python for 循环也是如此。

---

## 10. 易错点清单

### 从 NumPy 转到 Wolfram

1. 下标从 1 开始；`;;` 区间**包含两端**。
2. `Max`、`Min` 不是逐元素运算；用 `Ramp` 组合，或 `MapThread[Max, {a, b}, 2]`（慢）。
3. 线程化按**第一维**配对，不是按尾维广播。
4. `Transpose` 的置换语义与 `np.transpose` 互逆。
5. `StandardDeviation` 是样本标准差（n−1）。
6. 避免精确数：写 `0.5` 而不是 `1/2`，写 `10.^-6` 而不是 `10^-6`。
7. 大写单字母（`N`、`D`、`E`、`I`、`C`、`O`、`K`）是内置符号，不能当变量名。
8. 滤波函数的参数是**半径**而不是窗口大小或 σ：`MedianFilter[a, 1]` 是 3×3，`GaussianFilter[a, r]` 默认 σ = r/2。
9. 图像几何坐标的原点在**左下**。
10. 不要用 NaN 标记缺失：`Indeterminate` 会解包数组并污染求和，改用 0/1 掩膜。

### 从 Wolfram 转到 NumPy

1. 下标从 0 开始；切片**不含**终点。
2. 切片是**视图**：修改会影响原数组，需要独立副本时用 `.copy()`。
3. 整数 dtype 会溢出、整除会截断：`uint8` 255 + 1 = 0，`uint16` 相减可能得到巨大正数。图像运算前先 `astype(np.float32)`。
4. 广播从尾维对齐：列向量要写 `v[:, None]`。
5. `np.std` 默认是总体标准差（`ddof=0`）。
6. OpenCV 的通道顺序是 **BGR**。
7. NaN 会传染：`np.median` 遇到 NaN 返回 NaN，要用 `np.nanmedian`。
8. 只有数组和基本运算是 NumPy 自带的，图像算法要按需选 SciPy / skimage / OpenCV / astropy，各库的坐标约定和数值范围各不相同。

---

## 参考

- NumPy 官方文档：《NumPy for MATLAB users》；NumPy 论文 Harris et al., *Array programming with NumPy*, Nature 585 (2020)
- Travis Oliphant, *Guide to NumPy* (2006)
- Greenfield et al., *numarray: A New Scientific Array Package for Python*（STScI）
- Iverson, *A Programming Language* (1962)；Iverson, *Notation as a Tool of Thought* (1979 图灵奖演讲)
- Wolfram Language 文档：Packed Arrays、Image Processing 指南页
- 本仓库：`scripts/astrotools.py`（NumPy）、`wolfram/AstroTools.wl`（Wolfram）
