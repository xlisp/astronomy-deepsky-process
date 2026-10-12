# MATLAB vs NumPy vs Wolfram：图像处理详细对比

三种数组语言在图像处理上的详细对照，例子取自本仓库的天文后期流程。
姊妹篇：[NumPy 与 Wolfram 对比](numpy_vs_wolfram_image_processing.md)（偏语言基础）、
[数组计算历史](array_computing_history.md)（MATLAB 是 NumPy 函数命名的主要来源）。

> 本仓库的 Python 代码都实际运行过，Wolfram 代码尚未在真实内核上运行，本文的 MATLAB 代码则是示例性质（同样未运行），
> 并标注了所需工具箱和最低版本。MATLAB 近年有不少函数改名（如 `fitgeotrans` → `fitgeotform2d`），旧版本请使用括号里的旧名。

- [1. 三者一览](#1-三者一览)
- [2. 数据模型与图像类型](#2-数据模型与图像类型)
- [3. 坐标系](#3-坐标系)
- [4. 同名不同义：默认参数差异表](#4-同名不同义默认参数差异表)
- [5. 逐项对照（带代码）](#5-逐项对照带代码)
  - [5.1 读写与类型转换](#51-读写与类型转换) · [5.2 显示与拉伸](#52-显示与拉伸) · [5.3 线性滤波](#53-线性滤波)
  - [5.4 非线性滤波与降噪](#54-非线性滤波与降噪) · [5.5 形态学](#55-形态学) · [5.6 阈值、连通域与测量](#56-阈值连通域与测量)
  - [5.7 频域](#57-频域) · [5.8 几何变换与配准](#58-几何变换与配准) · [5.9 多帧叠加](#59-多帧叠加)
  - [5.10 背景拟合](#510-背景拟合) · [5.11 色彩空间与合成](#511-色彩空间与合成) · [5.12 反卷积](#512-反卷积) · [5.13 小波](#513-小波)
- [6. 完整小流程：三种语言各写一遍](#6-完整小流程三种语言各写一遍)
- [7. 生态、授权与性能](#7-生态授权与性能)
- [8. 迁移易错点](#8-迁移易错点)

---

## 1. 三者一览

| | MATLAB | NumPy 生态 | Wolfram Language |
|---|---|---|---|
| 出现 | 约 1979（Fortran 版），1984 年商业化 | Numeric 1995 → NumPy 2006 | Mathematica 1988；图像函数自 2008 年的 v7 起 |
| 图像能力在哪 | 基础 MATLAB + **Image Processing Toolbox（IPT）** + Computer Vision Toolbox + Wavelet Toolbox | NumPy + SciPy（`ndimage`）+ scikit-image + OpenCV + astropy + PyWavelets | 全部内置 |
| 图像是什么 | 普通矩阵（H×W 或 H×W×3），类型决定数值范围 | 普通 ndarray，没有统一的数值范围约定 | 独立的 `Image` 对象，也可以直接用数组 |
| 下标 | 1 起始，**列优先** | 0 起始，行优先 | 1 起始，行优先（嵌套列表） |
| 赋值语义 | 写时复制（值语义） | 引用，切片是视图 | 值语义 |
| 函数命名 | 缩写：`imfilter`、`bwlabel`、`regionprops` | 大多沿用 MATLAB：`regionprops`、`rgb2gray`、`medfilt2d` | 英文全称：`ComponentMeasurements` |
| 价格 | 商业授权，工具箱另购 | 免费开源 | Mathematica 商业授权；Wolfram Engine 个人开发免费 |
| 开源替代 | GNU Octave + image 包（兼容大部分 IPT） | — | — |

---

## 2. 数据模型与图像类型

### 2.1 类型与数值范围约定

这是三者差异最大、也最容易出错的地方。

| 数据类型 | MATLAB 约定 | NumPy 生态 | Wolfram |
|---|---|---|---|
| 8 位 | `uint8`，0–255 | `uint8`，0–255 | `"Byte"`，函数内部按 [0,1] 解释 |
| 16 位 | `uint16`，0–65535 | `uint16`，0–65535 | `"Bit16"`，函数内部按 [0,1] 解释 |
| 浮点 | `double`/`single`，**约定范围 [0,1]**（`imshow` 按此显示） | `float32/64`，没有约定；skimage 约定 [0,1]，OpenCV 不做约定 | `"Real32"`/`"Real64"`，原值保留，显示时按 [0,1] |
| 二值 | `logical` | `bool` | `"Bit"` |
| 类型转换 | `im2double`（uint8 除以 255，uint16 除以 65535）、`im2uint16`、`im2single` | `skimage.util.img_as_float`（**照搬 MATLAB 语义**）、`astype` | `Image[x, "Bit16"]`、`ImageData[img, "Real32"]` |
| 默认数值类型 | `double` | `float64`（但 FITS、OpenCV 常用 `float32`） | Real64 |

**整数运算的行为完全不同**：

```matlab
uint8(250) + 10      % 255  —— MATLAB 整数运算饱和（截断到上限）
uint8(7) / 2         % 4    —— 四舍五入
```

```python
np.uint8(250) + np.uint8(10)   # 4    —— NumPy 溢出回绕（模 256）
np.uint8(7) // 2               # 3    —— 整除向下取整
```

```mathematica
(* Wolfram 没有定宽整数：250 + 10 = 260；Image 函数会自动截断到合法范围 *)
```

所以处理天文数据时三者的共同建议是：**一读进来就转成浮点**（MATLAB 用 `double(...)`/`im2double`，
NumPy 用 `astype(np.float32)`，Wolfram 用 `N[...]`）。

### 2.2 多帧数据放在第几维

| | MATLAB | NumPy | Wolfram |
|---|---|---|---|
| 习惯形状 | **H×W×N**（帧放在第 3 维；彩色序列为 H×W×3×N） | **N×H×W**（帧放在第 0 维） | **N×H×W**（列表的第一层） |
| 逐像素中值 | `median(cube, 3)` | `np.median(cube, axis=0)` | `Median[cube]` |
| 原因 | 列优先：第 1、2 维连续，单帧在内存中是连续的 | 行优先：最后两维连续，单帧在内存中是连续的 | 嵌套列表的自然结构 |

### 2.3 向量化与形状对齐

| | MATLAB | NumPy | Wolfram |
|---|---|---|---|
| 逐元素乘、除、幂 | `.*`、`./`、`.^`（`*` 是**矩阵乘**） | `*`、`/`、`**`（`@` 是矩阵乘） | `*`（或空格）、`/`、`^`（`.` 是矩阵乘） |
| 形状对齐 | **隐式扩展**（R2016b 起；之前要用 `bsxfun`），规则与 NumPy 广播相同 | 广播：从尾维对齐 | 按第一维线程化 |
| H×W×N 减 H×W | `cube - med` 直接成立 | `cube - med`（N×H×W 减 H×W）直接成立 | 需写 `(# - med) & /@ cube` |
| 逐元素最大 | `max(A, B)` ✅ | `np.maximum(A, B)` | `A + Ramp[B - A]`（`Max[A, B]` 是整体最大值） |
| 沿某维最大 | `max(A, [], 3)` | `A.max(axis=2)` | `Max /@ …` 或 `Map[Max, A, {2}]` |
| 全部元素 | `sum(A, 'all')`（R2018b 起），或 `sum(A(:))` | `A.sum()` | `Total[A, Infinity]` |
| 布尔掩膜 | `A(mask) = 0`（逻辑索引，MATLAB 首创） | `A[mask] = 0` | `A (1 - m)` |

---

## 3. 坐标系

三者各不相同，配准、测星、裁剪时最容易错。

| | MATLAB | NumPy / scikit-image | OpenCV | Wolfram |
|---|---|---|---|---|
| 数组下标 | `I(row, col)`，从 1 开始 | `a[row, col]`，从 0 开始 | `a[row, col]`，从 0 开始 | `a[[row, col]]`，从 1 开始 |
| 几何点的写法 | **`[x y]` = [列 行]** | `(row, col)`（skimage 的多数函数） | `(x, y)` = (列, 行) | `{x, y}` |
| 原点 | 左上；第一个像素中心在 **(1, 1)**，左上角在 (0.5, 0.5)（「内在坐标」） | 左上；第一个像素中心在 (0, 0) | 左上；第一个像素中心在 (0, 0) | **左下**；像素中心在 (j − ½, h − i + ½) |
| y 轴方向 | 向下 | 向下 | 向下 | **向上** |
| 世界坐标 | `imref2d`（指定像素尺寸和范围，配准时 `imwarp` 的 `'OutputView'` 用它） | `skimage.transform` 的变换矩阵 | 变换矩阵 | `DataRange` 选项 |

**同一个星点在四个系统里的坐标**：假设它在第 101 行、第 51 列（MATLAB 的 1 起始写法），图像高 1000：

| 系统 | 写法 |
|---|---|
| MATLAB `regionprops` 的 `Centroid` | `[51 101]` |
| skimage `regionprops` 的 `centroid` | `(100, 50)` |
| OpenCV / sep / 本仓库 `detect_stars` | `x=50, y=100` |
| Wolfram `ComponentMeasurements` | `{50.5, 899.5}` |

---

## 4. 同名不同义：默认参数差异表

| 操作 | MATLAB | NumPy 生态 | Wolfram |
|---|---|---|---|
| 高斯模糊的参数 | `imgaussfilt(I, sigma)`：**σ**；核大小默认 `2*ceil(2σ)+1` | `ndimage.gaussian_filter(a, sigma)`：**σ**，截断 4σ；`cv2.GaussianBlur(a, (k,k), σ)` | `GaussianFilter[a, r]`：**半径**，默认 σ = r/2；`{r, σ}` 可以同时指定 |
| 中值滤波的参数 | `medfilt2(I, [3 3])`：**窗口尺寸** | `ndimage.median_filter(a, size=3)`：窗口尺寸 | `MedianFilter[a, 1]`：**半径**（1 表示 3×3） |
| 边界默认 | `imfilter`：**补零**；`imgaussfilt`：`'replicate'`；`medfilt2`：补零 | `ndimage`：`'reflect'`（d c b a \| a b c d，含边缘像素）；`np.pad(mode='reflect')` **不重复**边缘像素；OpenCV：`BORDER_REFLECT_101` | 多数滤波器默认 `Padding -> "Fixed"`（重复边缘像素） |
| 相关还是卷积 | `imfilter` 默认**相关**；`conv2` 是卷积 | `ndimage.correlate` / `convolve` 分开；`cv2.filter2D` 是相关 | `ListCorrelate` / `ListConvolve` 分开；`ImageConvolve` 是卷积 |
| 质心坐标顺序 | `[x y]` | skimage `(row, col)` | `{x, y}`，y 向上 |
| Lab 的 L 范围 | `rgb2lab`：0–100 | `skimage.color.rgb2lab`：0–100；OpenCV float：0–100，uint8：0–255 | `ColorConvert[…, "LAB"]`：0–1 |
| 标准差 | `std` 用 n−1 | `np.std` 用 n | `StandardDeviation` 用 n−1 |
| 分位数 | `prctile`：中点插值 | `np.percentile`：线性插值（type 7） | `Quantile`：默认**不插值**（逆经验分布） |
| MAD | `mad(x, 1)`（Statistics Toolbox，**不含** 1.4826）；`isoutlier` 内部使用 scaled MAD | `scipy.stats.median_abs_deviation(x, scale='normal')` 含 1.4826 | `MedianDeviation` 不含 1.4826 |
| FFT 指数符号 | `exp(−i…)` | `exp(−i…)` | `exp(+i…)`（可用 `FourierParameters -> {1, -1}` 改成信号处理约定） |
| 图像缩放插值 | `imresize`：默认**双三次** | `skimage.transform.resize`：默认双线性（`order=1`）；`cv2.resize`：默认双线性 | `ImageResize`：默认自动选择 |
| `max(A, B)` | 逐元素 | `np.max(A, B)` 会把 B 当作 axis（报错），逐元素要用 `np.maximum` | `Max[A, B]` 返回整体最大值 |

---

## 5. 逐项对照（带代码）

### 5.1 读写与类型转换

**FITS**：

```matlab
% 基础 MATLAB 自带 FITS 支持（底层是 CFITSIO）
info = fitsinfo('master_L.fits');                 % 头信息
I    = double(fitsread('master_L.fits'));
fitswrite(single(I), 'out.fits');
% 低层接口：matlab.io.fits.openFile / readImg / readKey
% 注意：MATLAB 列优先，FITS 的 NAXIS1 是变化最快的轴。读入后用 size(I) 核对行列，必要时转置
```

```python
from astropy.io import fits
I = fits.getdata('master_L.fits').astype(np.float32)   # 形状 (NAXIS2, NAXIS1)
fits.writeto('out.fits', I, overwrite=True)
```

```mathematica
{I0, hdr} = ReadFITS["master_L.fits"];    (* 本仓库自己写的解析器；也可用 Import[f, "FITS"] *)
WriteFITS["out.fits", I0];
```

**普通图片 / 16 位 TIFF**：

| | MATLAB | NumPy 生态 | Wolfram |
|---|---|---|---|
| 读 | `I = imread('a.tif')` | `tifffile.imread`、`cv2.imread(f, cv2.IMREAD_UNCHANGED)`、`PIL.Image.open` | `Import["a.tif"]` |
| 写 16 位 | `imwrite(im2uint16(J), 'a.tif')` | `tifffile.imwrite(f, (J*65535).astype(np.uint16))` | `Export["a.tif", Image[J, "Bit16"]]` |
| 写 JPEG 质量 | `imwrite(J, 'a.jpg', 'Quality', 95)` | `PIL ... save(quality=95)` | `Export[…, "CompressionLevel" -> 0.05]` |
| 相机 RAW | `rawread`（IPT，R2021a 起）+ `raw2rgb` | `rawpy`（LibRaw） | `Import["a.CR2"]`（部分格式） |
| 元数据 / EXIF | `imfinfo` | `PIL ... getexif()`、`exifread` | `Import[f, "Exif"]` |

### 5.2 显示与拉伸

```matlab
imshow(I, [])                                    % 自动把 [min, max] 映射到黑白
J = imadjust(rescale(I), stretchlim(rescale(I), [0.001 0.999]), []);   % 1%-99.9% 截断拉伸
J = histeq(rescale(I));   K = adapthisteq(rescale(I));                  % 直方图均衡 / CLAHE
% PixInsight 的 STF 自动拉伸（本仓库的 MTF 公式）
mtf = @(m, x) ((m - 1) .* x) ./ ((2*m - 1) .* x - m);
x = rescale(I);  med = median(x(:));  madn = 1.4826 * mad(x(:), 1);
c0 = min(max(med - 2.8*madn, 0), 1);
J = mtf(mtf(0.25, (med - c0)/(1 - c0)), min(max((x - c0)/(1 - c0), 0), 1));
```

```python
plt.imshow(I, cmap='gray', vmin=np.percentile(I, 0.1), vmax=np.percentile(I, 99.9))
J = skimage.exposure.rescale_intensity(I, in_range=tuple(np.percentile(I, (0.1, 99.9))), out_range=(0, 1))
J = skimage.exposure.equalize_adapthist(rescale(I))              # CLAHE
J = at.stf(I)                                                    # 本仓库 scripts/astrotools.py
```

```mathematica
ImageAdjust[Image[I0, "Real32"]]                (* 自动线性拉伸 *)
ImageAdjust[img, {0, 0, 1}, {0.001, 0.999}]      (* 截断拉伸 *)
HistogramTransform[img]                          (* 直方图均衡 *)
Image@AutoSTF[I0]                                (* wolfram/AstroTools.wl *)
Manipulate[Image@Stretch[I0, t], {t, 0.02, 0.4}] (* 交互调参：MATLAB 对应 App Designer / imcontrast *)
```

MATLAB 的 `imcontrast`、`imtool` 提供交互式调对比度的界面，最接近 PixInsight 的使用体验；Wolfram 靠 `Manipulate`；
Python 要用 matplotlib 控件或 napari。

### 5.3 线性滤波

```matlab
G  = imgaussfilt(I, 2);                                   % IPT R2015a 起；默认边界 'replicate'
K  = fspecial('gaussian', 13, 2);
G2 = imfilter(I, K, 'symmetric', 'conv');                 % 默认补零 + 相关，这里改为对称边界 + 卷积
B  = imboxfilt(I, 5);                                     % 均值滤波
D  = conv2(I, K, 'same');                                 % 基础 MATLAB，补零
HP = I - imgaussfilt(I, 8);                               % 高通：去掉大尺度结构
```

```python
G  = ndimage.gaussian_filter(I, 2)                        # 默认 mode='reflect'
G2 = ndimage.convolve(I, K, mode='reflect')
B  = ndimage.uniform_filter(I, 5)
G3 = cv2.GaussianBlur(I, (0, 0), 2)
G4 = at.gaussian_blur(I, 2)                               # 本仓库纯 numpy 可分离实现
```

```mathematica
g  = GaussianFilter[I0, {6, 2}];                          (* {半径, σ} *)
g2 = ListConvolve[GaussianMatrix[{6, 2}], ArrayPad[I0, 6, "Reflected"]];
b  = MeanFilter[I0, 2];                                   (* 半径 2 → 5×5 *)
```

### 5.4 非线性滤波与降噪

| 功能 | MATLAB | NumPy 生态 | Wolfram |
|---|---|---|---|
| 中值 | `medfilt2(I, [3 3], 'symmetric')` | `ndimage.median_filter(I, 3)` | `MedianFilter[I0, 1]` |
| 最大 / 最小 | `ordfilt2(I, 9, ones(3))` / `ordfilt2(I, 1, ones(3))`，或 `imdilate`/`imerode` | `ndimage.maximum_filter` / `minimum_filter` | `MaxFilter` / `MinFilter` |
| 双边 | `imbilatfilt(I, degreeOfSmoothing, spatialSigma)`（R2018a） | `cv2.bilateralFilter`、`skimage.restoration.denoise_bilateral` | `BilateralFilter[img, σ, μ]` |
| 非局部均值 | `imnlmfilt`（R2018b） | `skimage.restoration.denoise_nl_means`、`cv2.fastNlMeansDenoising` | `NonlocalMeansFilter` |
| 全变分 | — （需自己实现或用第三方） | `skimage.restoration.denoise_tv_chambolle` | `TotalVariationFilter` |
| 维纳 | `wiener2(I, [5 5])` | `scipy.signal.wiener` | `WienerFilter` |
| 深度学习降噪 | `denoiseImage` + `denoisingNetwork("DnCNN")`（Deep Learning Toolbox） | PyTorch / ONNX 模型 | `NetModel[...]` |

天文中常用的「亮度遮罩降噪」（本仓库 `compose.py` 中的 `denoise`）：

```matlab
blur = imbilatfilt(x, 0.01, 3);
w = min(x / 0.5, 1);                     % 亮处保留细节
y = w .* x + (1 - w) .* blur;
```

```python
blur = cv2.bilateralFilter(x.astype(np.float32), 0, 25, 3)
w = np.clip(x / 0.5, 0, 1)
y = w * x + (1 - w) * blur
```

```mathematica
blur = ImageData@BilateralFilter[Image[x, "Real32"], 3, 0.1];
w = Clip[x/0.5, {0, 1}];
y = w x + (1 - w) blur
```

### 5.5 形态学

| 功能 | MATLAB（IPT） | NumPy 生态 | Wolfram |
|---|---|---|---|
| 结构元 | `strel('disk', 2)`、`strel('square', 3)` | `skimage.morphology.disk(2)`、`np.ones((3, 3))` | `DiskMatrix[2]`、`BoxMatrix[1]` |
| 腐蚀 / 膨胀 | `imerode`、`imdilate` | `skimage.morphology.erosion` / `dilation`、`cv2.erode` / `dilate`、`ndimage.grey_erosion` | `Erosion`、`Dilation` |
| 开 / 闭 | `imopen`、`imclose` | `opening`、`closing` | `Opening`、`Closing` |
| 顶帽 | `imtophat` | `white_tophat` | `TopHatTransform` |
| 删除小目标 | `bwareaopen(BW, 3)` | `remove_small_objects(BW, 3)` | `DeleteSmallComponents[bw, 3]` |
| 填洞 | `imfill(BW, 'holes')` | `ndimage.binary_fill_holes` | `FillingTransform` |
| 距离变换 | `bwdist` | `ndimage.distance_transform_edt` | `DistanceTransform` |

**缩星（本仓库 `star_reduce.py`）**：

```matlab
small = L - imgaussfilt(L, 4);                                     % 小尺度结构 = 星点
mask  = bwareaopen(small > 4*sig, 3);
mask  = imgaussfilt(double(imdilate(mask, strel('disk', 2))), 2);  % 膨胀 + 羽化
er    = imerode(I, strel('disk', 1));
out   = (1 - a*mask) .* I + a*mask .* min(er, I);                  % H×W 的 mask 隐式扩展到 H×W×3
```

```python
mask = at.star_mask(I, k=4, grow=2)                                # 本仓库实现
er = np.dstack([at.min_filter(I[..., c], 3) for c in range(3)])
m = (a * mask)[..., None]
out = m * np.minimum(er, I) + (1 - m) * I
```

```mathematica
mask = StarMask[img];                                              (* wolfram/AstroTools.wl *)
StarReduce[img, 1, 0.7]                                            (* 内部使用 Erosion[img, DiskMatrix[1]] *)
```

### 5.6 阈值、连通域与测量

这是星点检测的核心。三者都有「二值化 → 连通域 → 区域测量」这套流程，属性命名差异很大：

| 属性 | MATLAB `regionprops` | skimage `regionprops`（0.19+ 命名） | Wolfram `ComponentMeasurements` |
|---|---|---|---|
| 面积 | `Area` | `area` | `"Count"` / `"Area"` |
| 几何质心 | `Centroid`，`[x y]` | `centroid`，`(row, col)` | `"Centroid"`，`{x, y}` y 向上 |
| 亮度加权质心 | `WeightedCentroid`（需传入灰度图） | `centroid_weighted`（旧名 `weighted_centroid`） | `"IntensityCentroid"` |
| 总亮度 | `sum(PixelValues)` | `intensity_mean * area` | `"Total"` |
| 峰值 | `MaxIntensity` | `intensity_max`（旧名 `max_intensity`） | `"Max"` |
| 偏心率 | `Eccentricity` | `eccentricity` | `"Eccentricity"` |
| 长短轴 | `MajorAxisLength`、`MinorAxisLength` | `axis_major_length`、`axis_minor_length` | `"SemiAxes"`（**半轴**长度） |
| 方向角 | `Orientation`（度） | `orientation`（弧度，定义不同） | `"Orientation"`（弧度） |
| 结果格式 | 结构体数组，或 `'table'` 选项得到 table | 对象列表，或用 `regionprops_table` 得到 dict | `标签 -> {值…}` 规则列表 |

**同一段星点检测代码**：

```matlab
% 需要 IPT；mad 需要 Statistics Toolbox
bkg = imresize(blockproc(I, [64 64], @(b) median(b.data(:))), size(I), 'bilinear');
S   = I - bkg;
sig = 1.4826 * mad(S(:), 1);
BW  = bwareaopen(imgaussfilt(S, 1) > 5*sig/2, 3);
st  = regionprops(bwconncomp(BW), max(S, 0), 'WeightedCentroid', 'MaxIntensity', 'Area');
xy  = vertcat(st.WeightedCentroid);              % N×2，[x y]
```

```python
bkg = at.background_map(I, 64)
S = I - bkg
sig = at.mad_std(S)
lab, n = ndimage.label(ndimage.gaussian_filter(S, 1) > 5 * sig / 2)
props = skimage.measure.regionprops(lab, intensity_image=np.clip(S, 0, None))
xy = np.array([p.centroid_weighted[::-1] for p in props if p.area >= 3])   # 转成 (x, y)
# 或者直接：st = at.detect_stars(I)；sep.extract(S, 5, err=sig)
```

```mathematica
bkg = ArrayResample[BlockMap[Median@*Flatten, I0, {64, 64}], Dimensions[I0]];
s = I0 - bkg;  sig = 1.4826 MedianDeviation[Flatten[s]];
lab = MorphologicalComponents[UnitStep[GaussianFilter[s, 2] - 5 sig/2]];
xy = Values@ComponentMeasurements[{Image[Clip[s, {0, Infinity}], "Real32"], lab},
    "IntensityCentroid", #Count >= 3 &]
```

**阈值方法**：MATLAB 用 `imbinarize(I)`（默认 Otsu）、`imbinarize(I, 'adaptive')`、`graythresh`；
skimage 用 `threshold_otsu`、`threshold_local`；Wolfram 用 `Binarize[img]`、`FindThreshold[img, Method -> "Otsu"]`、`LocalAdaptiveBinarize`。
天文图像一般不用 Otsu，而是用「背景 + kσ」，三者的写法都和上面的例子相同。

### 5.7 频域

```matlab
F  = fft2(I);  P = abs(fftshift(F)).^2;                  % 功率谱
% 相位相关求平移（与本仓库 milkyway/ 的 phase_shift 相同）
X  = fft2(b) .* conj(fft2(a));
r  = real(ifft2(X ./ max(abs(X), 1e-12)));
[~, k] = max(r(:));  [dy, dx] = ind2sub(size(r), k);  dy = dy - 1;  dx = dx - 1;
% 现成函数：tform = imregcorr(moving, fixed)（IPT）；归一化互相关 normxcorr2
```

```python
X = np.fft.rfft2(b) * np.conj(np.fft.rfft2(a))
r = np.fft.irfft2(X / np.maximum(np.abs(X), 1e-12), s=a.shape)
dy, dx = np.unravel_index(np.argmax(r), r.shape)
# 现成函数：skimage.registration.phase_cross_correlation(a, b, upsample_factor=10)（亚像素）
```

```mathematica
x = Fourier[b] Conjugate[Fourier[a]];
r = Re@InverseFourier[x/(Abs[x] + 10.^-12)];
{dy, dx} = First@Position[r, Max[r]] - 1
```

MATLAB 和 NumPy 的 FFT 约定一致；Wolfram 的指数符号相反，但这个算法不受影响。
`fftshift` 在 MATLAB、NumPy 中同名，Wolfram 中用 `RotateLeft[r, Floor[Dimensions[r]/2]]`。

### 5.8 几何变换与配准

| 功能 | MATLAB | NumPy 生态 | Wolfram |
|---|---|---|---|
| 变换对象 | `affinetform2d`、`simtform2d`、`projtform2d`（R2022b 起；旧版 `affine2d`、`projective2d`） | 3×3 矩阵；`skimage.transform.SimilarityTransform` | `TransformationFunction` |
| 由点对拟合 | `fitgeotform2d(moving, fixed, "similarity")`（旧版 `fitgeotrans`） | `skimage.transform.estimate_transform('similarity', src, dst)`、`cv2.estimateAffinePartial2D` | `FindGeometricTransform[fixed, moving, TransformationClass -> "Similarity"]` |
| 带 RANSAC 拟合 | `estgeotform2d`（CV Toolbox，MSAC 算法；旧版 `estimateGeometricTransform2D`） | `skimage.measure.ransac`、`cv2.estimateAffinePartial2D(method=cv2.RANSAC)` | `FindGeometricTransform` 对图像输入时内部使用 RANSAC |
| 最近邻匹配 | `knnsearch`、`KDTreeSearcher`（Statistics Toolbox） | `scipy.spatial.cKDTree` | `Nearest` |
| 重采样 | `imwarp(I, tform, 'OutputView', imref2d(size(ref)))`，插值 `'nearest'`/`'linear'`/`'cubic'` | `cv2.warpAffine`（支持 Lanczos4）、`skimage.transform.warp` | `ImagePerspectiveTransformation[img, tf, DataRange -> Full, Resampling -> "Lanczos"]` |
| 自动配准（亮度） | `imregister`、`imregtform`（基于优化） | `cv2.findTransformECC` | `ImageAlign` |
| 自动配准（特征） | `detectORBFeatures` / `detectSURFFeatures` + `matchFeatures` | `cv2.ORB_create` + `BFMatcher`；天文用 `astroalign` | `ImageKeypoints` + `ImageCorrespondingPoints` |

**用星点做配准（对应本仓库 `stack.py` 的引导匹配）**：

```matlab
pred = transformPointsForward(tform0, xy);              % 预测位置
[idx, d] = knnsearch(refXY, pred);                       % 最近邻
ok = d < 6;
[tform, inl] = estgeotform2d(xy(ok, :), refXY(idx(ok), :), "similarity");   % RANSAC 精配
out = imwarp(I, tform, 'OutputView', imref2d(size(ref)), 'Interp', 'cubic');
```

```python
tree = cKDTree(ref_xy)
d, idx = tree.query(apply(T0, xy), distance_upper_bound=6)
ok = np.isfinite(d)
M, inl = cv2.estimateAffinePartial2D(xy[ok], ref_xy[idx[ok]], method=cv2.RANSAC, ransacReprojThreshold=2)
out = cv2.warpAffine(I, M, (W, H), flags=cv2.INTER_LANCZOS4)
```

```mathematica
{tf, n, rms} = GuidedTransform[refXY, xy, tf0];          (* Nearest + FindGeometricTransform *)
{out, mask} = WarpFrame[I0, tf];                          (* ImagePerspectiveTransformation *)
```

MATLAB 的 `imwarp` 不提供 Lanczos 插值（`imresize` 有 `'lanczos3'`）。对星点精度要求高时，可以用 `'cubic'`，或者自己实现。

### 5.9 多帧叠加

```matlab
cube = cat(3, frames{:});                                % H×W×N
med  = median(cube, 3);
sd   = 1.4826 * median(abs(cube - med), 3);              % 隐式扩展（R2016b+）
keep = abs(cube - med) <= 3 * max(sd, 1e-6);             % max 在这里逐元素
w    = reshape(weights, 1, 1, []);                       % 权重放到第 3 维
master = sum(cube .* keep .* w, 3) ./ sum(keep .* w, 3);
% 或者用 isoutlier(cube, 'median', 3, 'ThresholdFactor', 3)（R2017a+，内部即 scaled MAD）
```

```python
cube = np.stack(frames)                                  # N×H×W
master, rej = at.sigma_clip_mean(cube, 3, 3, weights)    # 本仓库：nanmedian + NaN 标记
# 或 astropy.stats.sigma_clipped_stats / ccdproc.Combiner
```

```mathematica
{master, rej} = SigmaClipStack[frames, valid, weights, 3.];   (* Median[cube] 按第一维 *)
```

三者的内存差别：MATLAB 默认 double（可以用 `single` 减半），NumPy 本仓库用 float32，Wolfram 计算时总是 Real64。
大规模叠加时 MATLAB 可以用 `matfile` / `datastore` 分块处理，NumPy 用 `memmap`，Wolfram 只能手动分条带。

### 5.10 背景拟合

MATLAB 的反斜杠 `\` 就是最小二乘求解（底层是 LAPACK QR 分解），写多项式背景模型非常简洁：

```matlab
[X, Y] = meshgrid(((1:g) - 0.5)/g);                      % 网格中心坐标（归一化）
A = [ones(g^2, 1), X(:), Y(:), X(:).^2, X(:).*Y(:), Y(:).^2];
v = tileMedians(:);  keep = true(size(v));
for it = 1:5
    c = A(keep, :) \ v(keep);                            % 最小二乘
    r = v - A*c;  s = 1.4826 * median(abs(r(keep)));
    keep = r < 2.5*s;                                    % 剔除星云所在的块
end
```

```python
c, *_ = np.linalg.lstsq(A[keep], vals[keep], rcond=None)    # scripts/gradient_removal.py
```

```mathematica
c = LeastSquares[design /@ Pick[pts, keep], Pick[vals, keep]];   (* RemoveGradient *)
```

### 5.11 色彩空间与合成

| 功能 | MATLAB | NumPy 生态 | Wolfram |
|---|---|---|---|
| 拆合通道 | `I(:,:,1)`、`cat(3, R, G, B)` | `I[..., 0]`、`np.dstack([R, G, B])` | `ColorSeparate`、`ColorCombine`、`Transpose[{r, g, b}, {3, 1, 2}]` |
| 灰度 | `rgb2gray`（0.2989R + 0.5870G + 0.1140B） | `skimage.color.rgb2gray`（Rec.709 系数） | `ColorConvert[img, "Grayscale"]` |
| Lab | `rgb2lab` / `lab2rgb`，L 范围 0–100 | `skimage.color.rgb2lab`，L 范围 0–100 | `ColorConvert[…, "LAB"]`，L 范围 0–1 |
| HSV | `rgb2hsv` | `skimage.color.rgb2hsv` | `ColorConvert[…, "HSB"]` |
| 白平衡 | `chromadapt`、`illumgray`、`illumwhite`（R2017b） | 自己写（本仓库 `color_calibration.py`） | 自己写（本仓库 `ColorBalance`，非内置） |

**LRGB 合成（本仓库 `lrgb_combine`）**：

```matlab
lab = rgb2lab(rgb);
lab(:,:,1)   = 100 * L;                % 用高信噪比亮度替换 L（注意 0–100）
lab(:,:,2:3) = 1.5 * lab(:,:,2:3);     % 提高饱和度
out = min(max(lab2rgb(lab), 0), 1);
```

```python
lab = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2Lab)   # float 输入时 L 范围 0–100
lab[..., 0] = L * 100
lab[..., 1:] *= 1.5
out = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)
```

```mathematica
lab = ImageData@ColorConvert[Image[rgb], "LAB"];
lab[[All, All, 1]] = L;                (* 0–1 *)
lab[[All, All, 2 ;; 3]] *= 1.5;
ImageData@ColorConvert[Image[lab, ColorSpace -> "LAB"], "RGB"]
```

**SCNR 去绿**：三者都可以写成一行。MATLAB 是 `G = min(G, (R + B)/2)`（`min` 逐元素），NumPy 是 `np.minimum(G, (R + B)/2)`，
Wolfram 是 `g - Ramp[g - (r + b)/2]`。

### 5.12 反卷积

| 方法 | MATLAB（IPT） | NumPy 生态 | Wolfram |
|---|---|---|---|
| Richardson–Lucy | `deconvlucy(I, PSF, 20)` | `skimage.restoration.richardson_lucy(I, psf, 20)`；本仓库手写 FFT 版 | `ImageDeconvolve[img, psf, Method -> "RichardsonLucy", MaxIterations -> 20]` |
| 维纳 | `deconvwnr(I, PSF, nsr)` | `skimage.restoration.wiener` | `ImageDeconvolve[…, Method -> "Wiener"]` |
| 正则化 | `deconvreg` | `unsupervised_wiener` | `Method -> "Tikhonov"` / `"TotalVariation"` |
| 盲反卷积 | `deconvblind`（同时估计 PSF） | 无内置 | — |
| 高斯 PSF | `fspecial('gaussian', 2*ceil(3*s)+1, s)` | 自己构造，或 `astropy.convolution.Gaussian2DKernel` | `GaussianMatrix[{r, σ}]` |
| 振铃抑制 | `deconvlucy` 的 `WEIGHT` 参数（可屏蔽坏像素或饱和星）、`edgetaper` | 自己做遮罩 | 自己做遮罩 |

```matlab
s   = fwhm / 2.3548;
PSF = fspecial('gaussian', 2*ceil(4*s) + 1, s);
J   = deconvlucy(edgetaper(I, PSF), PSF, 20);
```

MATLAB 的 `deconvlucy` 最完整：带阻尼参数 `DAMPAR` 和权重 `WEIGHT`，可以直接实现天文中常用的「保护饱和星」。

### 5.13 小波

| 功能 | MATLAB（Wavelet Toolbox） | NumPy 生态（PyWavelets） | Wolfram |
|---|---|---|---|
| 多级分解 | `wavedec2(I, n, 'db4')` | `pywt.wavedec2(I, 'db4', level=n)` | `DiscreteWaveletTransform[img, DaubechiesWavelet[4], n]` |
| 平稳（不抽取）小波 | `swt2` | `pywt.swt2` | `StationaryWaveletTransform` |
| 一键降噪 | `wdenoise2`（R2019b+） | `skimage.restoration.denoise_wavelet` | `WaveletThreshold` |
| starlet（à trous，PI MLT 用的） | 无内置，手写 | 无内置，本仓库手写 `astrotools.starlet` | 无内置，`wolfram/AstroTools.wl` 手写 `StarletLayers` |

starlet 在 MATLAB 中可以用 `conv2` 的可分离形式写得很短：

```matlab
k = [1 4 6 4 1] / 16;  c = I;  layers = {};
for j = 0:n-1
    kh = zeros(1, 4*2^j + 1);  kh(1:2^j:end) = k;        % 带孔核
    r  = 2*2^j;
    nxt = conv2(kh, kh, padarray(c, [r r], 'symmetric'), 'valid');   % 先列后行
    layers{end+1} = c - nxt;  c = nxt;
end
layers{end+1} = c;                                       % 各层之和等于原图
```

---

## 6. 完整小流程：三种语言各写一遍

任务：读入 master FITS → 去背景 → 统计星点 → STF 拉伸 → 保存 16 位 TIFF。

### MATLAB

```matlab
I   = double(fitsread('process/master_L.fits'));
bkg = imresize(blockproc(I, [64 64], @(b) median(b.data(:))), size(I), 'bilinear');
S   = I - bkg;
sig = 1.4826 * mad(S(:), 1);
st  = regionprops(bwconncomp(bwareaopen(imgaussfilt(S, 1) > 2.5*sig, 3)), max(S, 0), ...
                  'WeightedCentroid', 'Area');
fprintf('%d stars\n', numel(st));

mtf = @(m, x) ((m - 1) .* x) ./ ((2*m - 1) .* x - m);
x   = rescale(S);  med = median(x(:));  c0 = max(med - 2.8 * 1.4826 * mad(x(:), 1), 0);
J   = mtf(mtf(0.25, (med - c0)/(1 - c0)), min(max((x - c0)/(1 - c0), 0), 1));
imwrite(im2uint16(J), 'L.tif');
```

所需工具箱：IPT（`blockproc`、`imgaussfilt`、`bwareaopen`、`regionprops`）、Statistics（`mad`）。

### Python

```python
import numpy as np, sys
sys.path.insert(0, 'scripts'); import astrotools as at
I = at.load_fits('process/master_L.fits')
S = I - at.background_map(I, 64)
st = at.detect_stars(S)
print(len(st['x']), 'stars, FWHM', np.median(st['fwhm']))
at.save_image('L.tif', at.stf(S))
```

### Wolfram

```mathematica
Get["wolfram/AstroTools.wl"];
{i0, hdr} = ReadFITS["process/master_L.fits"];
s = i0 - BackgroundMap[i0, 64];
st = DetectStars[s];
Print[Length[st], " stars, {FWHM, ecc} = ", StarShape[s, st]];
Export["L.tif", Image[AutoSTF[s], "Bit16"]];
```

对比：MATLAB 用**内置函数**就能完成大部分工作（代价是工具箱费用）；Python 依赖本仓库自己封装的 `astrotools`
（或 astropy/sep/photutils）；Wolfram 的内置函数最多，但 FITS 和天文专用部分需要自己写。

---

## 7. 生态、授权与性能

| 维度 | MATLAB | NumPy 生态 | Wolfram |
|---|---|---|---|
| 授权 | 商业授权；IPT、CV、Wavelet、Statistics、Parallel 等工具箱分别收费；有学生版和高校授权 | 免费（BSD 等许可证） | Mathematica 商业授权；Wolfram Engine 个人开发免费 |
| 开源替代 | Octave（`pkg load image`，兼容大部分 IPT 函数，速度较慢） | — | — |
| 天文生态 | 弱：有 `fitsread`/`fitswrite`，没有 WCS、测光、星表库 | **最强**：astropy、photutils、sep、astroalign、ccdproc、reproject、astroquery | 弱：通用图像函数强，内置天体数据库（`StarData` 等） |
| 交互界面 | `imtool`、Image Viewer App、Image Segmenter App、Registration Estimator App、Live Script | Jupyter、napari、matplotlib 控件 | 笔记本 + `Manipulate` |
| 执行速度 | 2015 年起有 JIT 执行引擎，循环也不算慢；内置函数是多线程 C/C++ | 需要向量化，或用 numba；OpenCV 很快 | 需要保持打包数组；`Compile` / `FunctionCompile` |
| GPU | `gpuArray`（Parallel Computing Toolbox），大量 IPT 函数直接支持 | CuPy（与 NumPy 接口兼容）、PyTorch | 有限的 `CUDALink` 支持 |
| 并行 | `parfor`、`blockproc(..., 'UseParallel', true)` | `multiprocessing`、dask、joblib | `ParallelMap`、`ParallelTable` |
| 部署 | MATLAB Compiler（收费）、MATLAB Runtime | 到处都能跑 | 需要 Wolfram 内核 |
| 文档质量 | 很好，例子丰富 | 分散在各个库 | 很好，每个函数都有大量例子 |

**怎么选**：
- 已经有 MATLAB 授权（高校、工业界）、偏工程、要 GPU、喜欢 App 式交互：**MATLAB**；
- 做天文、要免费、要自动化部署、要用 astropy 生态：**NumPy**；
- 研究算法、写带公式和图的笔记、快速做原型：**Wolfram**。

---

## 8. 迁移易错点

### MATLAB → NumPy

1. 下标从 0 开始，切片 `a[i:j]` 不含 j；`end` 改成 `-1`。
2. `*` 在 NumPy 中是**逐元素**乘，矩阵乘用 `@`。
3. 帧维度通常从第 3 维移到第 0 维：`median(cube, 3)` → `np.median(cube, axis=0)`。
4. `max(A, B)` → `np.maximum(A, B)`；`max(A, [], 3)` → `A.max(axis=2)`。
5. 切片是视图：`B = A(1:2:end, :)` 在 MATLAB 中得到拷贝，`B = A[::2]` 在 NumPy 中与 A 共享内存。
6. 整数溢出是回绕，不是饱和：图像运算前先转换成 float。
7. `regionprops` 的质心 `[x y]` 对应 skimage 的 `(row, col)`，顺序相反。
8. `std` 默认分母不同（MATLAB n−1，NumPy n）。
9. `imfilter` 默认补零，`scipy.ndimage` 默认反射。从 MATLAB 移植过来的结果在边缘会不同。
10. 内存布局：MATLAB 列优先，NumPy 行优先。`reshape` 的填充顺序不同，`I(:)` 等价于 `I.ravel(order='F')`。

### MATLAB → Wolfram

1. 下标同样从 1 开始，但要写双括号 `a[[i, j]]`；`end` 对应 `-1`。
2. `max(A, B)` 不能直接写成 `Max[A, B]`，要用 `A + Ramp[B - A]`。
3. 隐式扩展从尾维对齐；Wolfram 从第一维对齐。
4. 滤波参数是**半径**而不是窗口大小：`medfilt2(I, [3 3])` → `MedianFilter[I, 1]`。
5. 几何坐标 y 轴向上，原点在左下。
6. Lab 的 L 范围是 0–1，不是 0–100。
7. 不能用 `N`、`D`、`E`、`I`、`C`、`O` 做变量名，所以 MATLAB 习惯的 `I = imread(...)` 要换个名字。

### NumPy → MATLAB

1. 一切都是 double，记得用 `single` 节省内存。
2. 逐元素运算要写 `.*`、`./`、`.^`。
3. 函数大多在工具箱里，先确认授权；Octave 可以作为免费替代。
4. 没有命名空间：函数名可能被同名脚本或变量遮蔽（`which -all 名字` 可以查看）。
