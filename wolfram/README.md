# Wolfram Language（Mathematica / wolframscript）版深空后期

和 `process/`（Python）同一套流程的 Wolfram Language 实现：配准 → 叠加 → 去梯度 → 拉伸 → LRGB / SHO / HOO 合成，
外加反卷积、小波降噪、缩星等工具。目的是把 **PixInsight、Python、Wolfram** 三种做法放在一起对比。

> ⚠️ **测试状态**：作者的开发环境（Android Termux）无法运行 Wolfram Engine，这些 `.wl/.wls` 代码**还没有实际运行过**。
> 代码按 Wolfram Language 13/14 的文档语义编写，算法与已验证的 Python 版逐行对应。首次运行如果报错，
> 多半是某个内置函数的选项名或返回格式与版本不一致，按报错信息调整即可。

| 文件 | 对应 Python | 内容 |
|---|---|---|
| `AstroTools.wl` | `scripts/astrotools.py` | 程序包：FITS 读写、鲁棒统计、星点检测、配准、σ-clip 叠加、去梯度、拉伸、合成、WCS、小波、反卷积、缩星 |
| `stack.wls` | `process/stack.py` | 去热像素、配准、加权 σ-clip 叠加 → `wolfram/out/master_*.fits` |
| `compose.wls` | `process/compose.py` | 去条纹、去梯度、WCS 裁剪、拉伸、LRGB / SHO / HOO / HaLRGB |
| `tools.wls` | `scripts/*.py` | 单步工具：info / stretch / gradient / stars / deconv / starlet / wavelet / layers / starreduce / hoo |

## 安装与运行

1. 安装 [Wolfram Engine](https://www.wolfram.com/engine/)（个人开发免费）或 Mathematica，并完成激活；
2. 安装 [WolframScript](https://www.wolfram.com/wolframscript/)；
3. 原始 `*.fit` 放在仓库根目录（与 Python 版相同）。

```bash
wolframscript -file wolfram/stack.wls             # 全部滤镜，或: stack.wls 5 6 7
wolframscript -file wolfram/compose.wls           # 用 wolfram/out 的 master
wolframscript -file wolfram/compose.wls python    # 直接用 process/stack.py 产出的 master（对比两边合成）

wolframscript -file wolfram/tools.wls stars   process/master_L.fits
wolframscript -file wolfram/tools.wls deconv  process/master_L.fits L_dc.fits 4.2 20
wolframscript -file wolfram/tools.wls hoo     process/master_H.fits process/master_O.fits hoo.jpg
```

在 Mathematica 笔记本里交互使用：

```mathematica
Get["wolfram/AstroTools.wl"]
{a, hdr} = ReadFITS["process/master_H.fits"];
Image@AutoSTF[a]                                          (* 等同 PI 的 STF 自动拉伸预览 *)
Manipulate[Image@Stretch[a, t, k], {{t, 0.12}, 0.02, 0.4}, {{k, -1.5}, -4, 0}]   (* 拖动滑块调拉伸 *)
```

`Manipulate` 让 Wolfram 具备了 PI 那样的「实时预览 + 调参数」能力，这是 Python 脚本不太方便做到的。

## 三种做法对比：同一步骤，三种写法

### 1. 读 FITS

| PI | Python | Wolfram |
|---|---|---|
| File → Open | `fits.getdata(f)`（astropy） | 自己写的 `ReadFITS[f]`：解析 2880 字节头块，`BinaryReadList[s, "Integer16", n, ByteOrdering -> 1]`，再乘 BSCALE 加 BZERO |

Wolfram 自带 `Import[f, "FITS"]`，但各版本返回的结构（多 HDU、元数据格式）不一致，所以这里自己解析头。
FITS 格式本身很简单：80 字符一行的 ASCII 卡片，`END` 结束，按 2880 字节对齐，后面是大端二进制数据。

### 2. 星点检测

```python
# Python (sep / 自写 numpy)
objs = sep.extract(img - bkg, 5, err=bkg.globalrms, minarea=6)
```

```mathematica
(* Wolfram：二值化 -> 连通域 -> 区域测量，全是内置函数 *)
lab = MorphologicalComponents[UnitStep[GaussianFilter[sub, 2] - k s/2]];
ComponentMeasurements[{Image[sub], lab}, {"IntensityCentroid", "Total", "Max", "Count"}, #Count >= 3 &]
```

PI 对应 StarAlignment 内部的星点检测 / DynamicPSF。

### 3. 配准

| | PI StarAlignment | Python `stack.py` | Wolfram `stack.wls` |
|---|---|---|---|
| 初始匹配 | 三角形/多边形描述子 | astroalign 三角形；窄带用上一帧变换做预测 | **相位相关**（`Fourier`）求平移，所有滤镜通用 |
| 精配 | RANSAC | KD 树 + `cv2.estimateAffinePartial2D` | `Nearest` 互为最近邻 + `FindGeometricTransform` |
| 重采样 | Lanczos-3 | `cv2.warpAffine` Lanczos4 | `ImagePerspectiveTransformation[..., Resampling -> "Lanczos"]` |

```mathematica
d   = PhaseShift[refHP, HighpassBin[a]];                  (* 粗平移 *)
tf0 = TranslationTransform[{-4 d[[1]], 4 d[[2]]}];        (* 数组行向下、图像 y 向上：注意符号 *)
{tf, n, rms} = GuidedTransform[refXY, xy, tf0];           (* 20 → 6 → 3 px 逐步收紧 *)
{warped, mask} = WarpFrame[a, tf];
```

### 4. σ-clip 叠加

```python
med = np.nanmedian(cube, axis=0)                          # numpy：显式 axis
```

```mathematica
med = Median[cube];                                       (* Wolfram：对 3 维数组默认按第一维逐元素求中值 *)
sd  = 1.4826 Median[Abs[(# - med) & /@ cube]];
keep = vm (UnitStep[k sd - Abs[# - med]] & /@ cube);
master = Total[weights keep cube]/Total[weights keep];
```

PI 对应 ImageIntegration（Winsorized / Linear Fit Clipping 等更多剔除算法）。

### 5. 拉伸（STF / MTF）

三者公式完全相同：`MTF(m, x) = (m−1)x / ((2m−1)x − m)`，`m = MTF(target, 背景)`。

```mathematica
MTF[m_, x_] := ((m - 1) x)/((2 m - 1) x - m);
AutoSTF[a_] := ... MTF[MTF[0.25, (med - c0)/(1 - c0)], Clip[(x - c0)/(1 - c0), {0, 1}]]
```

### 6. 去梯度（ABE）

```mathematica
c = LeastSquares[design /@ pts, vals];                    (* 二维多项式最小二乘，迭代剔除亮块 *)
```

Python 用 `np.linalg.lstsq`，PI 用 ABE / DBE。

### 7. LRGB 合成

```mathematica
lab = ImageData@ColorConvert[Image[rgb], "LAB"];
lab[[All, All, 1]] = l;  lab[[All, All, 2 ;; 3]] *= sat;
ImageData@ColorConvert[Image[lab, ColorSpace -> "LAB"], "RGB"]
```

Python 用 `cv2.cvtColor(..., COLOR_RGB2Lab)`，PI 用 LRGBCombination。

### 8. 定位目标（WCS）

Python 用 `astropy.wcs.WCS(header).all_world2pix`；Wolfram 版自己实现了 TAN（日心投影）`WorldToPixel`：

```
ξ = cos δ sin(α−α0) / cos c,   η = (cos δ0 sin δ − sin δ0 cos δ cos(α−α0)) / cos c
cos c = sin δ0 sin δ + cos δ0 cos δ cos(α−α0)
pixel = CRPIX + CD⁻¹ · (ξ, η)
```

M27 的坐标先尝试从 Wolfram 知识库获取（`NebulaData["M27", "RightAscension"]`，需要联网），失败时用内置常数。
这是 Wolfram 独有的优势：**天文数据是内置的**（`StarData`、`NebulaData`、`GalaxyData`、`PlanetData`、`Sunrise` 等）。

### 9. 反卷积 / 降噪 / 缩星

| | PI | Python `scripts/` | Wolfram |
|---|---|---|---|
| 反卷积 | Deconvolution、BlurXTerminator | 手写 FFT Richardson–Lucy | `ImageDeconvolve[img, GaussianMatrix[...], Method -> "RichardsonLucy"]` |
| 小波降噪 | MLT / MMT | 手写 starlet | 手写 `StarletDenoise`（与 PI 一致）；或内置 `StationaryWaveletTransform` + `WaveletThreshold` |
| 双边/非局部降噪 | TGVDenoise、NoiseXTerminator | `cv2.bilateralFilter` | `BilateralFilter`、`NonlocalMeansFilter`、`TotalVariationFilter` |
| 星点遮罩 | StarMask | 手写阈值 + 形态学 | `Binarize` → `DeleteSmallComponents` → `Dilation` → `Blur` |
| 缩星 | MorphologicalTransformation | 手写最小值滤波 | `Erosion[img, DiskMatrix[r]]` |

## 总体对比

| 维度 | PixInsight | Python（本仓库） | Wolfram Language |
|---|---|---|---|
| 价格 | 约 €300 永久授权 | 免费开源 | Mathematica 商业授权；Wolfram Engine 个人开发免费 |
| 交互 | 最好：实时预览、遮罩、历史记录 | 弱：改参数后重跑 | 好：笔记本 + `Manipulate` 滑块实时调参 |
| 批处理 / 自动化 | WBPP 强，其余靠 PJSR 脚本 | 最好：任意环境、CI、服务器、手机 | 可以（wolframscript），但需要激活授权 |
| 代码量 | 不需要写 | 中等（numpy 底层操作多） | 最少：图像处理、几何变换、反卷积、小波、色彩空间都有内置函数 |
| 算法透明度 | 闭源，靠文档 | 完全透明 | 内置函数是黑盒，手写部分透明 |
| 天文专用功能 | 最全：SPCC、ImageSolver、Drizzle、局部归一化、AI 工具 | 靠 astropy / sep / astroalign / photutils 生态 | 通用图像处理强，天文生态弱；但内置天体数据库 |
| 性能 | C++ 多线程，最快 | numpy + OpenCV，较快 | 打包数组向量化后较快，逐像素循环很慢 |
| 内存 | 优化好 | float32，可控 | 默认 Real64，同样数据约是 Python 的 2 倍 |
| 部署 | 桌面 GUI | 到处都能跑 | 需要 Wolfram 内核与激活 |

**选择建议**：
- **出片、调色**：PixInsight（加 RC Astro 的 AI 工具）；
- **批处理、自动化、学习原理、在服务器或手机上跑**：Python；
- **研究算法、快速做原型、交互探索参数、写带公式和图的教程笔记**：Wolfram。同样的功能代码量大约是 Python 的一半，`Manipulate` 调参很顺手。

## 写 Wolfram 代码时的几个坑

1. **坐标系**：数组 `a[[row, col]]` 的第 1 行在上；`ComponentMeasurements`、`FindGeometricTransform`、
   `ImagePerspectiveTransformation[..., DataRange -> Full]` 用的是图像坐标，原点在左下、y 轴向上。
   像素 (i, j) 的中心是 `{j − 1/2, h − i + 1/2}`，平移量换算时 y 要反号。
2. **`Fourier` 的符号约定**和 numpy 相反（正指数），但互功率谱峰值的位置不受影响。
3. **`Max[a, b]` 不是逐元素运算**，作用于两个矩阵时返回一个标量。逐元素取最大用 `a + Ramp[b − a]`，
   逐元素取最小用 `a − Ramp[a − b]`；`Clip` 只能夹在常数之间。
4. **大写单字母符号是内置的**（`N`、`D`、`E`、`I`、`C`、`O`），不能当变量名，所以通道存放在 `m["O"]` 这样的关联里。
5. **保持打包数组（packed array）**：避免 `Table` 逐像素循环和混入 `Indeterminate`，否则会慢上百倍。
   缺失像素用 0/1 掩膜表示，而不是像 numpy 那样用 NaN。
6. **图像类型决定数值范围**：`"Byte"`、`"Bit16"` 图像的 `ImageData` 返回缩放到 [0,1] 的值，滤波函数也按 [0,1] 理解阈值。
   要保留原始 ADU 数值，就用 `Image[a, "Real32"]` 和 `ImageData[img, "Real32"]` 配对。
