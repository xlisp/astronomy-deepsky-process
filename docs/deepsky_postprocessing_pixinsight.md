# 深空后期全流程：为什么要 DBE，PixInsight 整套流程，星云 / SHO / 星系分别怎么处理

这篇是**流程篇**：从一堆原始子帧到一张成品图，业余天文摄影在 PixInsight（PI）里要经过哪些步骤，每一步为什么放在那个位置；
然后分别给出**发射星云（宽带）**、**窄带 SHO / HOO**、**星系**三类目标的具体流程和差异。

每一步都注明本仓库对应的 Python 脚本，方便对照原理。校准帧的来历见 [calibration_frames_history.md](calibration_frames_history.md)，叠加与反卷积的原理见 [stacking_deconvolution_history.md](stacking_deconvolution_history.md)。

- [1. 总览：线性阶段与非线性阶段](#1-总览线性阶段与非线性阶段)
- [2. 为什么要 DBE（去背景梯度）](#2-为什么要-dbe去背景梯度)
- [3. 预处理：从子帧到主帧](#3-预处理从子帧到主帧)
- [4. 线性后期](#4-线性后期)
- [5. 拉伸：线性 → 非线性](#5-拉伸线性--非线性)
- [6. 非线性后期](#6-非线性后期)
- [7. 目标一：发射星云（宽带 RGB / OSC，HaRGB）](#7-目标一发射星云宽带-rgb--oschargb)
- [8. 目标二：窄带 SHO / HOO](#8-目标二窄带-sho--hoo)
- [9. 目标三：星系（LRGB）](#9-目标三星系lrgb)
- [10. 其他目标简述：反射星云、暗星云、星团、IFN](#10-其他目标简述反射星云暗星云星团ifn)
- [11. 工具演变时间线](#11-工具演变时间线)
- [12. 常见错误清单](#12-常见错误清单)
- [13. 参考](#13-参考)

> 说明：PI 的 Process 名称以 1.8.x / 1.9.x 版本为准。带 XTerminator 的是 RC Astro 的第三方插件（付费），GraXpert、StarNet 是免费第三方工具。

---

## 1. 总览：线性阶段与非线性阶段

整个流程最重要的分界线是**拉伸（stretch）**。

- **线性阶段**：像素值与光子数成正比。所有依赖物理模型的操作必须在这里做：校准、叠加、去梯度、颜色校准、反卷积。
- **非线性阶段**：拉伸之后像素值已经被曲线压缩，比例关系不再成立。这里做的是「审美」操作：对比度、饱和度、局部增强、星点缩小。

```
┌──────────── 预处理 ────────────┐ ┌──────────────── 线性后期 ────────────────┐ ┌─ 拉伸 ─┐ ┌──────── 非线性后期 ─────────┐
Blink → 校准 → 坏点 → 去马赛克 →   裁边 → DBE → 颜色校准 → 反卷积 → 线性降噪 →   STF/HT   曲线 → 局部对比 → 饱和度 →
子帧筛选 → 配准 → 局部归一化 →     (→ 去星)                                        GHS      星点缩小 → 加回星 → 最终降噪 →
叠加 → (Drizzle)                                                                   MaskedSt 锐化 → 输出
```

为什么顺序很重要，用一句话概括：**每一步都假设它之前的问题已经被修好**。

| 步骤 | 依赖前面哪一步 | 原因 |
|---|---|---|
| 颜色校准 | DBE | 梯度会让背景偏色，背景中和要求背景是平的 |
| 反卷积 | DBE、线性 | 卷积模型 `观测 = 真实 ⊗ PSF` 只在线性数据上成立 |
| 降噪 | 线性 / 非线性均可 | 线性时噪声统计还是泊松 + 高斯，模型更准确 |
| 拉伸 | 颜色校准 | 拉伸是非线性的，先拉再校色会让不同亮度的颜色比例不一致 |
| 星点缩小 | 拉伸后 | 形态学操作作用于视觉上的星点大小 |

PI 里 **STF（ScreenTransferFunction）** 只是一个**显示用的临时拉伸**，不改数据，让你在线性阶段也能看清图像。很多初学者以为看到的就是数据——不是。

---

## 2. 为什么要 DBE（去背景梯度）

### 2.1 梯度从哪里来

叠加出的主帧，背景几乎从来不是平的。常见来源：

| 来源 | 形态 | 性质 |
|---|---|---|
| **光污染**（城市灯光） | 一侧亮一侧暗的线性 / 二次渐变 | 加性 |
| **月光** | 大尺度渐变，方向随月亮位置变化 | 加性 |
| **地平线附近的大气**（低高度角） | 靠近地平线一侧更亮 | 加性 |
| **气辉（airglow）** | 大片不规则的绿色 / 红色波纹，暗夜也有 | 加性 |
| **暗角残余**（平场没拍或不准） | 中心亮四角暗的同心圆 | **乘性** |
| **杂散光 / 内反射** | 不规则亮斑 | 加性 |
| **一夜之间的变化** | 不同子帧的梯度方向不同，叠加后变成复杂形态 | 加性 |

这些信号的特点是**空间尺度远大于目标**：它们在整幅图上缓慢变化，而星云和星系有清晰的结构。

### 2.2 为什么不能靠「拉伸 + 曲线」解决

梯度的幅度往往比目标的微弱部分还大。一张城市郊区拍的星云，背景左右可能差 20%，而星云外围的暗弱结构只比背景高 1–2%。
拉伸会把这 20% 一起放大，画面一边发白一边死黑；用曲线去压，暗弱结构也一并被压掉了。

另外，梯度通常**各通道不一样**（光污染偏橙，月光偏蓝，气辉偏绿），不去掉就无法做颜色校准。

### 2.3 DBE 怎么做

DBE = **DynamicBackgroundExtraction**。思路是：

1. 在图上放若干**采样点**，只放在「确定是背景」的地方——避开星、星云、星系
2. 每个采样点取一小块区域的稳健均值（剔除其中的星）
3. 用这些点拟合一个**光滑的二维曲面**（PI 用的是薄板样条 / 2D 曲面样条）作为背景模型
4. **减去**（加性梯度）或**除以**（乘性的暗角）这个模型

ABE（AutomaticBackgroundExtractor）则是自动在网格上取样并拟合多项式，不需要手放采样点，适合背景干净、目标占比小的图。

本仓库 [`scripts/gradient_removal.py`](../scripts/gradient_removal.py) 实现的是 ABE 思路：网格中值采样 + 2D 多项式最小二乘 + 迭代剔除亮于模型的样本。

### 2.4 减还是除

- **减（Subtraction）**：光污染、月光、气辉——这些是额外叠加的光。绝大多数情况选这个。
- **除（Division）**：暗角——光被按比例削弱了。只在**没有平场**时才用来补救；但它同时会把加性梯度也当成乘性处理，结果往往不完美。

正确的做法是用平场修乘性、用 DBE 修加性，各管各的。

### 2.5 DBE 的风险

DBE 最大的风险是**把目标当背景减掉**。如果采样点落在了星云的暗弱外围、星系的晕、或者整幅充满星云的视场（比如天鹅座、猎户座大片区域），模型会把这部分信号拟合进去，减完之后暗弱结构就没了，还会在目标周围产生一圈暗环。

应对：

- 采样点宁少勿多，只放在你确定的背景上
- 视场被星云填满时，降低模型自由度（ABE 用低阶多项式，DBE 加大 smoothing）
- 检查背景模型图（DBE 可以输出模型）：模型里如果出现了目标的形状，就是采错了
- 新工具（见第 11 节）：GraXpert 的 AI 背景模型、PI 的 MultiscaleGradientCorrection（MGC，用全天参考巡天数据作为背景参考，2024 年加入）

### 2.6 去梯度的历史

| 时期 | 做法 |
|---|---|
| 底片时代 | 基本无法处理；冲洗时用局部减薄 / 加厚，或暗房里遮挡 / 加光（dodge & burn） |
| 专业 CCD | 天空平场 + 逐行 / 逐区拟合天空背景；测光时用环形孔径测局部天空 |
| 2000 年代初 业余 | Photoshop 里复制图层 → 大半径高斯模糊 / 中值 → 用「差值」或「减去」混合模式；手工画渐变 |
| 约 2006 起 | PixInsight 加入 ABE 和 DBE（Juan Conejero），把背景建模变成标准步骤 |
| 约 2008 | Russell Croman 的 GradientXTerminator（Photoshop 插件） |
| 2022 前后 | GraXpert（免费开源）发布，后来加入 AI 背景提取 |
| 2024 | PI 加入 GradientCorrection 和 MultiscaleGradientCorrection（MGC，基于 MARS 参考巡天数据库） |

---

## 3. 预处理：从子帧到主帧

现在一般用 **WBPP**（WeightedBatchPreprocessing 脚本）一键跑完下面大部分步骤，但了解每一步才能在出问题时排查。

| # | PI Process | 作用 | 本仓库 |
|---|---|---|---|
| 1 | **Blink** | 快速逐帧翻看，删掉云、飞机、跟踪失败、结露的帧 | `scripts/blink.py` |
| 2 | **ImageIntegration**（合成主校准帧） | 偏置 / 暗场 / 平场各自叠加成 master | `scripts/calibrate.py master` |
| 3 | **ImageCalibration** | 亮场减暗场、除平场 | `scripts/calibrate.py apply` |
| 4 | **CosmeticCorrection** | 修残余热像素 / 冷像素 / 坏列 | `calibrate.py --cosmetic` |
| 5 | **Debayer** | 彩色相机（OSC / 单反）的 Bayer 阵列插值成 RGB | — |
| 6 | **SubframeSelector** | 测每帧的 FWHM、偏心率、信噪比、星数，剔除差的、给好的更高权重 | `scripts/subframe_selector.py` |
| 7 | **StarAlignment** | 用星点三角形匹配做配准（平移 + 旋转 + 缩放，必要时畸变校正） | `process/stack.py`（astroalign） |
| 8 | **LocalNormalization** | 把每帧的背景和亮度局部地匹配到参考帧，让梯度不同的帧可以一起叠 | — |
| 9 | **ImageIntegration** | 加权平均 + 离群值剔除（sigma clip / winsorized / linear fit / ESD） | `process/stack.py` |
| 10 | **DrizzleIntegration** | 欠采样且抖动足够时提高分辨率；OSC 可用 Bayer Drizzle 跳过插值 | — |

**单色相机**每个滤镜分别走一遍，得到 master_L、master_R、master_G、master_B、master_Ha、master_OIII、master_SII。
**彩色相机**得到一张 RGB 主帧；双窄带滤镜（L-eXtreme 等）拍的 OSC 数据可以用脚本拆出 Ha（R 通道）和 OIII（G+B）。

### 子帧权重为什么重要

不同帧质量差别很大（视宁度、透明度、高度角）。WBPP 默认用 **PSF Signal Weight / PSFSNR** 一类的指标给权重，比单纯按噪声加权更能反映「这帧对最终图像的贡献」。

---

## 4. 线性后期

### 4.1 DynamicCrop —— 先裁边

配准后各帧位置不同，叠加结果的边缘只有部分帧覆盖，噪声大、背景怪。**必须先裁掉**，否则 DBE 会把边缘当成梯度来拟合，颜色校准也会被干扰。

### 4.2 DBE / ABE / GraXpert —— 去梯度

见第 2 节。单色数据每个通道分别做；彩色数据可以合成后做（DBE 支持 RGB）。

### 4.3 合成 RGB（单色相机）

**ChannelCombination** 把 R、G、B 三个主帧合成彩色图。合成前建议用 **LinearFit** 把三通道的亮度水平拉到一致，方便后续颜色校准。

### 4.4 颜色校准

| Process | 原理 | 时期 |
|---|---|---|
| **BackgroundNeutralization** | 让背景 R = G = B（中性灰） | 早期 |
| **ColorCalibration** | 选一个白参考（一批星或一个星系）让它变白 | 早期 |
| **PhotometricColorCalibration（PCC）** | Plate solve 后查星表（APASS 等）里星的颜色指数，按真实颜色定标 | 2018 前后 |
| **SpectrophotometricColorCalibration（SPCC）** | 查 Gaia DR3 的光谱数据，结合你的**相机 QE 曲线和滤镜透过率曲线**计算每颗星应该的颜色 | 2022/2023 |

SPCC 是目前最「物理」的做法：它知道你用的是哪个滤镜、哪个传感器，所以不同设备拍出来的颜色能对得上。
白参考常用「Average Spiral Galaxy」（星系平均光谱）或 G2V（太阳型星）。

本仓库 [`scripts/color_calibration.py`](../scripts/color_calibration.py) 是不查星表的简化版：背景中和 + 假设星点中值为白色。

### 4.5 反卷积 —— 恢复被大气和光学模糊掉的细节

- 传统：**Deconvolution**（Richardson–Lucy / 正则化 van Cittert），需要先用 **PSFImage / DynamicPSF** 测出 PSF，配合星点保护蒙版（deringing）和目标蒙版，参数难调。
- 现在主流：**BlurXTerminator**（BXT，Russell Croman，2022）。基于神经网络，自动处理各位置不同的 PSF（场曲、慧差），还能修正星点形状、分别控制星点和非星点的锐化力度。

反卷积**必须在线性数据上**，并且在拉伸之前；也最好在降噪之前（降噪会改变噪声统计，误导反卷积）。
本仓库：[`scripts/deconvolve.py`](../scripts/deconvolve.py)。原理和历史见 [stacking_deconvolution_history.md](stacking_deconvolution_history.md) 第二部分。

### 4.6 线性降噪（可选）

- 传统：**MultiscaleLinearTransform（MLT）** / **TGVDenoise**，配合亮度蒙版（只降背景，保护亮区）
- 现在：**NoiseXTerminator**（NXT，2021）在线性或非线性阶段都能用
- PI 自带：**MultiscaleMedianTransform**、2023 年后的 **NoiseX** 类 AI 工具也逐渐出现

线性阶段做轻度降噪，拉伸后再做一次轻度降噪，通常比一次重降噪自然。

### 4.7 去星（可选，但现在几乎是标配）

**StarXTerminator**（SXT，2022）或 **StarNet / StarNet++**（Nikita Misiura，2019 起，免费）把图像分成「无星图」和「星点图」两张。

好处：星云可以放心大力拉伸、提对比、加饱和度，而不会让星点膨胀、发白、变色；最后再把（缩小过的）星点加回去。
线性阶段去星时，要在 SXT 里选择 unscreen 模式，或者用减法得到星点图，保证之后能正确加回。

---

## 5. 拉伸：线性 → 非线性

叠加后的线性图像 99% 的像素挤在直方图最左边的一小段里（天空背景很暗，目标也只比背景亮一点点），必须用非线性曲线把暗部拉开。

| 方法 | 特点 |
|---|---|
| **HistogramTransformation（HT）** | 经典：设黑点 + 中间调传递函数（MTF）。常用做法是把 STF 的自动参数直接拖到 HT 上应用 |
| **ArcsinhStretch** | 反双曲正弦拉伸，**保持颜色比例**，星点颜色不会被拉白；源自 Lupton 等 2004 年 SDSS 彩色图的方法 |
| **MaskedStretch** | 多次迭代的小幅拉伸 + 蒙版保护亮区，星点不易膨胀，适合星系 |
| **GeneralizedHyperbolicStretch（GHS）** | 2022 年 Mike Cranfield 和 David Payne 提出：可以指定「拉伸焦点」（在哪个亮度区间拉得最狠）、保护高光和暗部，控制最精细 |
| **Statistical Stretch / SetiAstro 等脚本** | 社区脚本，按目标中值自动拉伸 |

本仓库 [`scripts/stretch.py`](../scripts/stretch.py) 实现了 AutoSTF / 手动 MTF / asinh 三种（GHS 未实现）。

**分开拉伸**：如果做了去星，无星图可以大力拉，星点图单独温和拉伸（或者用线性星点图 + 少量拉伸），这是现在的标准做法。

---

## 6. 非线性后期

| 步骤 | PI Process | 说明 | 本仓库 |
|---|---|---|---|
| 曲线 | **CurvesTransformation** | S 曲线提对比；单独调 R/G/B、饱和度（S）、色相（H）曲线 | — |
| 局部对比 | **LocalHistogramEqualization（LHE）** | 局部直方图均衡（CLAHE），增强星云内部的明暗结构 | — |
| 高动态范围 | **HDRMultiscaleTransform（HDRMT）** | 压缩亮核心（M42 核心、星系核），把被拉爆的高光细节找回来 | — |
| 多尺度锐化 | **MultiscaleLinearTransform / MMT** | 小波分层：给中小尺度层加权锐化，大尺度层不动 | `scripts/wavelet.py` |
| 去绿 | **SCNR** | 去除 RGB 图里不自然的绿色（自然界几乎没有绿色深空天体，OIII 除外）；SHO 里用来压绿 | `pixelmath.py`（表达式 `min(G, (R+B)/2)`） |
| 饱和度 | **ColorSaturation / CurvesTransformation S 曲线** | 配合亮度蒙版，只提星云 / 星系的饱和度，不提背景（避免彩色噪声） | — |
| 星点缩小 | **MorphologicalTransformation（Erosion / Morphological Selection）** + 星点蒙版 | 让星点不喧宾夺主；去星流程下可直接对星点图做 | `scripts/star_reduce.py` |
| 加回星点 | **PixelMath** | `~((~starless)*(~stars))`（screen 混合）或 `starless + stars` | `pixelmath.py` |
| 最终降噪 | NXT / MMT / TGVDenoise | 轻度 | — |
| 输出 | **Resample / IntegerResample**、**ICCProfileTransformation** | 按需缩小、转 sRGB、存 16-bit TIFF 和 JPEG | — |

### 蒙版贯穿始终

PI 后期的核心技巧是**蒙版（Mask）**：亮度蒙版、星点蒙版（StarMask）、范围蒙版（RangeSelection）、颜色蒙版（ColorMask 脚本）。几乎每个增强操作都应该只作用于该作用的区域：

- 降噪 → 反向亮度蒙版（作用于暗部）
- 锐化 / 局部对比 → 亮度蒙版（作用于亮部），并排除星点
- 饱和度 → 目标区域
- 颜色微调 → 颜色蒙版（只选红色 / 青色区域）

---

## 7. 目标一：发射星云（宽带 RGB / OSC，HaRGB）

典型目标：M42 猎户座大星云、M8 礁湖、M16 鹰、M20 三叶、NGC 7000 北美、IC 1805 心状。

### 特点

- 主要是 Hα（656.3 nm）红色发射 + 少量 OIII（500.7 nm）青色 + Hβ，常伴随反射星云（蓝）和暗星云
- 亮度动态范围大（M42 核心与外围差上万倍）
- 往往占满视场 → **DBE 难度大**

### 流程

```
1. WBPP 预处理 → master RGB（OSC）或 master R/G/B(/L/Ha)
2. DynamicCrop
3. DBE（采样点少而谨慎，避开所有星云；视场充满星云时考虑 GraXpert / MGC）
4. （单色）ChannelCombination + LinearFit
5. SPCC（或 PCC / BN + CC）
6. BXT 反卷积
7. （可选）Ha 混入：
     先对 Ha 做 DBE；
     连续谱扣除：Ha_net = Ha − k·(R − median(R))，k 使星点在 Ha_net 里消失；
     NBRGBCombination 或 PixelMath：R' = R + w·Ha_net
8. SXT 去星 → starless + stars
9. 线性降噪（NXT 轻度）
10. 无星图：GHS / HT 拉伸；星点图：温和拉伸
11. 无星图：
      HDRMT 压亮核心（M42 必做）
      LHE 两次：大尺度（kernel 200–300）提整体结构，小尺度（kernel 30–60）提细节
      CurvesTransformation 对比 + 饱和度（亮度蒙版）
      SCNR（如偏绿）
12. 星点图：星点缩小 / 降饱和
13. PixelMath screen 合并
14. 最终降噪、轻锐化、输出
```

### 要点

- **Ha 混入**是宽带星云最有效的增强：Ha 窄带不受光污染影响、信噪比高，能补出 RGB 拍不到的暗弱红色结构。直接加 Ha 会让星点发红变大，所以要先做连续谱扣除或在无星图上加。
- **L 通道**：发射星云用 Ha 作亮度（HaRGB）往往比 L 更好；反射星云、星系用 L。
- M42 这类极高动态范围目标常用**多曝光 HDR**：短曝光（几秒）拍核心，长曝光拍外围，用 **HDRComposition** 合并。

---

## 8. 目标二：窄带 SHO / HOO

### 8.1 窄带是什么，为什么能在城市拍

| 线 | 波长 | 来源 | 颜色（真实） |
|---|---|---|---|
| **Hα** | 656.3 nm | 氢电离区复合 | 深红 |
| **[SII]** | 671.6 / 673.1 nm | 一次电离硫，常在激波、电离前沿 | 深红（比 Ha 更红） |
| **[OIII]** | 495.9 / 500.7 nm | 二次电离氧，需要更高能量，多在热星附近和行星状星云 | 青绿 |

窄带滤镜带宽只有 3–7 nm，把 99% 的光污染和月光挡在外面。所以窄带可以在城市里、满月下拍。
代价是信号也被大幅削减，需要**几小时到几十小时**的总曝光，SII 和 OIII 通常比 Ha 弱很多（SII 可能只有 Ha 的 1/5–1/10）。

### 8.2 为什么叫「哈勃色板」

Ha 和 SII 在真实颜色里都是红色，直接按真实颜色合成就看不出区别。1995 年 Jeff Hester 和 Paul Scowen 用 HST 的 WFPC2 拍鹰状星云 M16「创生之柱」时，把三个窄带**按波长顺序**映射到 RGB：

```
SII (672 nm, 最红)   → R
Ha  (656 nm)         → G
OIII (501 nm, 最蓝)  → B
```

这就是 **SHO**，也叫 **Hubble Palette**。颜色不是真实的，但按波长顺序保留了「越红越红」的物理直觉，并且能一眼看出不同元素的分布。
因为 Ha 通常最强又被映射到 G，原始 SHO 合成会非常绿，所以后期需要大量的**去绿和调色**——典型成品是金黄色 + 青蓝色。

其他常见映射：

| 色板 | R | G | B | 特点 |
|---|---|---|---|---|
| **SHO** | S | H | O | 哈勃色板，金黄 + 青蓝 |
| **HOO** | H | O | O | 双窄带，接近真实颜色（红 + 青），双窄带 OSC 滤镜常用 |
| **HSO / OHS** | 各种排列 | | | 不同风格 |
| **Foraxx** | 动态：`R = (Oiii^~Oiii)*Sii + ~(Oiii^~Oiii)*Ha` 等 | | | Paulo Vieira（Foraxx）提出的动态 PixelMath 色板，按 OIII 强度在 Ha 和 SII 间混合，一步得到金蓝配色，减少调色工作 |
| **HaRGB / Ha+OIII 混入 RGB** | RGB + 窄带 | | | 真彩色基础上增强 |

本仓库 `process/compose.py` 生成了 `M27_SHO`、`M27_HOO` 和 `M27_HaLRGB`，可以直接对比。

### 8.3 SHO 流程

```
1. 每个滤镜分别 WBPP → master_Ha, master_OIII, master_SII
2. 三张都 DynamicCrop（用同一个裁剪框）
3. 三张分别 DBE（窄带梯度一般很小，但 OIII 受月光影响较大）
4. 三张分别 BXT（或只对 Ha 做反卷积，S/O 做轻度）
5. 三张分别 NXT 线性降噪（S 和 O 噪声大，可以更重）
6. 三张分别 SXT 去星，保留 Ha 的星点图（或另拍 RGB 星点）
7. 线性匹配：
     做法 A：LinearFit 把 S、O 匹配到 Ha 的水平 → 线性合成后整体拉伸
     做法 B：三张无星图**分别**拉伸到相近的背景和亮度 → 非线性合成（更常用，可控性更好）
8. 合成：ChannelCombination（S→R, H→G, O→B）或 PixelMath（Foraxx 等）
9. 调色：
     SCNR 去绿（Average Neutral，强度 0.5–1.0）
     CurvesTransformation 的色相曲线 / ColorMask 脚本：
        绿 → 黄 / 金
        洋红 → 去除（星点和背景常出现洋红）
        青 / 蓝 → 增强
     饱和度提升
10. HDRMT / LHE / 曲线 提结构（同星云流程）
11. 星点：
      做法 A：用 RGB 短曝光单独拍星点，得到真彩色星（最佳）
      做法 B：用窄带星点图，降饱和或做成白色
      做法 C：用 SHO 星点，去洋红
12. PixelMath screen 加回星点
13. 最终降噪、输出
```

### 8.4 SHO 的关键问题

| 问题 | 原因 | 解决 |
|---|---|---|
| 整幅画面绿 | Ha 最强且映射到 G | SCNR + 色相曲线；或分别拉伸时压低 Ha；或用 Foraxx 动态色板 |
| 星点洋红色光晕 | 三个滤镜的星点大小 / 亮度不一致，G 通道星点小 → R+B 剩下洋红 | 去星后单独处理星点；反转 → SCNR → 反转（「去洋红」技巧） |
| OIII / SII 噪声大 | 信号弱 | 更长曝光；更重降噪；用 Ha 作为亮度（L = Ha）合成 |
| 三通道细节不对齐 | 色差、不同夜视宁度 | StarAlignment 时三通道都配准到同一参考；BXT 各通道统一 PSF |
| 颜色「脏」 | 背景不中性 | 合成前每通道背景都要拉到同一水平 |

### 8.5 HOO

双窄带 OSC 滤镜（Optolong L-eXtreme、Antlia ALP-T 等）现在非常流行：一台彩色相机同时拍 Ha 和 OIII。

```
R  = Ha
G  = OIII   （或 0.5·Ha + 0.5·OIII 做「合成绿」，让颜色更丰富）
B  = OIII
```

调色比 SHO 简单得多，结果是红 + 青的近真实配色。也可以用 Ha 和 OIII 合成一个「伪 SII」做 SHO 风格，但这只是审美，不包含 SII 的物理信息。

---

## 9. 目标三：星系（LRGB）

典型目标：M31 仙女座、M33 三角座、M51 涡状、M81/M82、M101、M106、NGC 891。

### 特点

- **目标小**：大部分星系只有几个到十几个角分，焦距要长、像素尺度要细 → **视宁度和跟踪精度决定成败**
- **核心极亮、旋臂和晕极暗**：动态范围大
- 宽带目标，**不能用窄带过滤光污染** → 需要暗夜、DBE 至关重要
- 颜色有物理意义：核心偏黄（老年恒星），旋臂偏蓝（年轻恒星），点缀粉红色 HII 区（Ha）
- 细节靠**分辨率**：尘埃带、HII 区、旋臂结构 → 反卷积是关键

### 流程

```
1. WBPP（SubframeSelector 严格筛选：按 FWHM 剔除视宁度差的帧，L 通道尤其要严格）
2. DynamicCrop
3. DBE：
     采样点远离星系，**特别要避开星系的晕和外围暗弱旋臂**
     星系周围常有 IFN（银河系分子云反射光），不要把它当梯度减掉——除非你就是不要它
4. RGB：ChannelCombination + SPCC（白参考选 Average Spiral Galaxy）
5. L：BXT 反卷积（星系图里最关键的一步，参数可以比星云更激进）
   RGB：BXT 只做轻度或「correct only」
6. 线性降噪（NXT 轻度，L 更轻）
7. 拉伸：
     L：MaskedStretch 或 GHS（焦点放在晕的亮度上，同时保护核心）
     RGB：ArcsinhStretch 或 GHS（保持颜色）
     两者的背景和亮度要匹配
8. LRGB 合成：
     LRGBCombination（L 提供亮度细节，RGB 提供颜色；色度降噪选项可用）
     或：RGB → CIE L*a*b*，用拉伸后的 L 替换 L*
9. Ha 增强 HII 区（如果拍了 Ha）：
     连续谱扣除：Ha_net = Ha − k·R
     只把 Ha_net 加到 R 通道（和少量 L），HII 区变成粉红色点缀
10. HDRMT：恢复核心细节（M31、M81 核心）
11. LHE：小 kernel 提旋臂和尘埃带对比（亮度蒙版保护背景）
12. 多尺度锐化：MLT 小尺度层（星点蒙版保护星）
13. 饱和度：只对星系提，背景不动
14. 星点缩小：星系图里星点不要太抢眼，但也不要缩到消失
15. 最终降噪、输出
```

### 星系与星云流程的区别

| 方面 | 发射星云 | 星系 |
|---|---|---|
| 亮度通道 | 常用 Ha | 用 L |
| 光污染 | 窄带可规避 | 必须暗夜 + DBE |
| DBE 难点 | 星云充满视场 | 保护晕和外围暗弱结构 |
| 反卷积 | 中等 | **核心步骤**，决定细节 |
| 子帧筛选 | 宽松 | **严格**按 FWHM |
| 去星 | 几乎必做 | 可做可不做（星系和星点尺度接近，去星有时会伤到 HII 区和星团） |
| 颜色 | 夸张也可接受 | 尽量物理准确（SPCC），黄核蓝臂 |
| 拉伸 | 大力 | 控制核心，MaskedStretch / GHS |
| 窄带 | 主角 | 只作为 Ha 点缀 |

### 星系的特殊技巧

- **Drizzle**：焦距不够、欠采样时用 2× Drizzle 提高分辨率（需要充分抖动）
- **幸运成像**（lucky imaging）：亮星系核心可以拍大量 1–10 s 短曝光，只挑最锐的 10% 叠加
- **双星系 / 星系群**（M81/M82、Leo Triplet、Markarian's Chain）：DBE 采样点要放在星系之间的真正背景上
- **侧向星系**（NGC 891、NGC 4565）：尘埃带是重点，反卷积 + 小尺度 LHE

---

## 10. 其他目标简述：反射星云、暗星云、星团、IFN

| 目标 | 例子 | 要点 |
|---|---|---|
| 反射星云 | M45 昴星团、M78、IC 2118 | 宽带蓝色，没有窄带可用；亮星的光晕和内反射多，DBE 要小心；L 通道为主 |
| 暗星云 | B33 马头背景、LDN 1251、Barnard 系列 | 需要极深的曝光和极平的背景；DBE 几乎没有「真背景」可采样，常用低阶模型或 MGC |
| IFN（集成通量星云） | M81 周围、北天高银纬区 | 银河系尘埃反射银河星光，极暗；需要暗夜 + 数十小时 + 极准的平场；DBE 稍不注意就被减掉 |
| 球状星团 | M13、M3、ω Cen | 星点是主角：不去星；核心用 HDRMT；颜色校准要准（红巨星、蓝离散星） |
| 疏散星团 | M35、双星团 | 浅拉伸，保留星点颜色，ArcsinhStretch |
| 行星状星云 | M27、M57、NGC 7293 | 小目标：长焦 + 反卷积；HOO / SHO 都适合；外晕极暗，需要窄带长曝光（本仓库 M27 示例） |
| 超新星遗迹 | 面纱星云、IC 443 | Ha + OIII 丝状结构；HOO 最经典；去星后大力拉伸 |

---

## 11. 工具演变时间线

```
1990s      MaxIm DL、CCDSoft、Iris（Christian Buil）：校准 + 叠加 + 基础处理
           Photoshop 成为业余后期主力：曲线、图层、手工去梯度
约 2003–04 PixInsight 首个版本（Pleiades Astrophoto，Juan Conejero，Vicent Peris 等）
约 2006    DeepSkyStacker 免费发布：业余标准叠加软件
约 2006 起 PI 加入 ABE / DBE；小波（ATrousWaveletTransform）、HDRWaveletTransform
2008 前后  GradientXTerminator（Photoshop）；Noel Carboni 的 Astronomy Tools 动作集
2010s 初   PI：MultiscaleLinearTransform、TGVDenoise、LocalNormalization、MaskedStretch
2012       PI BatchPreprocessing 脚本普及
2016–17    Astro Pixel Processor（APP）、Siril（开源）成熟
2018 前后  PI PhotometricColorCalibration（PCC）
2019       StarNet（Nikita Misiura）：第一个被广泛使用的神经网络去星工具
2019–21    WBPP 2.x 重写：自动分组、加权、LocalNormalization 集成
2021       NoiseXTerminator（RC Astro）
2022       StarXTerminator、BlurXTerminator（RC Astro）—— AI 后期全面普及
2022       GeneralizedHyperbolicStretch（GHS）
2022–23    SPCC（基于 Gaia DR3 光谱）
2022 前后  GraXpert（免费开源去梯度，后加入 AI 模型）
2024       PI GradientCorrection、MultiscaleGradientCorrection（MARS 参考数据）
```

可以看到 2021–2022 年是一个分水岭：RC Astro 的三个 XTerminator 把过去最依赖经验的降噪、反卷积、去星三步变成了「几乎一键」，业余后期的门槛和流程形态都随之改变。
但它们取代的是「怎么做」，没有取代「为什么这么做和按什么顺序做」——线性 / 非线性的分界、DBE 在颜色校准之前、反卷积在拉伸之前，这些都没有变。

---

## 12. 常见错误清单

| 错误 | 后果 | 正确做法 |
|---|---|---|
| 没裁边就 DBE | 边缘被当梯度，模型扭曲 | 先 DynamicCrop |
| DBE 采样点放在星云 / 星系晕上 | 暗弱结构被减掉，目标周围出现暗环 | 只采真背景，检查背景模型图 |
| 先拉伸再颜色校准 | 颜色比例错误，亮暗区颜色不一致 | 颜色校准在线性阶段 |
| 拉伸后才反卷积 | 卷积模型不成立，产生振铃和伪影 | 反卷积在线性阶段 |
| 把 STF 当成真实拉伸 | 保存后发现图是黑的 | STF 只是显示，要用 HT 应用 |
| 降噪过重 | 「塑料感」、涂抹感 | 线性 + 非线性各轻度一次，配蒙版 |
| 饱和度对整图提 | 背景出现彩色噪声 | 用亮度蒙版只提目标 |
| SHO 星点不处理 | 洋红色光晕 | 去星后单独处理星点 |
| 黑点切得太狠 | 暗弱结构被裁掉，背景死黑 | 背景保留在 0.08–0.12 左右的暗灰 |
| 没有平场靠 DBE 除法补救 | 灰尘圈残留，暗角修不干净 | 拍平场 |
| 星系去星 | HII 区、星团被当成星去掉 | 星系慎用去星，或只用于星点缩小 |

---

## 13. 参考

- PixInsight Reference Documentation：<https://pixinsight.com/doc/>
- R. Berry, J. Burnell, *The Handbook of Astronomical Image Processing*, Willmann-Bell, 2005.
- W. Keller, *Inside PixInsight*, 2nd ed., Springer, 2018.
- R. H. Lupton et al., *Preparing Red-Green-Blue Images from CCD Data*, PASP 116 (2004) 133.（arcsinh 拉伸）
- J. J. Hester et al., *Hubble Space Telescope WFPC2 Imaging of M16: Photoevaporation and Emerging Young Stellar Objects*, AJ 111 (1996) 2349.（创生之柱）
- M. Cranfield, D. Payne, *Generalized Hyperbolic Stretch* 文档：<https://ghsastro.co.uk/>
- RC Astro（BlurXTerminator / NoiseXTerminator / StarXTerminator）：<https://www.rc-astro.com/>
- GraXpert：<https://www.graxpert.com/>
- StarNet：<https://www.starnetastro.com/>
- Light Vortex Astronomy PixInsight 教程（经典的分步教程合集）
- Adam Block Studios PixInsight 教程
