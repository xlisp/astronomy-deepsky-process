# astronomy-deepsky-process

用 Python 代码完成深空天体摄影后期：从原始 FITS 子帧到 LRGB / SHO / HOO 彩色成品，
并附带一套可独立使用的深空后期工具脚本（`scripts/`）。

> PixInsight（PI）本质上是「图形界面 + 一组图像处理算法（Process）+ 脚本引擎（PJSR）」。
> 本仓库把这些算法拆开，用几十到一百多行 numpy 代码逐个实现，每一步都能看懂、能改、能批量自动跑。

- [1. 项目与数据](#1-项目与数据)
- [2. PixInsight vs 代码处理](#2-pixinsight-vs-代码处理)
- [3. 主流程 process/ 的实现细节](#3-主流程-process-的实现细节)
- [4. 工具脚本 scripts/](#4-工具脚本-scripts)
- [5. 核心算法原理速查](#5-核心算法原理速查)
- [6. 局限与拍摄建议](#6-局限与拍摄建议)
- [7. 银河 / 星野叠加 milkyway/](#7-银河--星野叠加-milkyway)
- [8. Wolfram Language 版本 wolfram/](#8-wolfram-language-版本-wolfram)

---

## 1. 项目与数据

目标：**M27 哑铃星云**（行星状星云，约 8′×5.7′）。

| 项目 | 参数 |
|---|---|
| 相机 | QHY183M（黑白，2.4 µm，5544×3694） |
| 焦距 | 1197 mm → 0.41″/px |
| 滤镜 | L R G B + SII Ha OIII，共 7 个 |
| 曝光 | 每滤镜 10 × 5 s，即每通道仅 50 s |
| 校准帧 | **无**暗场 / 平场 / 偏置场 |

文件名形如 `..._Bin1_<n>_..._0006.fit`，`n` 为滤镜编号：`1=L 2=R 3=G 4=B 5=S 6=H 7=O`。

### 运行

```bash
pip install numpy scipy opencv-python astropy sep astroalign   # process/ 需要
# 原始 *.fit 放在仓库根目录（已被 .gitignore 忽略）
python process/stack.py          # 配准 + 叠加 -> process/master_{L,R,G,B,S,H,O}.fits
python process/stack.py 5 6 7    # 只叠指定滤镜
python process/compose.py        # 去梯度 + 拉伸 + 合成 -> process/M27_*.jpg / .tif
```

### 输出（`process/` 下，每张都有 8-bit `.jpg` 和 16-bit `.tif`）

| 文件 | 说明 |
|---|---|
| `M27_HaLRGB` | ⭐ 效果最好：LRGB 基础上把 Ha 叠入红、OIII 叠入绿/蓝 |
| `M27_LRGB` | 真彩色：青蓝色主体、外围红色 Ha |
| `M27_HOO` | 双色窄带（R=Ha，G/B=OIII），偏青 |
| `M27_SHO` | 哈勃色板 |
| `M27_{L,S,H,O}_mono` | 各通道黑白图 |

---

## 2. PixInsight vs 代码处理

### 2.1 流程对照

| 步骤 | PixInsight 里的 Process | 本仓库实现 | 核心原理 |
|---|---|---|---|
| 查看文件 / 头信息 | FITSHeader、Statistics | `scripts/fits_info.py` | 读 FITS 头，按滤镜汇总曝光 |
| 采样 / 视场计算 | （外部工具） | `scripts/fov_calc.py` | 206.265·像元/焦距 |
| 主校准帧 | ImageIntegration | `scripts/calibrate.py master` | 逐像素 σ-clip 均值 |
| 校准 | ImageCalibration | `scripts/calibrate.py apply` | (light − dark)/flat，暗场缩放 |
| 热像素 | CosmeticCorrection | `stack.py: load()`、`calibrate.py --cosmetic` | 与 3×3 中值差 > kσ 则替换 |
| 子帧评分 | SubframeSelector | `scripts/subframe_selector.py` | 星点矩 → FWHM / 偏心率 / 权重 |
| 快速翻看 | Blink | `scripts/blink.py` | 统一 STF 后生成 GIF |
| 星点检测 / PSF | StarAlignment、DynamicPSF | `sep`（stack.py）、`astrotools.detect_stars` | 背景图 + 局部极大 + 二阶矩 |
| 配准 | StarAlignment | `stack.py`（astroalign + 引导匹配） | 三角形不变量 / KD 树 + RANSAC |
| 叠加 | ImageIntegration | `stack.py`、`astrotools.sigma_clip_mean` | σ-clip + FWHM 加权平均 |
| 去梯度 | ABE / DBE | `compose.py: remove_gradient`、`scripts/gradient_removal.py` | 网格采样 + 二维多项式拟合 |
| 去条纹 | CanonBandingReduction | `compose.py: deband`、`gradient_removal.py --deband` | 行/列中值剖面减去平滑趋势 |
| 通道匹配 | LinearFit | `scripts/pixelmath.py --linearfit` | 鲁棒线性回归 a·x+b |
| 通道合成 | ChannelCombination、PixelMath | `scripts/pixelmath.py` | numpy 表达式 |
| 色彩校准 | BackgroundNeutralization、ColorCalibration / SPCC | `compose.py: color_balance`、`scripts/color_calibration.py` | 背景减到中性 + 星点平均白 |
| 反卷积 | Deconvolution / BlurXTerminator | `scripts/deconvolve.py` | Richardson–Lucy + 高斯 PSF |
| 降噪 / 锐化 | MLT / MMT / NoiseXTerminator | `scripts/wavelet.py`、`compose.py: denoise` | starlet 小波软阈值 / 双边滤波 |
| 拉伸 | ScreenTransferFunction + HistogramTransformation | `compose.py: stretch`、`scripts/stretch.py` | MTF 中间调变换 |
| LRGB 合成 | LRGBCombination | `compose.py: lrgb_combine` | Lab 空间替换 L 通道 |
| 去绿 | SCNR | `compose.py: scnr` | G = min(G, (R+B)/2) |
| 缩星 | StarMask + MorphologicalTransformation | `scripts/star_reduce.py` | 星点遮罩内做最小值滤波 |
| 定位目标 | ImageSolver / 注释 | `compose.py`（astropy WCS） | 天球坐标 → 像素 |

### 2.2 各自优缺点

| | PixInsight | 代码（本仓库） |
|---|---|---|
| 上手 | 图形界面，实时预览，学习曲线陡 | 要会 Python，但每个参数含义都写在代码里 |
| 可重复性 | 靠 ProcessIcon / 项目文件 | 一条命令重跑全部，git 管理每次修改 |
| 批量 / 自动化 | WBPP 很强，其余要写 PJSR | 天然批量，服务器、手机（Termux）、CI 都能跑 |
| 透明度 | 闭源，算法细节要查文档 | 每一行可读可改，适合学习原理 |
| 高级算法 | SPCC（Gaia 光度校准）、BXT/NXT/SXT（AI）、Drizzle、局部归一化 | 需要自己实现，或调用 Siril、GraXpert、StarNet 等外部工具 |
| 交互调色 | 曲线、遮罩随手画 | 要反复调参数再看图 |

**结论**：前期的批处理部分（校准、评分、配准、叠加、去梯度）用代码更快、更可控；
后期的审美调整（曲线、局部遮罩、AI 降噪/去星）PI 更方便。两者可以混用，`process/master_*.fits`
是 32 位浮点线性数据，可以直接拖进 PI 继续做 SPCC、BlurXTerminator、NoiseXTerminator。

### 2.3 同一操作，两种写法

以「HOO 双色合成」为例：

PixInsight：`LinearFit`（参考 Ha）→ `PixelMath`
（R: `Ha`，G: `0.35*Ha+0.65*OIII`，B: `OIII`）→ `STF` → `HistogramTransformation`

代码：

```bash
python scripts/pixelmath.py -i H=process/master_H.fits -i O=process/master_O.fits --linearfit H \
       --r "H" --g "0.35*H+0.65*O" --b "O" -o hoo.fits --preview hoo.jpg
```

其中 STF 自动拉伸只有几行（`scripts/astrotools.py`）：

```python
def mtf(m, x):                                   # PI 的中间调传递函数
    return ((m - 1) * x) / ((2 * m - 1) * x - m)

def stf_params(ch, shadows=-2.8, target=0.25):   # PI AutoSTF 的默认参数
    med = np.median(ch); madn = 1.4826 * MAD(ch)
    c0 = clip(med + shadows * madn, 0, 1)        # 黑点：背景中值往左 2.8σ
    m = mtf(target, (med - c0) / (1 - c0))       # 让背景中值落到 0.25
    return c0, m
```

---

## 3. 主流程 process/ 的实现细节

### 3.1 `stack.py` — 校准、配准、叠加

```
原始帧 ──► 去热像素 ──► sep 星点检测 ──► 选基准帧（L 中 FWHM 最小）
       ──► 求变换（LRGB：三角匹配；窄带：引导匹配） ──► Lanczos4 重采样
       ──► 减背景 ──► σ-clip + 1/FWHM² 加权平均 ──► master_X.fits
```

1. **去热像素（`load`）**：没有暗场，只能做 cosmetic correction。用 3×3 中值滤波得到「应有的值」，
   与原值之差超过 8 倍 MAD-σ 的像素视为热像素并替换。
2. **星点检测（`stars`）**：`sep`（SExtractor 的 Python 版）先估计背景网格，在减背景后的图上找连通区域，
   去掉 flag 异常和接近饱和（peak ≥ 60000）的星，按流量取最亮的 N 颗。
3. **基准帧**：遍历所有 L 帧，取星点半长轴 `a` 中位数最小（星点最锐）的一帧。
4. **三角形匹配（`astroalign`）**：把星点三三组成三角形，用边长比（平移、旋转、缩放不变量）
   在两张图之间配对，再用 RANSAC 求相似变换。LRGB 每帧星点多（> 60 颗），一次成功。
5. **引导匹配（`guided_transform`）**：窄带 5 s 单帧只有 10–20 颗星，三角形不够，全部失败。
   改进做法：
   - 赤道仪漂移是连续的，用**上一帧的变换**作为初始预测 `T0`；
   - 把当前帧星点按 `T0` 投到基准帧，用 KD 树在 30 px 半径内找最近邻；
   - `cv2.estimateAffinePartial2D`（相似变换 + RANSAC）求新变换，再用 6 px、3 px 半径逐步收紧重复。
   - 基准帧额外用低阈值（3σ）提取 1500 颗星作为稠密参考，提高命中率。

   结果 70 帧全部配准，漂移约 170 px，旋转约 0.07°，轨迹连续平滑。
6. **重采样**：`cv2.warpAffine` + Lanczos4，边界填 NaN（叠加时自动忽略）。
7. **叠加**：每帧先减去自身背景中值（相当于简化版归一化），然后：
   - 逐像素算中值和 MAD-σ，偏离 > 3σ 的像素记为 NaN（卫星轨迹、宇宙线、残余热像素）；
   - 剩余像素按 `w = 1/FWHM²` 加权平均。星点越锐的帧权重越大。

### 3.2 `compose.py` — 去梯度、拉伸、合成

1. **公共有效区**：因为漂移，各 master 的边缘有 0 值，取所有通道都有数据的矩形再内缩 20 px。
2. **去条纹 `deband`**：没有偏置场时，CMOS 的行/列固定图案噪声会很明显。先剔除 > 3σ 的像素（星、星云），
   按行取中值得到剖面，减去 101 点滑动平均（保留真实的大尺度梯度），剩下的就是条纹；先行后列各做一次。
3. **去梯度 `remove_gradient`**（类 ABE）：
   - 分成 24×24 块，每块取中值作为背景样本；
   - 拟合二维 2 次多项式 `z = Σ c_ab · x^a · y^b`；
   - 比模型亮 2.5σ 以上的样本（星云、亮星）剔除后重拟合，迭代 5 次；
   - 原图减去模型。
4. **WCS 定位**：一开始按「最亮区域」裁剪，结果被 QHY183 右侧的放大器辉光带偏。改为读取基准帧 FITS 头里的
   plate-solve 结果（WCS），把 M27 的赤经/赤纬（299.9016°, 22.7212°）换算成像素坐标，裁 2000×2000。
5. **拉伸 `stretch`**：STF 的写法，黑点 = 背景中值 − 1.5σ，白点取 99.995% 分位，
   再用 MTF 把背景压到 0.06–0.12（比 PI 默认的 0.25 更暗、更有反差）。
6. **色彩平衡 `color_balance`**：各通道减去中值（背景中性化），再用最亮 0.1% 像素（排除饱和）的中值
   做白平衡，即「亮星平均为白色」。
7. **合成亮度**：`Lsyn = L/σ_L² + (R+G+B)/σ_RGB²`，按噪声方差倒数加权（最优信噪比组合），把 RGB 的光子也用上。
8. **LRGB 合成 `lrgb_combine`**：RGB 转 Lab，用拉伸后的亮度替换 L 通道，a/b 乘以 1.3–1.6 提饱和度，再转回 RGB。
   色彩通道先做 σ=1.5 的高斯模糊（彩色噪声主要在高频，人眼对色度分辨率不敏感）。
9. **降噪 `denoise`**：双边滤波，并用亮度做遮罩：暗处多平滑，亮处保留细节。
10. **SCNR 去绿**：`G = min(G, (R+B)/2)`（Average Neutral），天体图中几乎没有真正绿色的天体。
11. **色板**：
    - SHO：R = 0.55·S + 0.45·H，G = 0.6·H + 0.4·O，B = O，再部分 SCNR；
    - HOO：R = H，G = 0.35·H + 0.65·O，B = O；
    - HaLRGB：`R = max(R, Ha)`，`G = max(G, 0.5·OIII)`，`B = max(B, OIII)`，增强发射线结构。
    - 窄带亮度 `Lnb` 同样按方差倒数加权合成三个窄带。

---

## 4. 工具脚本 scripts/

`scripts/astrotools.py` 是公共库，**核心算法只依赖 numpy**（手机 Termux 上也能跑）。
FITS 读写才需要 `astropy`，16-bit 彩色 TIFF 输出优先用 `tifffile` / `cv2`。
所有脚本都支持 `-h` 查看完整用法，输入可以是 FITS / TIFF / PNG / JPG。

```bash
pip install numpy pillow astropy matplotlib   # 可选：tifffile
```

| 脚本 | 对应 PI | 作用 |
|---|---|---|
| `fits_info.py` | FITSHeader / Statistics | 列出滤镜、曝光、增益、温度，按滤镜统计总积分时间；`--stats` 附带 FWHM |
| `fov_calc.py` | — | 像元比例尺、视场、采样是否合适（Nyquist）、Dawes 极限 |
| `calibrate.py` | ImageIntegration + ImageCalibration | 生成 master bias/dark/flat，校准亮场，暗场缩放，热像素 |
| `subframe_selector.py` | SubframeSelector | FWHM / 偏心率 / 星数 / 噪声 评分，离群帧剔除，输出 CSV 和合格列表 |
| `blink.py` | Blink | 子帧动画 GIF，检查云、卫星、跟踪、结露 |
| `gradient_removal.py` | ABE / DBE | 多项式背景模型，减法（光污染）或除法（暗角），可选去条纹 |
| `color_calibration.py` | BackgroundNeutralization + ColorCalibration | 背景中性化 + 星点白平衡（线性数据） |
| `pixelmath.py` | PixelMath + ChannelCombination + LinearFit | 任意表达式合成通道/色板 |
| `stretch.py` | STF + HistogramTransformation | AutoSTF / arcsinh / 手动 MTF，可输出直方图 |
| `wavelet.py` | MultiscaleLinearTransform | starlet 小波分层降噪、锐化、导出各层 |
| `deconvolve.py` | Deconvolution / DynamicPSF | Richardson–Lucy 反卷积，PSF 自动测量，亮度遮罩 + 去振铃 |
| `star_reduce.py` | StarMask + MorphologicalTransformation | 缩星、导出星点遮罩 |

### 4.1 一套完整流程示例

```bash
S=scripts
# 0. 盘点数据、算采样
python $S/fits_info.py "raw/*.fit"
python $S/fov_calc.py --focal 1197 --pixel 2.4 --sensor 5544x3694 --aperture 200 --seeing 3.7 --target 8x5.7

# 1. 校准
python $S/calibrate.py master bias -o cal/bias.fits raw/bias/*.fit
python $S/calibrate.py master dark -o cal/dark.fits raw/dark/*.fit
python $S/calibrate.py master flat -o cal/flat_L.fits --bias cal/bias.fits raw/flat/L_*.fit
python $S/calibrate.py apply -o calibrated/ --bias cal/bias.fits --dark cal/dark.fits \
       --flat cal/flat_L.fits --cosmetic raw/light/*_1_*.fit

# 2. 挑帧
python $S/blink.py "calibrated/*.fits" -o blink.gif
python $S/subframe_selector.py calibrated/*.fits --kappa 2.5 --accepted good.txt --csv grades.csv

# 3. 配准叠加（本仓库主流程）
python process/stack.py

# 4. 线性阶段：去梯度 → 色彩校准 → 反卷积 → 降噪
for c in L R G B H O; do python $S/gradient_removal.py process/master_$c.fits -o lin/$c.fits --deband; done
python $S/color_calibration.py -r lin/R.fits -g lin/G.fits -b lin/B.fits -o lin/RGB.fits
python $S/deconvolve.py lin/L.fits -o lin/L_dc.fits --iters 20
python $S/wavelet.py lin/L_dc.fits -o lin/L_nr.fits --denoise 3,2,1,0.5 --amount 0.8 --mask

# 5. 非线性阶段：拉伸 → 合成 → 锐化 → 缩星
python $S/stretch.py lin/RGB.fits -o out/RGB.tif
python $S/pixelmath.py -i H=lin/H.fits -i O=lin/O.fits --linearfit H \
       --r H --g "0.35*H+0.65*O" --b O -o out/HOO.tif --preview out/HOO.jpg
python $S/wavelet.py out/HOO.tif -o out/HOO_sharp.tif --bias 0,0.3,0.2 --mask
python $S/star_reduce.py out/HOO_sharp.tif -o out/HOO_final.tif --size 3 --amount 0.6
```

### 4.2 常用单项示例

```bash
# 采样检查：本数据 FWHM≈3.7″ 在 0.41″/px 下约 9 px，明显过采样 → 建议 bin2/bin3
python scripts/fov_calc.py --focal 1197 --pixel 2.4 --seeing 3.7
#   bin1: 0.41"/px  seeing 3.7" = 8.9 px FWHM (over-sampled)
#   bin3: 1.24"/px  seeing 3.7" = 3.0 px FWHM (good)

# 看小波各层（判断噪声在哪几层、星云结构在哪几层）
python scripts/wavelet.py process/master_H.fits --layers 6 --dump layers/

# 暗角用除法，光污染用减法；导出背景模型检查是否「吃掉」星云
python scripts/gradient_removal.py master_L.fits --degree 3 --model bg_model.png

# 合成亮度（表达式里可以用 mad / median / where / clip 等）
python scripts/pixelmath.py -i L=L.fits -i R=R.fits -i G=G.fits -i B=B.fits \
       --expr "L/mad(L)**2 + (R+G+B)/mad(R+G+B)**2" --rescale -o Lsyn.fits

# 星点遮罩，拿去 PS / GIMP 用
python scripts/star_reduce.py M27_LRGB.tif --mask-only -o starmask.png
```

---

## 5. 核心算法原理速查

**信噪比与叠加**：N 帧平均后噪声降为 1/√N，SNR ∝ √(总曝光时间)。50 s → 30 min，SNR 提升约 6 倍。

**鲁棒统计**：天文图像中星点是「离群值」，所以到处用中值和 MAD 而不是均值和标准差。
`σ ≈ 1.4826 · median(|x − median(x)|)`（对高斯分布无偏）。`noise_sigma` 进一步用「像素 − 4 邻域均值」
消去大尺度信号，只留下像素噪声。

**σ-clip 叠加**：逐像素在 N 帧中求中值 m 和 σ，剔除 `|x − m| > kσ` 的值后求（加权）平均。
卫星轨迹只出现在 1 帧，自然被剔除。

**星点测量**（`detect_stars`）：背景网格（每块 σ-clip 中值 + 双线性插值）→ 减背景 → 高斯平滑 →
5×5 局部极大且 > kσ → 在 r=7 的窗口里计算一阶矩（质心）和二阶矩矩阵
`[[Mxx, Mxy], [Mxy, Myy]]`，其特征值 λ1 ≥ λ2：
`FWHM = 2.3548·√((λ1+λ2)/2)`，`偏心率 = √(1 − λ2/λ1)`；半通量半径 HFR 对高斯星点满足 `FWHM = 2·HFR`。

**中间调传递函数 MTF**：`MTF(m, x) = (m−1)x / ((2m−1)x − m)`，满足 MTF(m,0)=0、MTF(m,1)=1、MTF(m,m)=0.5。
它有个巧妙的性质 `MTF(MTF(t, b), b) = t`，所以「把背景 b 映射到目标 t」所需的中间调就是 `m = MTF(t, b)`。

**arcsinh 拉伸**：`y = asinh(β·x)/asinh(β)`，暗部近似线性、亮部近似对数；作用在亮度上并按比例缩放 RGB，
可以保持星点颜色不被拉白。

**starlet 小波**（à trous）：用 B3 样条核 `[1,4,6,4,1]/16` 反复平滑，第 j 次在核的抽头之间插 2^j−1 个零（"带孔"）。
细节层 `w_j = c_{j−1} − c_j`，`图像 = Σ w_j + 残差`（精确重建）。白噪声在各层的 σ 是固定比例
（0.889、0.200、0.086、0.041…），所以每层按 `k_j · σ · 比例` 软阈值即可降噪；乘以 (1+bias) 则是锐化。

**Richardson–Lucy 反卷积**：观测 d = PSF ⊗ u + 噪声，迭代
`u ← u · PSFᵀ ⊗ (d / (PSF ⊗ u))`。高斯 PSF 在频域就是高斯
`exp(−2π²σ²(fx²+fy²))`，用 FFT 卷积。迭代多了会放大噪声、在亮星周围产生暗环，所以用亮度遮罩
（只在高信噪比区域生效）和去振铃遮罩（保护亮星）混合。

**多项式背景**：最小二乘解 `c = argmin ‖A·c − v‖²`，A 的每列是 `x^a·y^b`。只要采样能避开目标，
低阶多项式表达不了星云这种小尺度结构，所以不会把星云当背景减掉。目标充满画面时要降低阶数或改用 DBE 手动选点。

**LinearFit**：窄带各通道强度可能差 10 倍。用鲁棒线性回归 `ref ≈ a·img + b`（迭代剔除 3σ 残差，主要是星点）
把每个通道映射到参考通道的尺度，然后再按色板混合，颜色才可控。

**Lab LRGB 合成**：Lab 把亮度 L 和色度 a/b 分开。用高信噪比的 L 替换亮度，色度可以大幅模糊降噪而不损失细节。

---

## 6. 局限与拍摄建议

- **窄带信号太弱**：SHO 每通道只有 50 s。建议窄带单张 120–300 s，每通道至少 30 分钟。
- **星点偏软**：FWHM ≈ 9 px ≈ 3.7″，当晚视宁度一般；0.41″/px 明显过采样，bin2 / bin3 拍摄更划算（见 `fov_calc.py`）。
- **补拍校准帧**：没有暗场，右侧放大器辉光和行条纹只能靠裁剪和算法压制；没有平场，暗角和灰尘圈只能靠多项式近似。
  有了校准帧后用 `scripts/calibrate.py` 预处理，再交给 `process/stack.py`。
- **想在 PI 中精修**：直接打开 `process/master_*.fits`（32 位浮点线性数据），做 SPCC 色彩校准、
  BlurXTerminator、NoiseXTerminator。
- `scripts/` 中的反卷积、小波、色彩校准都是教学级的简化实现，效果不及 PI 的 SPCC 和 AI 工具，
  但原理一致，便于理解和批处理。

---

## 7. 银河 / 星野叠加 milkyway/

固定三脚架 + 广角镜头拍摄的银河照片，星空在旋转、地景不动、镜头畸变大，所以单独做了一套流程
（对标 Sequator / Starry Landscape Stacker）：天空按星点用 poly2 变换对齐后 σ-clip 叠加，
地景不对齐叠加，再用（自动识别或手绘的）天空遮罩羽化融合。

```bash
python milkyway/exposure_calc.py --focal 14 --fnum 2.8 --pixel 5.9     # NPF 法则算最长曝光
python milkyway/stack_milkyway.py "shots/*.ARW" -o out/ --darks darks/*.ARW
```

详见 [milkyway/README.md](milkyway/README.md)。

---

## 8. Wolfram Language 版本 wolfram/

用 Mathematica / wolframscript 把 `process/` 的流程又实现了一遍（`AstroTools.wl` + `stack.wls` + `compose.wls` + `tools.wls`），
用来对比三种做法。**这部分代码尚未在真实 Wolfram 内核上运行过**，详见 [wolfram/README.md](wolfram/README.md)。

```bash
wolframscript -file wolfram/stack.wls && wolframscript -file wolfram/compose.wls
wolframscript -file wolfram/compose.wls python      # 用 Python 叠出的 master 合成，对比两边结果
```

| 步骤 | PixInsight | Python | Wolfram |
|---|---|---|---|
| 读 FITS | 打开文件 | `astropy.io.fits` | 自写解析器 `ReadFITS`（`BinaryReadList`，大端） |
| 星点检测 | StarDetector | `sep.extract` | `MorphologicalComponents` + `ComponentMeasurements` |
| 配准 | StarAlignment | astroalign 三角形 + 引导匹配 | 相位相关 `Fourier` + `Nearest` + `FindGeometricTransform` |
| 重采样 | Lanczos-3 | `cv2.warpAffine` | `ImagePerspectiveTransformation` |
| 叠加 | ImageIntegration | numpy σ-clip | `Median` 作用于 3 维数组 + 掩膜加权 |
| 去梯度 | ABE / DBE | `np.linalg.lstsq` | `LeastSquares` |
| 拉伸 | STF / HT | MTF 公式 | 同一 MTF 公式，可用 `Manipulate` 实时调参 |
| LRGB | LRGBCombination | `cv2.cvtColor` Lab | `ColorConvert[..., "LAB"]` |
| 反卷积 | Deconvolution / BXT | 手写 RL | `ImageDeconvolve[..., Method -> "RichardsonLucy"]` |
| 小波 | MLT | 手写 starlet | 手写 starlet，或 `StationaryWaveletTransform` |
| 目标坐标 | ImageSolver / 注释 | `astropy.wcs` | 自写 TAN 投影 + `NebulaData` 内置天体数据 |

一句话总结：**PI 适合出片，Python 适合批处理和自动化，Wolfram 适合研究算法和交互探索**。
Wolfram 的代码量最少，内置图像处理函数最全，但需要授权，天文专用生态也比较薄弱。

> 延伸阅读（[docs/](docs/README.md)）：
> - [数组计算史：从 APL、MATLAB、Mathematica 到 NumPy，函数名的出处](docs/array_computing_history.md)
> - [叠加与反卷积的来历：为什么平均能降噪，反卷积最早用在哪里](docs/stacking_deconvolution_history.md)（附可运行的 numpy 实验）
> - [MATLAB vs NumPy vs Wolfram 图像处理详细对比](docs/matlab_numpy_wolfram_image_processing.md)
> - [NumPy 与 Wolfram 语言基础对比](docs/numpy_vs_wolfram_image_processing.md)
