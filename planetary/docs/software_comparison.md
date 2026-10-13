# PIPP / AutoStakkert!4 / RegiStax 6 / WinJUPOS 流程对比

行星后期的事实标准是一条「四件套」流水线：

```
采集 (FireCapture / SharpCap)          .ser 视频 + 时间戳
  │
  ▼
PIPP ─────────── 预处理：debayer、找行星、剔除坏帧、居中裁剪、排序、格式转换
  │
  ▼
AutoStakkert!4 ─ 叠加：质量排序、全局对齐、多点对齐(AP)、每个 AP 局部选帧、(drizzle)
  │                              ┌────────────────────────────┐
  ├──────────────────────────────► WinJUPOS：多段叠加图去自转  │（可在锐化前或后）
  ▼                              └────────────────────────────┘
RegiStax 6 ───── 小波锐化、去噪、RGB 对齐、色彩平衡、伽马
  │
  ▼
WinJUPOS ─────── 去自转（图像 / RGB / 视频）、测量、柱面投影图
  │
  ▼
Photoshop / GIMP 等：最终调色、裁剪、标注
```

本目录的脚本一一对应：`prep.py`（PIPP）→ `stack.py`（AS!4）→ `wavelets.py`（RegiStax 6）→ `derotate.py`（WinJUPOS）。
原理见 [principles.md](principles.md)。

---

## 1. 四个软件各自解决什么问题

| | PIPP | AutoStakkert!4 | RegiStax 6 | WinJUPOS |
|---|---|---|---|---|
| 作者 | Chris Garry | Emil Kraaikamp | Cor Berrevoets | Grischa Hahn |
| 定位 | 预处理 / 格式转换 | **叠加**（行星、月面、太阳） | 叠加 + **小波锐化** | **去自转**、测量、制图 |
| 输入 | AVI / SER / MOV / 图像序列 / FITS | AVI / SER / 图像序列 | AVI / 图像 | 叠加后的图像（TIFF/PNG…）、SER |
| 输出 | 居中裁剪后的 SER / AVI / TIFF 序列 | 16-bit TIFF 叠加图（可带锐化预览） | 锐化后的 TIFF/PNG/JPG | 去自转图、RGB 合成、柱面图、测量数据 |
| 速度 | 快 | 很快（多线程，优化好） | 叠加慢；小波实时 | 快 |
| 是否必需 | 可选（AS!4 自己也能对付漂移的视频） | 现在几乎必需 | 现在主要只用小波 | 拍木星/土星多段或滤镜轮时必需 |
| 平台 | Windows | Windows | Windows | Windows（Wine 可跑） |

**历史上的分工变化**：2000 年代 RegiStax 一个软件包办对齐、叠加、锐化（RegiStax 4/5 引入多点对齐）；
2010 年前后 AutoStakkert! 出现，叠加速度和质量（尤其是「每个 AP 局部选帧」）明显更好，于是变成
**AS! 叠加 + RegiStax 只用小波**；PIPP 补上视频预处理（尤其是彩色 SER、漂移大的视频、日食月食序列）；
WinJUPOS 本来是 JUPOS 项目测量木星云带漂移用的，因为「去自转」这个功能而成为业余行星摄影必备。

## 2. 逐步骤对比

### 2.1 读入与预处理

| 步骤 | PIPP | AS!4 | RegiStax 6 | 本目录 |
|---|---|---|---|---|
| 读 SER（含 12/16 bit、时间戳） | ✔ | ✔ | 部分（老版本只认 AVI/图像） | `planetools.SerReader`（memmap） |
| Debayer | ✔（多种算法，可强制拜耳模式） | ✔（或 Bayer drizzle 不插值） | ✔ | `debayer` 双线性 |
| 找行星 | Object detection：阈值 + 重心 | Planet (COG) 模式 | 无（手动选对齐框） | `find_disk`：阈值 + 重心 + 去卫星 |
| 剔除坏帧 | 部分出画、过暗、过亮、太小 | 质量排序后丢弃 | 质量排序 | `prep.py`：无行星 / 碰边 / 面积或亮度离群 |
| 居中裁剪 | ✔ 整数像素 | 不裁剪（输出可裁） | 不裁剪 | `prep.py --crop`，整数像素，不插值 |
| 质量排序 / 只保留前 N | ✔（Quality estimation） | ✔ | ✔ | `prep.py --keep 50% --sort` |
| 输出格式 | SER/AVI/TIFF/PNG/FITS，保留时间戳 | — | — | SER（保留时间戳）或 WinJUPOS 命名的图像序列 |

**PIPP 的价值**在于：
1. 视频体积变小（裁掉 80% 的黑天空），后续 AS! 快很多；
2. 漂移大、行星跑出画面的帧被剔除，否则 AS! 会对齐失败；
3. 彩色 SER debayer 后转成 RGB SER，避免某些老软件不认拜耳；
4. 日面、月食、凌日序列的批量居中 / 对齐；
5. 只裁剪不插值，所以不损失任何细节。

### 2.2 叠加

| 步骤 | AS!4 | RegiStax 6 | 本目录 `stack.py` |
|---|---|---|---|
| 全局质量 | 梯度/局部对比类指标，Noise Robust 参数控制预平滑 | 「Quality」估计（梯度类） | `frame_quality`：平滑后梯度能量 / 亮度² |
| 全局对齐 | Planet (COG) 或 Surface（相关） | 单个或多个对齐框，相关法 | 相位相关（p = 0.5），两遍（先对最好帧，再对参考帧） |
| 参考帧 | 最好的若干帧平均（可「Double stack reference」） | 单帧或若干帧 | 最好的 `--ref-frames` 帧（默认保留帧数的一半）对齐后平均 |
| 对齐点 | 自动铺网格（AP size / Min Bright），可手动加减 | 自动/手动对齐点（RegiStax 5/6 多点模式） | 50% 重叠网格，`--ap-size`、`--min-bright` |
| **局部选帧** | ✔ 每个 AP 独立按局部质量取前 k% | ✗ 全局选帧 | ✔ 积分图求每个 AP 的局部梯度能量 |
| 局部位移 | ✔ | ✔ | AP 小块批量相位相关，超出 `--max-shift` 置零 |
| 拼接 | 平滑权重混合 | 平滑混合 | 周期 Hann 窗（单位分解）+ 无 AP 处用全局叠加补 |
| 叠加方式 | 平均（可按质量加权） | 平均 / 中值 / 加权 | 平均 |
| 超分辨率 | Drizzle 1.5×/3×、Resampled | Drizzle | `--upscale`（Resampled 方式） |
| 输出 | 16-bit TIF，可附带「锐化预览」 | 进入小波页 | 线性 16-bit TIF / npy，附带质量 CSV、AP 分布图 |

**最关键的区别：局部选帧**。设一帧里左半个圆面清楚、右半个模糊。RegiStax 只能整帧取舍；AS! 对左边的 AP 选这帧、
右边的 AP 不选。帧数越多、口径越大（等晕角相对越小）优势越明显。

**AS!4 参数和 `stack.py` 参数对照**

| AS!4 界面 | 含义 | `stack.py` |
|---|---|---|
| Frames to stack: 25% / 500 | 每个 AP 叠多少帧 | `--keep 25%` / `--keep 500` |
| AP Size 50 / 75 / 100 … | 对齐点框大小（px） | `--ap-size 32` |
| Min Bright | AP 放置的最低亮度 | `--min-bright 0.15`（相对最大值） |
| Noise Robust | 质量估计前的平滑强度 | `--sigma 1.0`（px） |
| Planet (COG) / Surface | 全局对齐方式 | 前者交给 `prep.py`，后者为 `stack.py` 默认 |
| Drizzle 1.5x / 3.0x | 超分辨率 | `--upscale 1.5`（Resampled 方式，非真 drizzle） |
| Normalize stack | 输出亮度归一化 | `--normalize` |
| Sharpened 输出 | 自带简单锐化 | 不做，交给 `wavelets.py` |

### 2.3 锐化与色彩

| 步骤 | RegiStax 6 | AS!4 | WinJUPOS | 本目录 `wavelets.py` |
|---|---|---|---|---|
| 小波分层 | 6 层；Default（à trous）/ Gaussian；Dyadic / Linear；Initial layer、Step | 只有简单锐化输出 | 有「Wavelet」和「Image processing」简单锐化 | `--filter atrous/gaussian`、`--scheme dyadic/linear`、`--initial`、`--step` |
| 每层增益 | Sharpen 滑块 | — | — | `--sharpen g1,g2,…`（层 × (1+g)） |
| 每层降噪 | Denoise 滑块 | — | — | `--denoise d1,d2,…`（软阈值） |
| 去振铃 | Deringing | — | — | `--dering`（边缘带内夹到局部极值范围） |
| RGB 对齐 | RGB Align（自动估计 R/B 偏移） | RGB Align（叠加时） | ✔（RGB 合成时） | `--rgb-align`（相位相关 + 傅里叶平移） |
| 色彩平衡 | RGB Balance（Auto） | — | ✔ | `--balance auto` / `r,g,b` |
| 伽马 / 色阶 | Gamma、Histogram | — | ✔ | `--gamma`、`--stretch` |
| 查看各层 | 实时预览 | — | — | `--dump layers/` 输出每层 PNG |

**注意数值不可直接照搬**：RegiStax 的滑块数值和层的定义（以及 6 层的尺度）与这里的增益不是同一单位。
本目录的增益是「层 × (1 + g)」，仿真里最优约 0.5 / 0.5 / 0.3（见 README demo 表）；习惯 RegiStax 的人通常会把前两层拉得很高，
在这里会明显过锐。

### 2.4 去自转

| 功能 | WinJUPOS 菜单 | `derotate.py` |
|---|---|---|
| 圆面测量（outline frame） | Recording → Image measurement：手动/自动拟合边缘框，设置北极方向 | `measure`：自动阈值 + 面积求半径 + 二阶矩求极轴，`--grid` 输出经纬网叠加图检查 |
| 星历（CM 经度、De、P） | 内置精确星历 | 需要自己提供 `--de`、`--cm0`（只影响绝对经度，不影响去自转本身） |
| 多张叠加图去自转 | Tools → De-rotation of images | `stack`（μ 加权，同晚共用半径和极轴，相位相关微调中心） |
| R/G/B(/L) 合成 | Tools → De-rotation of R/G/B frames | `rgb`（默认以 G 或 L 的时间为准） |
| 视频去自转 | Tools → De-rotation of video streams | `video`（每帧按时间戳去自转到中间时刻，输出 SER 交给叠加） |
| 柱面图 | Analysis → Map computation | `map` |
| 测量特征经纬度 / 漂移 | JUPOS 的本职工作 | 未实现 |
| 自转系统 | 木星 I/II/III、土星、火星… | `--planet`、`--system` |
| 时间来源 | 文件名 `YYYY-MM-DD-HHMM_T` 或手动输入 | 同样的文件名规则，或 `--times`、SER 时间戳 |

## 3. 典型工作流

### 3.1 彩色相机、一段视频（最常见）

```bash
python planetary/prep.py capture.ser -o pipp.ser --crop 320            # PIPP
python planetary/stack.py pipp.ser -o 2026-10-13-1834_5-j.tif --keep 20% --ap-size 40   # AS!4
python planetary/wavelets.py 2026-10-13-1834_5-j.tif -o final.png --rgb-align --dering 0.6 --balance auto
```

### 3.2 彩色相机、多段视频（推荐：每段 ≤ 自转极限，然后 WinJUPOS 叠加）

```bash
python planetary/derotate.py limit --planet jupiter --diameter-arcsec 45 --aperture-mm 254   # 每段最长约 2 分钟
for f in 2026-10-13-18*.ser; do
  python planetary/prep.py $f -o pipp/$(basename $f) --crop 320
  python planetary/stack.py pipp/$(basename $f) -o stacks/$(basename ${f%.ser}).tif --keep 20%
done
python planetary/derotate.py measure stacks/*.tif --de 3 --grid check.png    # 确认极轴方向、边缘拟合
python planetary/derotate.py stack stacks/*.tif -o derot.tif --de 3
python planetary/wavelets.py derot.tif -o final.png --rgb-align --dering 0.6
```

N 段视频去自转叠加，噪声再降 √N，所以最终可以比单段锐化得更狠。
**锐化在去自转之前还是之后？** 两种都有人用。常见做法是对未锐化或轻度锐化的叠加图去自转，最后统一锐化，
这样各张图的锐化伪影不会叠在一起，噪声也是先平均再放大。`demo.py` 就是这样做的。

### 3.3 黑白相机 + 滤镜轮（R、G、B 或 L-RGB / IR-RGB）

```bash
# 每个滤镜一段视频：R 18:30、G 18:32、B 18:34，分别 prep + stack，文件名带时间
python planetary/derotate.py rgb --r 2026-10-13-1830_0-R.tif --g 2026-10-13-1832_0-G.tif \
       --b 2026-10-13-1834_0-B.tif [--l 2026-10-13-1836_0-IR.tif] -o rgb.tif --de 3
python planetary/wavelets.py rgb.tif -o final.png --balance auto
```

木星 4 分钟内 R 和 B 之间相差 2.4°，不去自转合成时大红斑边缘会出现红蓝错边（这不是大气色散，RGB 对齐也修不好）。

### 3.4 长视频去自转后再叠加（WinJUPOS 视频去自转）

```bash
python planetary/derotate.py video long_5min.ser -o long_derot.ser --de 3
python planetary/stack.py long_derot.ser -o stack.tif --keep 10%
```

优点是一次叠加就能用 5 分钟的帧；缺点是每帧都要测中心，视宁度差时单帧的圆面拟合会抖（这里的实现里由后续叠加的对齐来吸收）。

### 3.5 不同行星的差异

| | 木星 | 土星 | 火星 | 月面 / 太阳 |
|---|---|---|---|---|
| 自转极限（254 mm） | ~2 分钟 | ~6 分钟 | ~13 分钟 | 不需要 |
| 去自转 | 几乎必需 | 多段时有用（环不自转，去自转只对本体正确） | 很少需要 | — |
| 极轴自动测 | 可以（f = 0.065） | 可以，但环会干扰，建议手动 `--north` | 不行（f = 0.006），默认北在上 | — |
| AP | 圆面内均匀铺 | 环和本体都要有 AP | 小圆面，AP 小 | 全画面铺满（Surface 对齐） |
| De | ±4° | 最大 ±27°，必须给对 | ±25° | — |

## 4. 本目录实现与原软件的差距

诚实列出，避免误用：

| 方面 | 原软件 | 这里 |
|---|---|---|
| 速度 | C++ / 多线程，几千帧几秒到几十秒 | numpy，全部帧放内存，几百帧 / 200×200 在手机上十几秒；大视频请先 `prep.py --keep` |
| 叠加插值 | 优化过的插值 + 真 drizzle | 双线性（约 0.3 px 的额外模糊）；`--upscale` 只是重采样 |
| 圆面拟合 | WinJUPOS 手动微调边缘框，最准 | 自动阈值，叠加图上半径误差约 0.5%，极轴约 0.5–1°（同晚取中位数） |
| 星历 | 内置 | 需手动给 De、CM |
| 土星环 | WinJUPOS 有专门处理 | 环区域按天空平移，不旋转（物理上环本来就不需要旋转，但边界处可能不连续） |
| 去马赛克 | 多种算法 | 双线性 |
| 交互 | 实时预览 | 命令行；`--dump`、`--ap-map`、`--grid` 输出图像来检查 |

好处是：每一步都是几十行 numpy，能看懂、能改、能批量跑、能在没有 Windows 的地方跑（树莓派 / 手机 / 服务器），
并且可以用 `synth.py` 的真值做定量验证。

## 5. 常见问题对照

| 现象 | 原因 | 对应处理 |
|---|---|---|
| AS! 叠加后有「重影」/ 方块 | AP 太小或太暗处放了 AP，局部位移估错 | 增大 `--ap-size`、提高 `--min-bright`，或 `--max-shift` 更严格 |
| 边缘一圈黑环 | 小波过锐（Gibbs 振铃） | 降低第 1–2 层增益，`--dering 0.5–1` |
| 上缘蓝下缘红 | 大气色散 | `--rgb-align`；拍摄时用 ADC |
| 去自转后更糊 | 方向反了（镜像）、极轴角错、时间错 | `measure --grid` 检查经纬网；`--mirror`；检查文件名时间是否为 UTC |
| 去自转后边缘有亮度台阶 | 临边昏暗 + 权重突变 | 增大 `--mu-power` 让边缘更多取自看得正的图 |
| 叠加结果比单帧还糊 | 保留帧太多 | `--keep` 降到 10–20% |
| 颗粒感、彩色噪点 | 帧数不足就强力锐化 | 多拍、多段去自转叠加，或加 `--denoise` |
