# planetary/ — 行星后期（幸运成像）

用 Python 实现行星摄影后期的完整流水线，对应业余行星摄影的「四件套」：

```
synth.py ──► prep.py ──► stack.py ──────────► wavelets.py ──► 成品
(采集/仿真)   (PIPP)     (AutoStakkert!4)  │    (RegiStax 6)
                                          └─► derotate.py ──┘
                                              (WinJUPOS 去自转)
```

只依赖 **numpy**（+ Pillow 读写 PNG/JPG；读 AVI 才需要 OpenCV）。在手机 Termux 上就能跑完整流程。

| 文档 | 内容 |
|---|---|
| [docs/software_comparison.md](docs/software_comparison.md) | PIPP / AS!4 / RegiStax 6 / WinJUPOS 的分工、逐步骤对比表、参数对照、典型工作流、各行星差异、本实现与原软件的差距 |
| [docs/principles.md](docs/principles.md) | 原理：大气湍流（r0、τ0、等晕角、幸运概率）、采样、SER 格式、debayer、质量评估、相位相关、多点对齐、drizzle、小波锐化、去振铃与大气色散、扁球体去自转几何、柱面投影、仿真验证 |

---

## 1. 脚本

| 脚本 | 对应软件 | 作用 |
|---|---|---|
| `planetools.py` | — | 公共库：SER 读写（memmap + 时间戳）、debayer、双线性 / 傅里叶亚像素平移、批量相位相关、质量评估、圆面检测、扁球体投影与自转、16-bit TIFF 读写 |
| `synth.py` | FireCapture（仿真） | 生成带真值的木星 SER：纹理在经纬度上定义（自转是真实的），加抖动、漂移、局部扭曲、随时间和位置变化的模糊、大气色散、噪声、拜耳马赛克 |
| `prep.py` | **PIPP** | debayer → 找行星 → 剔除（无行星 / 碰边 / 面积或亮度离群）→ 整数像素居中裁剪 → 质量排序 / 保留前 N → SER（保留时间戳）+ CSV 日志 |
| `stack.py` | **AutoStakkert!4** | 质量排序 → 相位相关全局对齐 → 参考帧 → AP 网格 → **每个 AP 局部选帧 + 局部位移** → 一次重采样 → Hann 窗拼接；`--upscale`、`--ap-map`、质量 CSV |
| `wavelets.py` | **RegiStax 6** | 高斯 / à trous 小波分层，dyadic / linear，每层增益 + 降噪，去振铃，RGB 对齐，色彩平衡，伽马，`--dump` 输出各层 |
| `derotate.py` | **WinJUPOS** | `limit` 自转时长极限、`measure` 圆面拟合 + 经纬网、`stack` 多图去自转叠加、`rgb` 滤镜轮 RGB 去自转合成、`video` 视频逐帧去自转、`map` 柱面投影图 |
| `demo.py` | — | 端到端演示：3 段相隔 5 分钟的仿真视频走完全流程，每一步与真值比较 |

所有脚本 `-h` 查看完整参数。

## 2. 快速开始

```bash
pip install numpy pillow            # 可选：opencv-python（读 AVI）、astropy（读写 FITS）

# 一键演示（约 2–3 分钟，输出到 planetary/demo_out/）
python planetary/demo.py

# 自己的数据
python planetary/prep.py capture.ser -o pipp.ser --crop 320
python planetary/stack.py pipp.ser -o 2026-10-13-1834_5-j.tif --keep 20% --ap-size 40 --ap-map ap.png
python planetary/wavelets.py 2026-10-13-1834_5-j.tif -o final.png --rgb-align --dering 0.6 --balance auto

# 多段视频去自转（文件名按 WinJUPOS 规则带 UTC 时间）
python planetary/derotate.py limit --planet jupiter --diameter-arcsec 45 --aperture-mm 254
python planetary/derotate.py measure stacks/*.tif --de 3 --grid check.png
python planetary/derotate.py stack stacks/*.tif -o derot.tif --de 3
```

更多工作流（滤镜轮 RGB、长视频去自转、土星 / 火星）见 [docs/software_comparison.md §3](docs/software_comparison.md#3-典型工作流)。

## 3. 演示结果（`demo.py`，有真值）

3 段仿真木星视频（18:30 / 18:35 / 18:40，每段 150 帧，RGGB 8-bit，赤道半径 70 px，视宁度 σ ≈ 1.8 px），
全部结果与 18:35 的清晰真值比较：

| 结果 | 等效模糊 px | 残差 % | 非带状细节误差 % | 与清晰真值误差 % |
|---|---|---|---|---|
| 最好的单帧 | 1.4 | 11.4 | 31.5 | 14.6 |
| 中等单帧 | 2.1 | 15.5 | 34.9 | 22.9 |
| 叠加，仅全局对齐 | 1.8 | 2.8 | 5.6 | 12.9 |
| 叠加，多点对齐（AS!4 方式） | 1.7 | 2.8 | 5.4 | 12.6 |
| 　+ 小波（RegiStax 方式） | 0.7 | 6.1 | 12.4 | **6.9** |
| 3 段直接平均（不去自转） | 1.8 | 3.2 | 7.9 | 13.3 |
| 　+ 小波 | 0.8 | 6.4 | 14.7 | 7.4 |
| 3 段去自转后平均（WinJUPOS 方式） | 1.7 | **2.5** | **3.6** | 12.6 |
| 　+ 小波 | 0.7 | 6.0 | 11.1 | **6.8** |

读法：
- **叠加**把噪声（残差）从 11% 降到 3%，但单帧最好的那张反而比叠加更锐（1.4 vs 1.7 px），这正是幸运成像的矛盾 —— 所以要局部选帧 + 锐化；
- **多点对齐**比全局对齐略好（此仿真的局部扭曲只有 0.9 px rms，口径越大、视宁度越差差距越大）；
- **小波**把等效模糊从 1.7 px 降到 0.7 px，与清晰真值的误差几乎减半；
- **不去自转**直接平均 3 段（相邻两段相差 5 分钟、3.0°，圆面中心移动 3.7 px），细节误差从 5.4% 恶化到 7.9%；
  **去自转**后降到 3.6%（三段的噪声也平均掉了）。
- 小波增益扫描：0.5/0.5/0.3 时与真值误差 6.9%，1/1/0.5 时 11.7%，3/1.5/0.6 时 22% —— **过锐比不锐更偏离真值**。

其他验证：
- 相位相关：已知平移 (3.30, −1.70) px 测得 (3.25, −1.69)；
- RGB 对齐：仿真色散 ±1.00 px，测得 R −0.95、B +0.96 px；
- 圆面测量：真值半径 70 px、极轴 25°，测得 70.35 px、24.5°；
- 去自转几何：无模糊图从 18:30 转到 18:40（6.04°）与真值相比，非带状细节误差 0.0070 → 0.0008。

## 4. 注意

- 所有帧放入内存（float32）。几千帧的大视频先用 `prep.py --keep 30%` 预筛，或 `stack.py --limit`。
- `stack.py` 输出的是**线性、未锐化**的叠加图（16-bit TIFF）。TIFF 读写用的是 `planetools.save_tiff/load_tiff`（无压缩基线 TIFF，
  AS!/RegiStax/WinJUPOS 都能读），不依赖 tifffile。
- 去自转需要正确的 UTC 时间：文件名 `YYYY-MM-DD-HHMM_T`（T = 0.1 分钟），或 `--times`，或 SER 帧时间戳。
- 本实现与原软件的差距（插值、drizzle、星历、土星环等）见 [docs/software_comparison.md §4](docs/software_comparison.md#4-本目录实现与原软件的差距)。
