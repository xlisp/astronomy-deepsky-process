# 从 APL、MATLAB、Mathematica 到 NumPy：数组计算与图像处理函数的来历

[《NumPy 与 Wolfram 图像处理对比》](numpy_vs_wolfram_image_processing.md)的历史篇。
回答两个问题：**几个数学软件是怎么一路演化过来的**，以及 **NumPy（和 Wolfram）里那些函数名、约定最早出自哪里**。
最后追溯本仓库用到的天文图像算法的原始论文。

- [1. 总谱系](#1-总谱系)
- [2. 底层：Fortran 与数值库](#2-底层fortran-与数值库)
- [3. 各数学软件简史](#3-各数学软件简史)
- [4. Python 的数组之路](#4-python-的数组之路)
- [5. NumPy 函数溯源表](#5-numpy-函数溯源表)
- [6. 设计约定的来历](#6-设计约定的来历)
- [7. 图像处理函数溯源](#7-图像处理函数溯源)
- [8. 本仓库天文算法的出处](#8-本仓库天文算法的出处)
- [9. 参考](#9-参考)

> 说明：年份以首次公开发布或首篇论文为准。不同资料对个别年份的说法可能相差一两年，这类情况标注为「约」。

---

## 1. 总谱系

```
 1957 Fortran ──────────────────────────────────────────────────────────────┐
   │  (数值库: EISPACK 1974, LINPACK 1979, BLAS 1979, FFTPACK 1982, LAPACK 1992)│
   │                                                                          │
 1962 APL (Iverson) ─────────┬──────────────┬───────────────┐                 │
   │                         │              │               │                 │
   │                    1990 J, 1993 K   Lisp(1958)       S (1976, Bell Labs) │
   │                                        │               │                 │
 1977 IDL (LASP/Research Systems) ◄─────────┼───────────────┼─── Fortran 风格 ─┤
   │   天文界主力，IDL Astronomy Library     │               │                 │
   │                                        │          1993 R                 │
 ~1979 MATLAB (Moler, LINPACK/EISPACK 前端) ◄┼───────────────────────────────── ┤
   │   1984 MathWorks 商业化                 │                                 │
   │      ├── 1993 Octave (开源克隆)          │                                 │
   │      ├── 1994 Scilab (INRIA)            │                                 │
   │      └── 1993 Image Processing Toolbox  │                                 │
   │                                        │                                 │
 1968 Macsyma ─ 1981 SMP ─ 1988 Mathematica ◄┘ (Lisp 的函数式 + APL 的数组运算) │
   │                       1982 Maple                                         │
   │                                                                          │
 1991 Python ─ 1995 Numeric ─┬─ 2001 Numarray (STScI, 哈勃 / FITS) ◄─ IDL 用户 │
                             │                                                │
                       2006 NumPy ◄──────── LAPACK / FFTPACK (f2py 包装) ◄─────┘
                             │
         SciPy 2001 · matplotlib 2003 · scikit-image 2009 · Astropy 2013 · Julia 2012
```

一句话概括：**APL 给出了思想，Fortran 数值库给出了算法，MATLAB 给出了命名与交互方式，IDL 和天文界给出了需求，
Python 社区把它们整合成了 NumPy**。Mathematica 是同一家族里走符号计算与函数式路线的分支。

---

## 2. 底层：Fortran 与数值库

几乎所有数学软件（MATLAB、Mathematica、NumPy、R、Julia）的线性代数和 FFT，最终都落到这几套库上。

| 年份 | 库 | 作者 / 机构 | 影响 |
|---|---|---|---|
| 1957 | **Fortran** | John Backus，IBM | 第一个高级科学语言；**列优先**存储、**1 起始**下标，以及 `ATAN2`、`SIGN`、`MOD` 等内置函数 |
| 1965 | **FFT** 算法 | Cooley & Tukey | 让傅里叶变换从 O(n²) 降到 O(n log n) |
| 1971 | *Handbook for Automatic Computation, Vol. II* | Wilkinson & Reinsch | Algol 写的特征值、奇异值算法，EISPACK 的源头 |
| 1974 | **EISPACK** | Argonne 实验室 | 特征值计算（Fortran） |
| 1979 | **LINPACK** | Dongarra、Bunch、**Moler**、Stewart | 线性方程组 / 最小二乘；Moler 后来写了 MATLAB |
| 1979 | **BLAS** Level 1（1988 年 Level 2、1990 年 Level 3） | Lawson、Hanson 等 | 向量/矩阵基本运算的标准接口，至今仍是 `np.dot` 的后端 |
| 1982 | **FFTPACK** | Paul Swarztrauber，NCAR | NumPy 的 FFT 一直用它，直到 2019 年改用 pocketfft |
| 1985 | **Netlib** | Dongarra & Grosse | 通过电子邮件和 FTP 分发数值软件的仓库 |
| 1986 | 《Numerical Recipes》 | Press、Teukolsky 等 | 一代科学家的算法教科书 |
| 1992 | **LAPACK** | Anderson、Dongarra 等 | 取代 LINPACK 和 EISPACK，基于 BLAS 3 分块；`np.linalg` 和 MATLAB 的核心 |
| 1997 | **FFTW** | Frigo & Johnson，MIT | 自适应最快的 FFT，MATLAB 自 6.0 起使用 |

**LAPACK 的命名直接延续到今天**。受 Fortran 77 标识符最多 6 个字符的限制，函数名采用「类型 + 矩阵类型 + 运算」的编码：
`DGESVD` = **D**ouble、**GE**neral（一般矩阵）、**SVD**。`np.linalg.svd` 底层调用 `?gesdd`，`lstsq` 调用 `?gelsd`，
`solve` 调用 `?gesv`，`eigh` 调用 `?syevd`，`cholesky` 调用 `?potrf`。
MATLAB 和 NumPy 偏爱短函数名（`eig`、`svd`、`inv`、`det`），也与这种 Fortran 传统一脉相承。

---

## 3. 各数学软件简史

### 3.1 APL（1962）→ J（1990）→ K（1993）：数组思想的源头

- Kenneth Iverson 在哈佛发明了一套数学记号，1962 年出版《A Programming Language》，1966 年 IBM 实现为 APL\360。1979 年 Iverson 获图灵奖，
  获奖演讲题为《Notation as a Tool of Thought》。
- 核心思想：**数组是一等公民**；函数自动作用于整个数组（标量扩展）；**算子（operator）**作用于函数：
  `+/` 是归约（reduce），`+\` 是扫描（scan，即累积），`∘.×` 是外积，`+.×` 是内积（矩阵乘法）。
- 这些概念几乎原样进入了 NumPy：`np.add.reduce`、`np.add.accumulate`、`np.multiply.outer`、`np.dot`；
  函数名 `ravel`、`take`、`compress`、`roll`（APL 的 rotate）也来自 APL。
- Mathematica 的 `Fold`/`FoldList`、`Outer`、`Inner`、`Take`/`Drop`、`RotateLeft`、带置换参数的 `Transpose` 同样能在 APL 里找到对应。
- J（Iverson 与 Roger Hui）引入了「函数秩（rank）」的概念，即函数作用在数组的哪几维上，这是理解广播最清晰的理论框架。
  K 后来发展成金融行业的 kdb+/q。

### 3.2 MATLAB（约 1979，1984 年商业化）：命名的主要来源

- Cleve Moler 在新墨西哥大学教数值分析时，为了让学生不写 Fortran 也能调用 LINPACK 和 EISPACK，
  用 Fortran 写了一个只有约 80 个函数的交互式矩阵计算器：**MAT**rix **LAB**oratory。
- 1984 年 Jack Little、Steve Bangert 与 Moler 创立 MathWorks，用 C 重写并商业化。
- 继承了 Fortran 的**列优先**和 **1 起始**下标；`A(:,j)` 冒号切片、`[a b; c d]` 拼接、逻辑索引 `A(A>0)`。
- `zeros`、`ones`、`eye`、`linspace`、`logspace`、`meshgrid`、`repmat`、`cumsum`、`diff`、`polyfit`、`trapz`、`fliplr`、`squeeze` 等
  大批函数名被 NumPy 沿用（见第 5 节）。matplotlib 的 `pyplot` 更是有意模仿 MATLAB 的绘图接口（`plot`、`imshow`、`subplot`、`figure`）。
- 1993 年推出 Image Processing Toolbox，`imfilter`、`medfilt2`、`bwlabel`、`regionprops` 等名字后来被 SciPy 和 scikit-image 借用。
- 开源克隆：**Octave**（1988 年构思，1993 年发布，John W. Eaton，最初是化工课程的配套软件）；
  **Scilab**（1994 年，法国 INRIA）。

### 3.3 S（1976）→ R（1993）：统计的向量化

- Bell Labs 的 John Chambers 等人设计了 S 语言；1993 年 Ross Ihaka 和 Robert Gentleman 在新西兰开发了开源实现 R。
- 向量化运算加「**循环补齐**（recycling）」规则：短向量自动重复，以对齐长向量。这和广播是同一类思想。
- `NA` 缺失值的处理方式影响了后来的 pandas。

### 3.4 IDL（1977）：天文界的老主力

- 起源于科罗拉多大学大气与空间物理实验室（LASP），最初用来处理水手号火星探测器的数据。David Stern 1977 年创立 Research Systems 公司将其商业化。
- 语法类似 Fortran，数组运算向量化，图像显示能力强，1980–2000 年代在天文、遥感、等离子体物理领域几乎是标准工具。
- NASA Goddard 维护的 **IDL Astronomy User's Library**（Wayne Landsman 等）收录了 FITS 读写、`find`（移植自 DAOPHOT 的星点检测）、
  `aper`（孔径测光）、WCS 处理等。它们的功能后来在 Python 里分别由 astropy、photutils、sep 接替。
- 授权费用昂贵、语言封闭，是天文界在 2000 年代后转向 Python 的重要原因。STScI 开发 Numarray 和 PyFITS 正是为此。

### 3.5 Macsyma（1968）→ SMP（1981）→ Mathematica（1988）：符号计算一脉

- Macsyma（MIT，用 Lisp 写成）开创了计算机代数系统；Reduce（1968）、Maple（1982，滑铁卢大学）属于同一类。
- Stephen Wolfram 在加州理工开发了 SMP（Symbolic Manipulation Program，1981 年），1987 年创立 Wolfram Research，1988 年发布 Mathematica 1.0。
- **设计来源**：Lisp 的「一切皆表达式」和函数式风格（`Map`、`Apply`、`Fold`、`Nest`，以及纯函数 `#&`），
  APL 的数组运算（`Listable`、`Outer`、`Inner`、`Transpose[a, perm]`），再加上模式匹配和规则重写。
- 关键节点：1999 年 Mathematica 4 引入**打包数组**，数值性能从此大幅提升；2008 年 Mathematica 7 引入 `Image` 对象和成体系的图像处理函数；
  2014 年 Mathematica 更名为 Wolfram Language，并推出免费的 Wolfram Engine 和 wolframscript。
- **命名哲学与 MATLAB 相反**：全称驼峰命名（`MorphologicalComponents`、`ImagePerspectiveTransformation`），
  不用缩写，读起来就是英文。

### 3.6 Yorick（约 1996）

劳伦斯利弗莫尔国家实验室（LLNL）的 David Munro 为物理模拟后处理编写的解释型数组语言，下标和**广播**规则设计得很精炼，
Numeric 的早期讨论中常被拿来参考。

### 3.7 Julia（2012）

MIT 的 Bezanson、Karpinski、Shah、Edelman 开发，目标是「像 MATLAB 一样好写，像 C 一样快」。
1 起始、列优先（MATLAB 风格），用 `f.(x)` 点语法做显式广播（NumPy 风格），靠 JIT 编译消除了「必须向量化才快」的约束。

---

## 4. Python 的数组之路

### 4.1 时间线

| 年份 | 事件 |
|---|---|
| 1991 | Guido van Rossum 发布 Python |
| 1995 | Python 的 matrix-sig 邮件组成立；Jim Hugunin 写出 **Numeric**（又称 Numerical Python），Paul Dubois（LLNL）等人后来接手维护 |
| 1995 | Fredrik Lundh 开发 PIL |
| 1996 | **Python 1.4 为 Numeric 加入扩展切片语法**（`a[::2]`、`a[..., 0]`、`Ellipsis`）和复数类型 |
| 约 1999 | Pearu Peterson 开发 f2py，自动包装 Fortran 代码 |
| 2001 | **SciPy**（Travis Oliphant、Eric Jones、Pearu Peterson）；IPython（Fernando Pérez） |
| 2001 | **Numarray**（STScI：Perry Greenfield、Todd Miller、Rick White 等），带来 `recarray`（为 FITS 二进制表设计）、非本机字节序（FITS 是大端）、内存映射、大数组支持；同期还有 PyFITS 和 Peter Verveer 的 `nd_image`（后来成为 `scipy.ndimage`） |
| 2003 | John D. Hunter 开发 **matplotlib**（原本是为了替代 MATLAB 分析癫痫患者的脑电数据） |
| 2006 | **NumPy 1.0**：Travis Oliphant 合并 Numeric 与 Numarray，结束了社区分裂 |
| 2007 | `.npy` 文件格式（NEP 1）；PEP 3118 新缓冲区协议（Oliphant 等，Python 2.6/3.0），数组可以在不同库之间零拷贝共享 |
| 2009 | scikit-image（Stéfan van der Walt） |
| 2011 | `np.einsum`（NumPy 1.6，Mark Wiebe） |
| 2011–2013 | Astropy 项目整合 PyFITS 等天文工具 |
| 2014 | **PEP 465 为 NumPy 加入矩阵乘法运算符 `@`**（Python 3.5，2015 年发布）；Jupyter 从 IPython 分离 |
| 2019 | NumPy 1.17：FFT 后端改为 pocketfft；新随机数 API `default_rng()`（默认 PCG64 算法）。本仓库使用的就是这个 API |
| 2020 | 《Array programming with NumPy》发表于 *Nature*；Python Array API 标准联盟成立 |
| 2024 | NumPy 2.0：清理 API（例如 `trapz` 改名为 `trapezoid`），调整类型提升规则（NEP 50） |

### 4.2 Python 语言为数组计算做过的改变

Python 是少有的**为一个第三方库修改过语法**的主流语言：

| 语法 | 加入版本 | 起因 |
|---|---|---|
| 扩展切片 `a[i:j:k]`、多维下标 `a[i, j]`、`...`（Ellipsis） | Python 1.4（1996） | Numeric 需要；内置列表直到 Python 2.3 才支持步长切片 |
| 复数 `1j` | Python 1.4 | 科学计算需要 |
| 缓冲区协议 PEP 3118 | Python 2.6 / 3.0 | 让 NumPy、PIL 等库零拷贝共享内存 |
| 矩阵乘法运算符 `@`、`@=`（PEP 465） | Python 3.5 | 此前 `np.dot(a, np.dot(b, c))` 写法冗长 |

---

## 5. NumPy 函数溯源表

「来源」指命名或语义最早、最直接的出处；不少函数在多个系统里都有，这里列出 NumPy 实际借鉴的那一个。
最后一列给出 Wolfram 的对应写法，方便对照。

### 5.1 创建数组

| NumPy | 来源 | 说明 | Wolfram |
|---|---|---|---|
| `zeros`、`ones` | MATLAB | | `ConstantArray[0., {h, w}]` |
| `eye`、`identity` | MATLAB `eye`（谐音 I，单位矩阵） | | `IdentityMatrix[n]` |
| `empty` | Numeric | 分配内存但不初始化，C 语言思维 | 无（值语义不需要） |
| `arange` | Numeric，取「**a**rray **range**」之意 | 来自 Python `range`；概念上相当于 APL 的 `⍳`（iota） | `Range[0, n-1]`（`Range` 本身对应 APL 的 `⍳`，同样从 1 开始） |
| `linspace`、`logspace` | MATLAB | 「linearly spaced」 | `Subdivide[a, b, n-1]`、`Range[a, b, step]` |
| `meshgrid` | MATLAB | 默认 `indexing='xy'` 也沿用了 MATLAB 的习惯 | `Table[{x, y}, …]`、`Outer[List, ys, xs]` |
| `mgrid`、`ogrid` | NumPy 自创 | 借助切片语法模拟 MATLAB 的 `ndgrid` | — |
| `r_`、`c_` | NumPy 自创 | 模拟 MATLAB 的 `[a, b]`、`[a; b]` 拼接 | `Join` |
| `diag`、`tril`、`triu`、`kron`、`trace` | MATLAB（源自线性代数记号） | | `DiagonalMatrix`、`LowerTriangularize`、`KroneckerProduct`、`Tr` |
| `fromfunction` | Numeric | | `Table`、`Array[f, {h, w}]` |

### 5.2 形状与重排

| NumPy | 来源 | 说明 | Wolfram |
|---|---|---|---|
| `reshape` | APL 二元 `⍴`（reshape）；MATLAB `reshape` | NumPy 默认按行（C 顺序）填充，MATLAB 按列 | `ArrayReshape` |
| `ravel` | **APL** 的 `,`（ravel，意为「拆散成一条线」） | 这个词几乎只在 APL 和 NumPy 中使用 | `Flatten` |
| `shape` | APL 一元 `⍴` | | `Dimensions` |
| `transpose`、`.T` | APL `⍉`；MATLAB `'` | 置换参数的语义与 APL/Wolfram 互逆（见第 6 节） | `Transpose` |
| `squeeze` | MATLAB | | `Flatten` / 手动 |
| `concatenate`、`hstack`、`vstack` | MATLAB `[a b]` / `[a; b]`、`horzcat`/`vertcat` | | `Join`、`ArrayFlatten` |
| `tile` | MATLAB `repmat` | | `ConstantArray` + `ArrayFlatten` |
| `repeat` | APL replicate（`/`） | | `Riffle` / `Table` |
| `compress` | **APL** compress（布尔 `/`） | 与 APL 语义一致 | `Pick` |
| `take` | **APL** `↑`（take）的名字 | NumPy 中是按下标取值，与 APL 语义不同 | `Take`（与 APL 语义一致，负数表示从末尾取） |
| `roll` | **APL** `⌽`（rotate） | | `RotateLeft`、`RotateRight` |
| `flip`、`fliplr`、`flipud`、`rot90` | MATLAB（APL 的 `⌽`/`⊖` reverse） | | `Reverse`、`Reverse[a, 2]` |
| `newaxis`（`None`） | Numeric | | 不需要（线程化规则不同） |
| `pad` | NumPy 1.7（模式名参考 MATLAB `padarray`） | | `ArrayPad` |

### 5.3 ufunc 方法：APL 算子的直系后代

| NumPy | APL | Wolfram | 含义 |
|---|---|---|---|
| `np.add.reduce(a)` | `+/a` | `Fold[Plus, a]`、`Total[a]` | 归约 |
| `np.add.accumulate(a)` | `+\a` | `FoldList[Plus, a]`、`Accumulate[a]` | 扫描（累积） |
| `np.multiply.outer(a, b)` | `a ∘.× b` | `Outer[Times, a, b]` | 外积 |
| `np.dot(a, b)`、`a @ b` | `a +.× b` | `a . b`、`Inner[Times, a, b, Plus]` | 内积（`Inner` 是对 APL 内积的完整推广） |
| `np.add.reduceat` | 分段归约 | `Total /@ TakeList[a, lens]` | |
| `np.einsum` | 爱因斯坦求和约定（1916 年广义相对论论文） | `TensorContract`、`Inner` | 张量缩并 |

### 5.4 归约与统计

| NumPy | 来源 | 说明 | Wolfram |
|---|---|---|---|
| `sum`、`prod`、`max`、`min`、`mean`、`median`、`sort` | MATLAB（通用数学名称） | `axis` 参数对应 MATLAB 的 `dim` | `Total`、`Times @@`、`Max`、`Mean`、`Median`、`Sort` |
| `cumsum`、`cumprod`、`diff` | MATLAB | | `Accumulate`、`FoldList[Times, a]`、`Differences` |
| `std`、`var` | MATLAB | NumPy 默认 `ddof=0`（总体）；MATLAB 和 Wolfram 默认 n−1（样本） | `StandardDeviation`、`Variance` |
| `argmax`、`argmin`、`argsort` | Numeric（「arg」源自数学记号 argmax） | MATLAB 中是 `[m, i] = max(a)` | `Ordering[a, -1]`、`Ordering` |
| `nanmean`、`nanmedian` 等 | MATLAB Statistics Toolbox 的 `nanmean` | NumPy 1.8 加入 | 无，用掩膜 |
| `percentile` | MATLAB `prctile` | NumPy 用 0–100 | `Quantile`（0–1） |
| `ptp` | 工程术语「peak to peak」 | | `Max[a] - Min[a]` |
| `histogram` | MATLAB `hist`/`histc` | | `HistogramList`、`BinCounts` |
| `unique` | MATLAB | | `Union`、`DeleteDuplicates` |
| `cov`、`corrcoef` | MATLAB | | `Covariance`、`Correlation` |
| `any`、`all` | MATLAB（APL 的 `∨/`、`∧/`） | | `AnyTrue`、`AllTrue`、`Or @@` |
| `searchsorted`、`bincount`、`digitize` | NumPy / Numeric 自创 | | `BinLists`、`Nearest` |
| `median_abs_deviation`（scipy.stats） | Hampel 1974 | | `MedianDeviation`（内置） |

### 5.5 逻辑与条件

| NumPy | 来源 | Wolfram |
|---|---|---|
| 布尔索引 `a[a > 0]` | MATLAB 逻辑索引 | `Pick[a, UnitStep[a], 1]` |
| `where(c, x, y)` | Fortran 90 的 `WHERE` 结构和 `MERGE` 内置函数 | `y + UnitStep[c] (x - y)`、`If`（逐元素，慢） |
| `nonzero`、`argwhere` | MATLAB `find` | `Position`、`SparseArray[…]["NonzeroPositions"]` |
| `isnan`、`isinf`、`isfinite` | IEEE 754 浮点标准（1985）与 C99 | `NumericQ` 等 |
| `clip` | 信号处理术语（clipping，削波） | `Clip`（Wolfram 早在 1988 年版本就有） |
| `select`、`piecewise` | 数学分段函数 | `Piecewise` |
| `maximum`、`minimum`（逐元素） | ufunc 体系 | `Ramp` 组合（`Max` 不是逐元素） |

### 5.6 数学、信号与线性代数

| NumPy | 来源 | 说明 | Wolfram |
|---|---|---|---|
| `sin`、`exp`、`log`、`sqrt` | Fortran 内置函数和 C `libm` | | 同名首字母大写 |
| `arctan2` | Fortran `ATAN2`（1960 年代） | 参数顺序 (y, x) 也来自 Fortran | `ArcTan[x, y]`（**参数顺序相反**） |
| `hypot` | C `libm` | | `Norm[{x, y}]` |
| `sign`、`mod`、`floor`、`ceil` | Fortran / C | | `Sign`、`Mod`、`Floor`、`Ceiling` |
| `convolve`、`correlate` | MATLAB `conv`、`xcorr` | | `ListConvolve`、`ListCorrelate` |
| `polyfit`、`polyval`、`roots`、`poly` | MATLAB | 系数按降幂排列也沿用 MATLAB；新的 `np.polynomial` 改为升幂 | `Fit`、`LinearModelFit`、`NSolve` |
| `interp` | MATLAB `interp1` | | `Interpolation`、`ListInterpolation` |
| `trapz` → `trapezoid` | MATLAB `trapz` | NumPy 2.0 改名 | `Integrate` / `NIntegrate` |
| `gradient` | MATLAB | | `Differences` / `DerivativeFilter` |
| `unwrap`、`sinc` | MATLAB Signal Processing Toolbox | `np.sinc` 是归一化定义（含 π） | `Sinc`（**非归一化**定义） |
| `fft`、`ifft`、`rfft` | MATLAB `fft`；底层 FFTPACK（1982），2019 年起改用 pocketfft | | `Fourier`（指数符号相反） |
| `fftshift`、`fftfreq` | MATLAB `fftshift` | | `RotateLeft[…, n/2]` |
| `linalg.solve`、`lstsq`、`svd`、`eig`、`eigh`、`qr`、`cholesky`、`det`、`inv` | **LAPACK**（Fortran）；名字来自 MATLAB | | `LinearSolve`、`LeastSquares`、`SingularValueDecomposition`、`Eigensystem`、`QRDecomposition`、`CholeskyDecomposition`、`Det`、`Inverse` |
| `linalg.pinv`、`matrix_rank`、`cond`、`norm` | MATLAB `pinv`、`rank`、`cond`、`norm` | | `PseudoInverse`、`MatrixRank`、`Norm` |
| `random.rand`、`randn`（旧 API） | MATLAB `rand`、`randn` | 底层是 Mersenne Twister（Matsumoto & Nishimura 1997） | `RandomReal`、`RandomVariate` |
| `random.default_rng()` | NumPy 1.17 新设计 | PCG64（O'Neill 2014） | `SeedRandom` |

### 5.7 数据与 I/O

| NumPy | 来源 | 说明 |
|---|---|---|
| `ma`（掩码数组） | Numeric 的 MA 模块（Paul Dubois，LLNL） | 用掩码处理缺失值，是 IDL/天文用户的刚需 |
| `recarray`、结构化 dtype | **Numarray（STScI）** | 为 FITS 二进制表格设计 |
| 非本机字节序 dtype（`'>i2'`） | **Numarray** | FITS 是大端格式 |
| `memmap` | Numarray | 处理比内存还大的图像；本仓库 `milkyway/` 的叠加缓存就用它 |
| `loadtxt`、`savetxt` | MATLAB `load -ascii`、`save -ascii` | |
| `.npy`、`.npz` | NEP 1（2007） | 对应 MATLAB 的 `.mat` |

---

## 6. 设计约定的来历

| 约定 | NumPy | 出处 | 其他系统 |
|---|---|---|---|
| 下标起点 | 0 | C 语言、Python（Dijkstra 1982 年的短文《Why numbering should start at zero》常被引用） | Fortran、MATLAB、R、Julia、Wolfram 从 1 开始；APL 可以用 `⎕IO` 切换，默认 1 |
| 切片区间 | 左闭右开 `[i, j)` | Python | MATLAB、Wolfram 两端都包含 |
| 内存布局 | 行优先（C 顺序），可选 `order='F'` | C 语言；`order='F'` 是为了与 Fortran / LAPACK 零拷贝互通 | Fortran、MATLAB、Julia、R 列优先；Wolfram 嵌套列表相当于行优先 |
| 广播 | 尾维对齐，长度 1 的维度可扩展 | 思想来自 APL 标量扩展和 J 的秩一致规则；Numeric 从 1995 年起就有，并使「broadcasting」一词流行起来 | R 的循环补齐；MATLAB 直到 2016b 才支持隐式扩展；Wolfram 按第一维线程化 |
| 转置置换 | `axes[i]` 表示结果第 i 维取自输入的哪一维 | NumPy 自定 | APL 二元 `⍉` 和 Wolfram `Transpose` 相反：第 k 维要去结果的哪一维 |
| 标准差分母 | n（`ddof=0`） | Numeric | MATLAB、R、Wolfram 用 n−1 |
| 缺失值 | NaN | IEEE 754（1985） | R 用 `NA`；Wolfram 用 `Missing[]` |
| 视图 vs 拷贝 | 切片返回视图 | Numeric 为节省内存（当时内存很贵） | MATLAB 写时复制；Wolfram 值语义 |
| dtype 名称 | `float32`、`int16`、`uint8` | C99 `<stdint.h>` 风格 | FITS 的 BITPIX = 8/16/32/−32/−64；Fortran `REAL*4` |
| 矩阵乘法 | `@`（2014 年前用 `np.dot`；`*` 是逐元素） | PEP 465 | MATLAB 中 `*` 是矩阵乘、`.*` 是逐元素；Wolfram 中 `.` 是矩阵乘 |

---

## 7. 图像处理函数溯源

### 7.1 从 MATLAB IPT 到 Python 和 Wolfram

MATLAB Image Processing Toolbox（1993）定义了一代人的图像处理词汇。Python 生态大量沿用了它的名字，Wolfram 则用全称重新命名。

| MATLAB IPT | Python（scipy.ndimage / scikit-image / OpenCV） | Wolfram | 备注 |
|---|---|---|---|
| `imread`、`imwrite` | `skimage.io.imread`、`imageio`、`cv2.imread` | `Import`、`Export` | `im` 前缀来自 MATLAB |
| `imshow` | `matplotlib.pyplot.imshow` | 直接显示 `Image` | matplotlib 有意照搬 |
| `rgb2gray`、`rgb2lab` | `skimage.color.rgb2gray`、`rgb2lab` | `ColorConvert` | **同名** |
| `imresize`、`imrotate`、`imcrop` | `skimage.transform.resize`、`rotate` | `ImageResize`、`ImageRotate`、`ImageCrop` | |
| `fspecial`、`imfilter` | `ndimage.convolve`、`cv2.filter2D` | `ImageConvolve`、`GaussianMatrix` | |
| `medfilt2` | `scipy.signal.medfilt2d`、`ndimage.median_filter` | `MedianFilter` | **几乎同名** |
| `bwlabel` | `ndimage.label`、`skimage.measure.label` | `MorphologicalComponents` | `bw` 指黑白（二值）图 |
| `regionprops` | `skimage.measure.regionprops` | `ComponentMeasurements` | **同名** |
| `imerode`、`imdilate`、`strel('disk', r)` | `skimage.morphology.erosion`、`disk` | `Erosion`、`DiskMatrix` | |
| `bwareaopen` | `skimage.morphology.remove_small_objects` | `DeleteSmallComponents` | |
| `graythresh`（Otsu 1979） | `skimage.filters.threshold_otsu` | `FindThreshold`、`Binarize` | |
| `edge(I, 'canny')`（Canny 1986） | `skimage.feature.canny`、`cv2.Canny` | `EdgeDetect` | |
| `imadjust`、`histeq` | `skimage.exposure.rescale_intensity`、`equalize_hist` | `ImageAdjust`、`HistogramTransform` | |
| `deconvlucy`、`deconvwnr` | `skimage.restoration.richardson_lucy`、`wiener` | `ImageDeconvolve`、`WienerFilter` | |
| `imregcorr`（相位相关）、`imregister` | `skimage.registration.phase_cross_correlation` | `ImageAlign` | |
| `fitgeotrans`、`imwarp` | `skimage.transform.estimate_transform`、`warp` | `FindGeometricTransform`、`ImagePerspectiveTransformation` | |
| `wavedec2`（Wavelet Toolbox） | `pywt.wavedec2` | `DiscreteWaveletTransform` | **同名** |

`scipy.ndimage` 本身来自 Numarray 的 `nd_image` 模块（Peter J. Verveer），一开始就是 N 维设计，不限于二维图像。
OpenCV（Intel，1999–2000）是另一条独立路线，命名源自 C 语言的 `cv*` 前缀，其 BGR 通道顺序是早期摄像头和 Windows 位图格式留下的历史遗留。

### 7.2 命名风格的三种传统

| 风格 | 例子 | 来源 |
|---|---|---|
| 缩写 | `eig`、`svd`、`bwlabel`、`medfilt2`、`DGESVD` | Fortran 6 字符限制 → LAPACK → MATLAB → NumPy |
| 符号 | `⍴`、`⍉`、`+/`、`∘.×` | APL |
| 英文全称 | `MorphologicalComponents`、`SingularValueDecomposition` | Wolfram 的设计原则：名字就是文档 |

---

## 8. 本仓库天文算法的出处

| 算法 / 概念 | 本仓库位置 | 原始出处 |
|---|---|---|
| FITS 文件格式 | `astrotools.load_fits`、`AstroTools.wl` `ReadFITS` | Wells, Greisen & Harten 1981, *A&AS* 44, 363 |
| WCS TAN 投影 | `compose.py`（astropy）、`WorldToPixel` | Greisen & Calabretta 2002, *A&A* 395（WCS Paper I/II） |
| 鲁棒 σ：MAD × 1.4826 | `mad_std`、`MADSigma` | Hampel 1974；1.4826 = 1/Φ⁻¹(3/4)，使 MAD 成为高斯 σ 的无偏估计 |
| σ-clip 叠加 | `stack.py`、`sigma_clip_mean` | IRAF `imcombine`（NOAO，1980 年代），后来的 PixInsight ImageIntegration、ccdproc 都源于此 |
| 网格背景估计 | `background_map` | SExtractor：Bertin & Arnouts 1996, *A&AS* 117, 393；Python 版 sep：Barbary 2016 |
| 星点检测、矩测量 | `detect_stars`、`DetectStars` | DAOPHOT FIND：Stetson 1987, *PASP* 99, 191；SExtractor |
| 三角形匹配配准 | `stack.py`（astroalign） | Groth 1986, *AJ* 91, 1244；Valdes et al. 1995（FOCAS）；astroalign：Beroiz et al. 2020 |
| RANSAC | `stack.py`、`milkyway/` | Fischler & Bolles 1981, *CACM* 24 |
| KD 树最近邻 | `cKDTree`、`Nearest` | Bentley 1975, *CACM* 18 |
| 相位相关 | `milkyway/`、`PhaseShift` | Kuglin & Hines 1975 |
| 归一化 DLT 单应矩阵 | `milkyway/fit_homography` | Hartley 1997《In defense of the eight-point algorithm》；Hartley & Zisserman《Multiple View Geometry》 |
| Lanczos 重采样 | `cv2.INTER_LANCZOS4`、`Resampling -> "Lanczos"` | Lanczos 1956 年的 σ 因子；Duchon 1979（Lanczos 滤波器） |
| **MTF 中间调传递函数** | `mtf`、`MTF` | PixInsight（Juan Conejero，2000 年代中期）。数学上**与 Schlick 1994 的 bias 函数完全相同**（取 a = 1−m，见下文） |
| arcsinh 拉伸 | `asinh_stretch` | Lupton, Gunn & Szalay 1999（asinh 星等）；Lupton et al. 2004, *PASP* 116, 133（彩色合成） |
| starlet（à trous 小波） | `starlet`、`StarletLayers` | Holschneider et al. 1989；Starck & Murtagh 1994；Starck, Fadili & Murtagh 2007 |
| Richardson–Lucy 反卷积 | `deconvolve.py`、`Deconvolve` | Richardson 1972, *JOSA* 62；Lucy 1974, *AJ* 79 |
| 双边滤波 | `compose.py` 降噪 | Tomasi & Manduchi 1998 |
| CIE L\*a\*b\* 色彩空间 | `lrgb_combine`、`LRGBCombine` | CIE 1976 |
| 哈勃色板 SHO | `compose.py` | Hester & Scowen 1995 年拍摄的 M16「创生之柱」（SII→红、Hα→绿、OIII→蓝） |
| SCNR 去绿 | `scnr`、`SCNR` | 源自业余天文界的去绿方法（如 Photoshop 插件 HLVG，Rogelio Bernal Andreo），PixInsight 将其实现为 SCNR |
| Bayer 滤色阵列 | `milkyway/`（rawpy） | Bryce Bayer 1976（柯达专利）；解码：dcraw（Dave Coffin，1997）→ LibRaw（2008）→ rawpy |
| NPF 曝光法则 | `milkyway/exposure_calc.py` | Frédéric Michaud（法国勒阿弗尔天文学会），2010 年代 |

### MTF 与 Schlick bias 函数是同一个公式

Christophe Schlick 1994 年在《Graphics Gems IV》中提出 bias 函数，作为 Perlin bias 的快速替代：

```
bias_a(x) = x / ((1/a − 2)(1 − x) + 1)
```

PixInsight 的中间调传递函数是

```
MTF(m, x) = (m − 1)x / ((2m − 1)x − m)
```

令 a = 1 − m，分子分母同乘 a 化简即可证明 `MTF(m, x) ≡ bias_{1−m}(x)`。本仓库用 numpy 数值验证过，误差约 1e-16。
两者都满足 f(0) = 0、f(1) = 1；不同的是 MTF 规定 `MTF(m, m) = 0.5`，而 Schlick 规定 `bias_a(0.5) = a`。
这是同一个有理函数在计算机图形学和天文图像处理中被分别使用的例子。

---

## 9. 参考

**历史与综述**
- Iverson, K. E. *A Programming Language*. Wiley, 1962；Iverson, *Notation as a Tool of Thought*, CACM 23(8), 1980（图灵奖演讲）
- Moler, C. & Little, J. *A History of MATLAB*. Proc. ACM Program. Lang. 4 (HOPL), 2020
- Harris, C. R. et al. *Array programming with NumPy*. Nature 585, 357–362, 2020
- Oliphant, T. E. *Guide to NumPy*, 2006；Oliphant, *Python for Scientific Computing*, CiSE 9(3), 2007
- Greenfield, P. et al. *numarray: A New Scientific Array Package for Python*, PyCon 2003
- Virtanen, P. et al. *SciPy 1.0*. Nature Methods 17, 2020
- Hunter, J. D. *Matplotlib: A 2D Graphics Environment*. CiSE 9(3), 2007
- van der Walt, S. et al. *scikit-image: image processing in Python*. PeerJ 2, 2014
- Astropy Collaboration, *A&A* 558, A33, 2013
- Wolfram, S. 关于 Mathematica 设计与历史的文章（stephenwolfram.com）
- PEP 465 – A dedicated infix operator for matrix multiplication；PEP 3118 – Revising the buffer protocol
- Anderson, E. et al. *LAPACK Users' Guide*, SIAM, 1992/1999；Dongarra et al. *LINPACK Users' Guide*, SIAM, 1979

**算法原始论文**：见第 8 节表格。
