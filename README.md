# Ansys HPGe 自动化仿真工具

从 SolidWorks 装配体出发,自动在 Ansys Maxwell 3D 中构建 HPGe 半导体探测器模型,求解 **真实电场**、**电容矩阵**、**每个电极的权重势**,并按日期归档输出。支持任意数量电极的多电极探测器。

> **状态:规范定稿,实现待开发(`run.py` 尚未实现,仓库内 `run_icpc.py` 是手工 ICPC 原型)。**
> 本 README 是工具的输入/输出契约。规范变更须先改本文档、再改代码,反之即视为 bug。

---

## 目录

1. [工作流](#1-工作流)
2. [输入规范](#2-输入规范)
3. [CLI 用法](#3-cli-用法)
4. [输出结构](#4-输出结构)
5. [错误处理 & 恢复](#5-错误处理--恢复)
6. [依赖](#6-依赖)

---

## 1. 工作流

```
SolidWorks 装配体(STEP AP242) + crystal.yaml + operating.yaml
        │
        ▼
   Pre-flight 校验(秒级,失败立即 abort)
        │
        ▼
   AEDT headless 启动 → 导入 STEP → 拓扑识别零件
        │
        ▼
   设材料(Ge / PEC)、ρ 表达式、自适应 setup
        │
        ▼
   解 real_field(adaptive 8-10 pass)
        │
        ├── 导出 5 件 .h5 → real_field/
        └── 提取电容矩阵 → capacitance_matrix.csv
        │
        ▼
   循环 N 个电极:
     duplicate_design → 改 BC → 复制 mesh → 单 pass 解
        │
        └── 导出 5 件 .h5 → weighting_potential/<电极>/
        │
        ▼
   写 manifest.yaml,status = success → 关闭 AEDT
        │
        ▼
   输出落到 ./data_out/YYYYMMDD/<step>__<op>/
```

---

## 2. 输入规范

### 2.1 SolidWorks 装配体 → STEP AP242

**格式:** STEP AP242 (ISO 10303-242),扩展名 `.step` 或 `.stp`。

**零件命名约定**(在 SolidWorks 里命名,STEP 会保留 `PRODUCT` 名):

| 角色 | 零件名 pattern | 数量 |
|---|---|---|
| 晶体本体 | `Crystal` | 恰好 1 个 |
| P+ 电极 | `PPlus_NN`(NN = 01, 02, ...,两位零填充) | ≥ 1 个 |
| N+ 电极 | `NPlus_NN`(NN = 01, 02, ...,两位零填充) | ≥ 1 个 |

**硬性规则:**
- 名称必须为 **ASCII**,不含空格、中文、下划线以外的特殊符号
- 所有零件**共面紧贴但体积不重叠**(STEP/FEM 不允许实体重叠)
- 单电极示例:`Crystal`、`PPlus_01`、`NPlus_01`
- 多电极示例:`Crystal`、`PPlus_01`、`NPlus_01`、`NPlus_02`、… 、`NPlus_36`

**自动校验**(导入后立即执行,任一失败 abort):

1. 存在且只有一个 `Crystal`
2. 至少一个 `PPlus_*` 和一个 `NPlus_*`,均匹配 `_NN` 两位编号 pattern
3. 每个电极与 Crystal 有面接触(共享面)
4. Crystal 是所有零件中体积最大者
5. 零件名为 ASCII,无非法字符
6. STEP 文件可被 AEDT 成功导入

### 2.2 crystal.yaml(物理参数,绑定晶体)

**文件位置:** 默认与 STEP 同目录同名(`detector_A.step` → `detector_A.yaml`);可用 `--crystal PATH` 覆盖。

**完整模板:** 见 [`crystal.example.yaml`](crystal.example.yaml)。

**字段:**

```yaml
physics:
  carrier_type: p-type           # p-type | n-type
                                 #   p-type → ρ < 0 (净受主, 默认本次项目)
                                 #   n-type → ρ > 0 (净施主)
  impurity_concentration:        # cm⁻³, 永远为正, 符号由 carrier_type 注入
    PPlus_end:    2.0e10         # P+ 触面端杂质浓度 (默认 = 高浓度端)
    opposite_end: 1.0e10         # 对面端杂质浓度
  temperature: 77                # K, 仅记录于 manifest, 不影响计算
material:
  crystal: Germanium             # AEDT 材料库名, 默认 Germanium (ε_r = 16)
                                 # PPlus_*/NPlus_* 强制 PEC, 无需配置
```

**ρ 内部换算:**

```
rho_SI [C/m³] = sign × ELEMENTARY_CHARGE × concentration [cm⁻³] × 1e6
sign = -1 if carrier_type == p-type else +1
```

**ρ 线性梯度的空间形式:**

```
P+ 触面 与 Crystal 的公共面 → 合并质心 C_P
Crystal 其余面中, 距 C_P 最远且法向近反平行的面 → 质心 C_O
axis = (C_O - C_P) / |C_O - C_P|, 长度 L = |C_O - C_P|

ρ(r) = ρ_P + (ρ_O - ρ_P) × dot(r - C_P, axis) / L
```

多 P+ 时合并所有 `PPlus_*` 触面取整体质心,逻辑不变。

**字段校验:**
- `carrier_type` ∈ `{p-type, n-type}`
- `impurity_concentration.*` ∈ `[1e6, 1e13]` cm⁻³
- `temperature` > 0

### 2.3 operating.yaml(运行参数,绑定本次仿真)

**文件位置:** 通过 `--op PATH` 显式指定,**必填**(除非用 `--voltage` 在 CLI 完整覆盖且提供 `--run-name`)。

**完整模板:** 见 [`operating.example.yaml`](operating.example.yaml)。

**字段:**

```yaml
voltages:                        # V, 显式赋值, 优先级最高
  PPlus_01: 0
  NPlus_01: 2500

defaults:                        # 通配符默认值, 被 voltages 段覆盖
  "PPlus_*": 0
  "NPlus_*": 2500

grid:
  step_mm: 0.5                   # 各向同性网格步长
  padding_mm: 0.5                # bbox 外余量

# output_root: /custom/path      # 可选,覆盖默认 ./data_out
```

**电压覆盖规则:**
1. `defaults:` 段先按通配符展开到所有匹配电极
2. `voltages:` 段显式值覆盖默认值
3. CLI `--voltage NAME=V` 再覆盖一次

**字段校验:**
- 展开后,Maxwell 模型里**每一个非 Crystal 的 body 都必须有电压**
- 缺失任何电极 → abort 并列出缺失电极名
- 电压绝对值 > 10000 V → warning(不 abort)
- `step_mm` ∈ `[0.05, 5.0]`,`padding_mm` ∈ `[0.0, 10.0]`

---

## 3. CLI 用法

```
python run.py CRYSTAL.step [选项]
```

**必填:**

| 参数 | 说明 |
|---|---|
| `CRYSTAL.step` | SolidWorks 导出的 STEP AP242 装配体(位置参数) |
| `--op PATH` | operating.yaml 路径(必填,除非用 `--voltage` 全覆盖且提供 `--run-name`) |

**主要选项:**

| 选项 | 默认 | 说明 |
|---|---|---|
| `--crystal PATH` | `<step>.yaml` 同目录同名 | 覆盖 crystal.yaml 查找路径 |
| `--voltage NAME=V` | — | 单个电极电压覆盖,可多次 |
| `--grid-step MM` | operating.yaml 中的值 | 覆盖网格步长 |
| `--output-root PATH` | `./data_out` | 覆盖输出根目录 |
| `--run-name NAME` | `<step>__<op>` | 覆盖子目录名 |
| `--force` | off | 允许覆盖已存在的输出目录 |
| `--resume` | off | 从已有 partial 状态恢复,跳过已完成 design |
| `--gui` | off (默认 headless) | 显示 AEDT GUI |
| `--reuse-aedt` | off | 复用已存在的 .aedt(只改电压重算),需 STEP hash 一致 |
| `--cleanup-aedt` | off | 收尾时删除中间权重势 designs,.aedt 减肥 |
| `--aedt-version VER` | `2025.2` | AEDT 版本 |

**示例:**

```bash
# 标准跑法:STEP + 同名 crystal.yaml + 显式 operating.yaml
python run.py detector_A.step --op bias_2500V.yaml

# 同晶体跑不同电压
python run.py detector_A.step --op bias_3000V.yaml

# CLI 单点覆盖
python run.py detector_A.step --op bias_2500V.yaml --voltage NPlus_01=2800

# 从中断处恢复
python run.py detector_A.step --op bias_2500V.yaml --resume

# debug: 开 GUI, 不覆盖输出
python run.py detector_A.step --op bias_2500V.yaml --gui

# 收尾归档: 跑完删中间 design
python run.py detector_A.step --op bias_2500V.yaml --cleanup-aedt
```

---

## 4. 输出结构

### 4.1 目录布局

```
data_out/
└── 20260513/                                  # YYYYMMDD,本地时区,脚本启动时刻
    └── detector_A__bias_2500V/                # <step_stem>__<op_stem>
        ├── manifest.yaml                      # 运行参数快照 + 状态机
        ├── capacitance_matrix.csv             # N×N Maxwell 电容矩阵
        ├── real_field/                        # 带 ρ 的真实电场
        │   ├── phi.h5
        │   ├── E_x.h5
        │   ├── E_y.h5
        │   ├── E_z.h5
        │   └── E_mag.h5
        ├── weighting_potential/               # N 个权重势,无 ρ
        │   ├── PPlus_01/
        │   │   ├── phi.h5
        │   │   ├── E_x.h5
        │   │   ├── E_y.h5
        │   │   ├── E_z.h5
        │   │   └── E_mag.h5
        │   ├── NPlus_01/
        │   │   └── ... (同上 5 件)
        │   └── NPlus_02/
        │       └── ...
        ├── detector_A.aedt                    # AEDT 项目(--cleanup-aedt 时减肥)
        └── run.log                            # 完整 stdout/stderr 日志
```

**目录冲突策略:**
- 默认:存在即 abort,要求 `--force` 或 `--resume`
- `--force`:整目录覆盖(危险,需手动确认)
- `--resume`:保留现有目录,补齐未完成 design

### 4.2 manifest.yaml schema

```yaml
run_id: detector_A__bias_2500V
timestamp: 2026-05-13T15:30:45+08:00         # ISO 8601,本地时区
status: success                              # running | success | partial | failed
aedt_version: "2025.2"

inputs:
  step:
    path: /abs/path/detector_A.step
    sha256: abc123...                        # 完整哈希,验真 + reuse-aedt 检查
  crystal_yaml:                              # 完整内嵌,反序列化后的 dict
    physics:
      carrier_type: p-type
      ...
  operating_yaml:                            # 完整内嵌(含 CLI 覆盖后的最终值)
    voltages:
      PPlus_01: 0
      NPlus_01: 2500
    grid:
      step_mm: 0.5
      padding_mm: 0.5

geometry_detected:
  crystal_volume_mm3: 279500.0
  electrodes: [PPlus_01, NPlus_01]
  pplus_face_centroid_mm: [0.0, 0.0, 65.0]
  opposite_face_centroid_mm: [0.0, 0.0, 0.0]
  bbox_mm: { min: [-37, -37, 0], max: [37, 37, 65] }
  grid_shape: [148, 148, 130]

completed_designs:                            # 已完成的 design 名列表(支持 --resume)
  - real_field
  - wpot_PPlus_01
  - wpot_NPlus_01

errors: []                                    # status != success 时这里有详细信息
```

### 4.3 .h5 内部 schema

每个 `*.h5` 是一个单量(φ 或 Ex/Ey/Ez/E_mag)的三维标量场。

```
real_field/phi.h5
├── /grid
│   ├── x: float32[Nx]        # mm
│   ├── y: float32[Ny]
│   └── z: float32[Nz]
├── /data: float32[Nx, Ny, Nz]   # 单位见 attrs.unit
└── attrs (HDF5 attributes 挂在根)
    ├── quantity: "phi" | "E_x" | "E_y" | "E_z" | "E_mag"
    ├── unit: "V" | "V/m"
    ├── grid_assignment: "Crystal"     # Crystal 之外的格点值为 0
    ├── source_step: "detector_A.step"
    ├── step_sha256: "abc123..."
    ├── timestamp: "2026-05-13T15:30:45+08:00"
    └── aedt_version: "2025.2"
```

**压缩:** gzip level 4(平衡速度/比);可后期 `--compression-level N` 调整。

### 4.4 capacitance_matrix.csv

```csv
,PPlus_01,NPlus_01
PPlus_01,12.34,-12.30
NPlus_01,-12.30,12.45
```

- 行列标签 = 电极名,顺序为 `PPlus_*` 先、`NPlus_*` 后,组内按编号升序
- 单位 **pF**
- 约定:Maxwell 电容矩阵(对角自电容,非对角负互电容)

---

## 5. 错误处理 & 恢复

### 5.1 Pre-flight 校验(秒级,Maxwell 启动前)

按顺序检查,任一失败立即 abort 并打印**可执行**的错误信息:

1. STEP 文件存在且可读
2. crystal.yaml 存在 + schema 合法 + 取值范围合理
3. operating.yaml 存在 + schema 合法(若需要)
4. 输出目录可写;若已存在,看 `--force` / `--resume` flag
5. 磁盘空间 > 2 × 估算输出大小
6. AEDT license 可 checkout(试探性 acquire-release)
7. STEP 试探性导入,零件命名合规,拓扑校验通过

### 5.2 Checkpoint 粒度

每完成一个 design 后:
1. 导出 5 件 .h5 到对应子目录
2. 原子更新 `manifest.yaml` 的 `completed_designs` 列表(写到 `.tmp` → `rename`)
3. 写入 `run.log` 一条 INFO 日志

中断后:
- 找已有输出目录的 `manifest.yaml`
- 若 `status: partial`,`--resume` 跳过 `completed_designs` 中列出的 designs,从下一个未完成的开始

### 5.3 信号处理

- `SIGINT` / `SIGTERM`:捕获后优雅关闭 AEDT(`close_projects=True, close_desktop=True`),更新 manifest `status: partial`,退出码 130
- AEDT 崩溃:捕获 RpyC 异常,manifest `status: failed`,记录最后一次进行中的 design,退出码 1
- 不自动重连或重启 AEDT

### 5.4 状态转移

```
status:
  running   → success    (全部完成)
  running   → partial    (Ctrl+C / SIGTERM)
  running   → failed     (异常,AEDT 崩溃,license 丢失)
  partial   → running    (--resume 进入)
  partial   → success    (resume 跑完剩余)
  success   → 终态
  failed    → 终态(--resume 不接受,需 --force 重跑)
```

---

## 6. 依赖

### 6.1 软件

| 软件 | 版本 |
|---|---|
| Ansys Electronics Desktop | 2025.2(`--aedt-version` 可改) |
| SolidWorks | 任何能导出 STEP AP242 的版本 |
| Python | ≥ 3.10 |

### 6.2 Python 包

```
ansys-aedt-core >= 0.10        # AEDT Python 接口(原 pyaedt)
numpy >= 1.24
h5py >= 3.9
pyyaml >= 6.0
```

(`argparse` 在 stdlib,不另列。)

### 6.3 SolidWorks 导出 STEP 的正确姿势

1. 文件 → 另存为 → 类型选 `STEP AP242 (*.step)`
2. 选项 → 勾选 "保留装配体结构" / "Export part names"
3. 验证:用文本编辑器打开 STEP,搜 `PRODUCT(`,应能看到三个零件的中文/英文原名

---

## 附录 A:与现有 `run_icpc.py` 的差异

| 维度 | `run_icpc.py`(当前) | `run.py`(本规范) |
|---|---|---|
| 几何来源 | 手工搭好的 `ICPC.aedt` | 从 STEP 自动构建 |
| 电极数量 | 硬编码 2 个 | 任意 N |
| 电压输入 | 脚本内常量 | operating.yaml + CLI |
| ρ 分布 | 手工设的体电荷 | 线性梯度,从浓度自动推导 |
| 输出位置 | `./ICPC_out/` 扁平 | `/data_out/YYYYMMDD/<step>__<op>/` |
| 输出格式 | `.fld` ASCII | `.h5` 二进制 |
| 入口 | `python run_icpc.py` | `python run.py CRYSTAL.step --op op.yaml` |
| 失败恢复 | 无 | `--resume` + manifest 状态机 |
| AEDT 模式 | GUI | 默认 headless |

`run_icpc.py` 作为 ICPC 原型保留在仓库,用于验证物理结果,不作为产品入口。
