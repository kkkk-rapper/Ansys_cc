"""
Maxwell 3D 操作封装。所有 pyaedt 调用集中在这里。

依赖 ansys-aedt-core (pyaedt) >= 0.10。

约定:
- 所有几何/物理 setup 都在 source design (real_field) 上做
- 权重势 design 通过 duplicate_design 克隆,再修改 BC
- mesh 共享:duplicate_design 自动克隆 mesh;权重势 design 把 max_passes 设为 1

不确定的 pyaedt API 用 # TODO(verify) 标注。首次真跑时如果某个调用失败,通常
是 pyaedt 版本差异;按报错信息小范围调整即可。
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spec import CrystalSpec, OperatingSpec, build_rho_expression
from topology import CrystalGeometry

REAL_FIELD_DESIGN = "real_field"
SETUP_NAME = "Setup1"
MATRIX_NAME = "Matrix1"
RHO_BC_NAME = "BulkSpaceCharge"


@dataclass
class GridSpec:
    """场导出的笛卡尔网格(单位 mm,转字符串给 pyaedt 时附 "mm")。"""
    start: np.ndarray   # [xmin, ymin, zmin] mm
    stop: np.ndarray    # [xmax, ymax, zmax] mm
    step: float          # 各向同性步长 mm

    @property
    def shape(self) -> tuple[int, int, int]:
        diff = self.stop - self.start
        return tuple(int(np.round(d / self.step)) + 1 for d in diff)

    def aedt_args(self) -> dict[str, list[str]]:
        return {
            "grid_start": [f"{v}mm" for v in self.start],
            "grid_stop":  [f"{v}mm" for v in self.stop],
            "grid_step":  [f"{self.step}mm"] * 3,
        }


def build_grid_from_geometry(
    geom: CrystalGeometry,
    step_mm: float,
    padding_mm: float,
) -> GridSpec:
    """从 Crystal bbox + 余量 + 步长构造网格。"""
    pad = np.array([padding_mm] * 3)
    start = geom.bbox_min_mm - pad
    stop = geom.bbox_max_mm + pad
    return GridSpec(start=start, stop=stop, step=step_mm)


# ============================================================================
# Session 生命周期
# ============================================================================


def start_aedt(
    aedt_version: str,
    non_graphical: bool,
    project_path: Path,
    new_desktop: bool = True,
):
    """
    启动 AEDT 新 session,创建空项目。

    返回 (desktop, maxwell)。两者都需要在结束时显式释放。

    TODO(verify): pyaedt 0.10 中 Desktop 与 Maxwell3d 的初始化签名稳定;
                  new_desktop=True 强制新建,避免污染已有 session。
    """
    from ansys.aedt.core import Desktop, Maxwell3d

    desktop = Desktop(
        version=aedt_version,
        non_graphical=non_graphical,
        new_desktop=new_desktop,
    )
    maxwell = Maxwell3d(
        project=str(project_path),
        design=REAL_FIELD_DESIGN,
        solution_type="Electrostatic",
        version=aedt_version,
    )
    return desktop, maxwell


def close_aedt(desktop, save: bool = True) -> None:
    """关闭项目、关闭 desktop、释放 license。"""
    if desktop is None:
        return
    with contextlib.suppress(Exception):
        desktop.release_desktop(close_projects=save, close_desktop=True)


# ============================================================================
# 模型搭建
# ============================================================================


def import_step(maxwell, step_path: Path) -> list[str]:
    """
    导入 STEP 文件,返回导入后所有 object 名列表。

    TODO(verify): pyaedt 0.10 中 import_3d_cad 是通用 3D CAD 导入入口,
                  支持 .step/.stp/.iges/.sat 等。也可能叫 import_step。
    """
    importer = getattr(maxwell.modeler, "import_3d_cad", None) or \
               getattr(maxwell.modeler, "import_step", None)
    if importer is None:
        raise RuntimeError(
            "pyaedt 中未找到 import_3d_cad / import_step 方法;请检查 pyaedt 版本"
        )
    importer(str(step_path))
    return list(maxwell.modeler.object_names)


def setup_materials(maxwell, geom: CrystalGeometry, crystal_material: str) -> None:
    """Crystal 材料 = 用户指定(默认 Germanium);所有电极 = PEC。"""
    crystal_obj = maxwell.modeler[geom.crystal_name]
    crystal_obj.material_name = crystal_material
    for name in geom.pplus_names + geom.nplus_names:
        maxwell.modeler[name].material_name = "pec"


def setup_charge_density(
    maxwell,
    crystal_name: str,
    rho_expression: str,
) -> None:
    """
    给 Crystal 设位置依赖的 Volume Charge Density。

    TODO(verify): pyaedt 0.10 中 Maxwell3d 应有 assign_charge_density 或类似方法
                  支持 ChargeDensity 表达式形式的 Bulk Source。若 API 名不同,
                  在此处替换即可。表达式中 X/Y/Z 是 SI 单位(米)。
    """
    # 删除可能已存在的同名 BC
    for bc in list(maxwell.boundaries):
        if bc.name == RHO_BC_NAME:
            with contextlib.suppress(Exception):
                bc.delete()

    # 优先用 pyaedt 高层 API;若不存在,fallback 到 native commands
    assign_fn = (
        getattr(maxwell, "assign_charge_density", None)
        or getattr(maxwell, "assign_voltage_drop", None)
    )
    if assign_fn is not None:
        try:
            assign_fn(crystal_name, rho_expression, name=RHO_BC_NAME)
            return
        except TypeError:
            # 签名可能是 (assignment, value)
            assign_fn(crystal_name, rho_expression)
            return

    # Fallback:直接调底层 odesign("AssignChargeDensity") —— 仅在高层 API 不可用时
    raise RuntimeError(
        "pyaedt 中未找到 assign_charge_density 方法。可能需要使用 "
        "maxwell.odesign.AssignVolumeChargeDensity(...) native 命令,"
        "请按当前 pyaedt 版本调整。"
    )


def setup_voltages(maxwell, voltage_map: dict[str, float]) -> None:
    """对每个电极 body 加 Voltage BC。PEC body 整体等势。"""
    for name, v in voltage_map.items():
        bc_name = f"V_{name}"
        # 删旧
        for bc in list(maxwell.boundaries):
            if bc.name == bc_name:
                with contextlib.suppress(Exception):
                    bc.delete()
        # 加新
        maxwell.assign_voltage(name, voltage=f"{v}V", boundary=bc_name)


def setup_matrix_solution(maxwell, electrode_names: list[str]) -> None:
    """
    注册一个 Matrix 求解,让 Maxwell 在 real_field setup 中输出 N×N 电容矩阵。

    每个 PEC 电极对应一个 conductor;Matrix 自动给出对称矩阵。

    TODO(verify): pyaedt 0.10 中 Maxwell3d.assign_matrix 接受 sources 列表(conductor 名)。
                  名字可能要带 "V_" 前缀(即对应的 Voltage BC 名),而不是 body 名。
                  下方实现两种都尝试。
    """
    # 删旧
    with contextlib.suppress(Exception):
        for m in list(maxwell.matrices):
            if m.name == MATRIX_NAME:
                m.delete()

    if not hasattr(maxwell, "assign_matrix"):
        raise RuntimeError(
            "pyaedt 中未找到 assign_matrix。请按当前版本调整 Matrix 求解配置。"
        )

    # 用 Voltage BC 名作为 source(常见 Maxwell 习惯)
    sources = [f"V_{n}" for n in electrode_names]
    try:
        maxwell.assign_matrix(sources, matrix_name=MATRIX_NAME)
    except Exception:
        # 退而求其次,直接传 body 名
        maxwell.assign_matrix(electrode_names, matrix_name=MATRIX_NAME)


def setup_adaptive(
    maxwell,
    max_passes: int = 10,
    percent_error: float = 1.0,
    percent_refinement: int = 30,
) -> None:
    """real_field 的自适应求解 setup。"""
    # 删旧
    for s in list(maxwell.setups):
        if s.name == SETUP_NAME:
            with contextlib.suppress(Exception):
                s.delete()

    setup = maxwell.create_setup(name=SETUP_NAME)
    setup.props["MaximumPasses"] = max_passes
    setup.props["PercentError"] = percent_error
    setup.props["PercentRefinement"] = percent_refinement
    setup.update()


# ============================================================================
# 求解 + 字段导出
# ============================================================================


def solve(maxwell, setup_name: str = SETUP_NAME) -> None:
    """阻塞求解;失败抛异常。"""
    maxwell.analyze_setup(setup_name)


def export_field_files(
    maxwell,
    out_dir: Path,
    crystal_name: str,
    grid: GridSpec,
) -> dict[str, Path]:
    """
    导出 phi / E_x / E_y / E_z / E_mag 五个量到 .fld。返回 {quantity: path}。

    AEDT 的 quantity 名:
      - "Phi"       → 电势 φ        (V)
      - "Mag_E"     → |E|           (V/m)
      - "E_x" "E_y" "E_z" → 分量    (V/m)

    我们在文件名上用规范化的 quantity 名 (phi, E_x, ...) 以便对应 .h5 schema。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    aedt_args = grid.aedt_args()
    paths: dict[str, Path] = {}
    quantity_map = [
        ("phi",   "Phi"),
        ("E_x",   "E_x"),
        ("E_y",   "E_y"),
        ("E_z",   "E_z"),
        ("E_mag", "Mag_E"),
    ]
    for our_name, aedt_name in quantity_map:
        out_path = out_dir / f"{our_name}.fld"
        maxwell.post.export_field_file(
            quantity=aedt_name,
            output_file=str(out_path),
            grid_type="Cartesian",
            assignment=crystal_name,
            **aedt_args,
        )
        paths[our_name] = out_path
    return paths


def extract_capacitance_matrix(
    maxwell,
    electrode_names: list[str],
    setup_sweep_name: str = f"{SETUP_NAME} : LastAdaptive",
) -> np.ndarray:
    """
    读取 Maxwell Matrix solution,返回 N×N 电容矩阵(单位 pF)。

    Maxwell 提供的表达式名:C(<src>, <src>) 自电容,C(<src_i>, <src_j>) 互电容。
    我们的 sources 名 = "V_<electrode>"。

    TODO(verify): pyaedt 0.10 get_solution_data 的接口稳定;
                  report_category 名字 ("Matrix" 还是 "Matrix1") 可能因版本不同。
    """
    sources = [f"V_{n}" for n in electrode_names]
    expressions = []
    for i, si in enumerate(sources):
        for j, sj in enumerate(sources):
            expressions.append(f"C({si},{sj})")

    sol = maxwell.post.get_solution_data(
        expressions=expressions,
        setup_sweep_name=setup_sweep_name,
        report_category="Matrix",
    )
    if sol is None:
        raise RuntimeError(
            "Maxwell Matrix 求解结果为空。检查 setup_matrix_solution 是否成功"
            "以及 sources 名字是否匹配。"
        )

    N = len(electrode_names)
    mat = np.zeros((N, N), dtype=np.float64)
    for i, si in enumerate(sources):
        for j, sj in enumerate(sources):
            expr = f"C({si},{sj})"
            try:
                vals = sol.data_real(expr)
                # 取最后一个 adaptive pass 的值
                c_farads = float(vals[-1]) if hasattr(vals, "__iter__") else float(vals)
            except Exception:
                c_farads = float("nan")
            # F → pF
            mat[i, j] = c_farads * 1e12
    return mat


# ============================================================================
# 权重势设计:克隆 + 改 BC + 单 pass 解
# ============================================================================


def setup_weighting_design(
    maxwell_real,
    target_electrode: str,
    all_electrodes: list[str],
) -> Any:
    """
    从 real_field 克隆出 wpot_<target> 设计,
    删除 ρ BC,改电压(target=1V, 其余=0V),max_passes=1。

    返回新设计的 Maxwell3d handle。

    几何不变 ⇒ duplicate_design 会带上 mesh;max_passes=1 ⇒ 不再自适应,
    直接在继承的 mesh 上做一次求解。
    """
    from ansys.aedt.core import Maxwell3d

    dst_design = f"wpot_{target_electrode}"

    # 删除旧的同名设计(如果有)
    with contextlib.suppress(Exception):
        if dst_design in maxwell_real.design_list:
            maxwell_real.delete_design(dst_design)

    maxwell_real.duplicate_design(REAL_FIELD_DESIGN, dst_design)

    m_w = Maxwell3d(design=dst_design)

    # 删 ρ BC
    for bc in list(m_w.boundaries):
        if bc.name == RHO_BC_NAME:
            with contextlib.suppress(Exception):
                bc.delete()

    # 改电压
    voltage_map = {n: (1.0 if n == target_electrode else 0.0) for n in all_electrodes}
    setup_voltages(m_w, voltage_map)

    # 单 pass(继承 real_field 的 mesh)
    setup = next((s for s in m_w.setups if s.name == SETUP_NAME), None)
    if setup is None:
        raise RuntimeError(f"权重势设计 {dst_design} 中找不到 setup {SETUP_NAME}")
    setup.props["MaximumPasses"] = 1
    setup.update()

    return m_w
