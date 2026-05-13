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
    """关闭项目、关闭 desktop、释放 license。

    pyaedt 0.26+ 的 release_desktop 参数是 close_on_exit(不是 close_desktop)。
    """
    if desktop is None:
        return
    with contextlib.suppress(Exception):
        try:
            desktop.release_desktop(close_projects=save, close_on_exit=True)
        except TypeError:
            # 旧版本 fallback
            desktop.release_desktop(close_projects=save)


# ============================================================================
# 模型搭建
# ============================================================================


def import_step(maxwell, step_path: Path) -> list[str]:
    """
    导入 STEP 文件,返回导入后所有 object 名列表。

    实测 pyaedt 0.26 + AEDT 2025 R2 不保留 STEP PRODUCT 名(变成 ttpkp_attributeN
    这种内部 ID),所以下游 topology 模块会再用体积启发式重命名。
    """
    importer = getattr(maxwell.modeler, "import_3d_cad", None)
    if importer is None:
        raise RuntimeError(
            "pyaedt 中未找到 import_3d_cad;请检查 pyaedt 版本"
        )
    # pyaedt 0.26 默认 group_by_assembly=False ⇒ 每个零件独立 body(正是我们要的)
    importer(str(step_path))
    return list(maxwell.modeler.object_names)


def rename_bodies_by_heuristic(
    maxwell, log_fn=print
) -> list[str]:
    """
    STEP 导入后 AEDT 把零件名扔了,用体积启发式重命名:
      - 体积最大 → Crystal
      - 剩余中最小 → PPlus_01(点电极)
      - 剩余中其它 → NPlus_NN(按体积升序编号)

    仅在 ICPC 型 1 个 P+ + N 个 N+ 时正确;多 P+ 时这个启发式不可用,
    需要手动指定或恢复 STEP 名字。

    返回重命名后的 object 名列表(已按 Crystal / PPlus_* / NPlus_* 排序)。
    """
    obj_names = list(maxwell.modeler.object_names)
    if not obj_names:
        raise RuntimeError("STEP 导入后没有 body")

    # 已经命名正确就不动
    already_ok = (
        "Crystal" in obj_names
        and any(n.startswith("PPlus_") for n in obj_names)
        and any(n.startswith("NPlus_") for n in obj_names)
    )
    if already_ok:
        log_fn("STEP PRODUCT 名已被保留,无需重命名")
        return obj_names

    # 取体积
    vol_pairs = []
    for name in obj_names:
        try:
            vol = float(maxwell.modeler[name].volume)
        except Exception:
            vol = 0.0
        vol_pairs.append((name, vol))

    if len(vol_pairs) < 3:
        raise RuntimeError(
            f"STEP 至少要含 3 个 body(Crystal + 1 P+ + 1 N+),实际 {len(vol_pairs)}"
        )

    # 体积降序
    vol_pairs.sort(key=lambda t: -t[1])
    log_fn(f"按体积排序: {[(n, round(v, 2)) for n, v in vol_pairs]}")

    # 最大 = Crystal
    crystal_old = vol_pairs[0][0]
    _safe_rename(maxwell, crystal_old, "Crystal", log_fn)

    # 剩余:体积升序,最小是 P+(点电极),其它是 N+
    others = vol_pairs[1:]
    others.sort(key=lambda t: t[1])
    pplus_old = others[0][0]
    _safe_rename(maxwell, pplus_old, "PPlus_01", log_fn)

    nplus_idx = 1
    for old_name, _ in others[1:]:
        _safe_rename(maxwell, old_name, f"NPlus_{nplus_idx:02d}", log_fn)
        nplus_idx += 1

    return list(maxwell.modeler.object_names)


def _safe_rename(maxwell, old_name: str, new_name: str, log_fn) -> None:
    """把 body 重命名为 new_name。如果 new_name 已被占用就报错。"""
    if old_name == new_name:
        return
    existing = set(maxwell.modeler.object_names)
    if new_name in existing:
        raise RuntimeError(f"重命名冲突:目标名 {new_name!r} 已被占用")
    obj = maxwell.modeler[old_name]
    obj.name = new_name
    log_fn(f"  {old_name} -> {new_name}")


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

    pyaedt 0.26 没有高层 assign_charge_density;直接调 AEDT native 命令
    oboundary.AssignChargeDensity。表达式中 X/Y/Z 是 Global CS 下 SI 单位 (m)。

    参考 ICPC.aedt 里手工建的 'VolumeChargeDensity1' boundary 结构:
        BoundType='Volume Charge Density'
        Objects=[<crystal_name>]
        Value=<expression>
        CoordinateSystem='Global'
    """
    # 删除可能已存在的同名 BC
    for bc in list(maxwell.boundaries):
        if bc.name == RHO_BC_NAME:
            with contextlib.suppress(Exception):
                bc.delete()

    args = [
        f"NAME:{RHO_BC_NAME}",
        "Objects:=", [crystal_name],
        "Value:=", rho_expression,
        "CoordinateSystem:=", "Global",
    ]
    # AEDT native 方法名:在 2025 R2 中 Maxwell 3D 用 AssignVolumeChargeDensity。
    # 顺次尝试几个常见名字,选第一个能成功的。
    candidates = [
        "AssignVolumeChargeDensity",
        "AssignChargeDensity",
        "AssignVolumeCharge",
    ]
    last_err = None
    for method_name in candidates:
        method = getattr(maxwell.oboundary, method_name, None)
        if method is None:
            continue
        try:
            method(args)
            return
        except Exception as e:
            last_err = (method_name, e)
            continue
    raise RuntimeError(
        f"无法 assign Volume Charge Density。尝试过的方法名:{candidates};"
        f"最后失败:{last_err}"
    )


def setup_voltages(maxwell, voltage_map: dict[str, float]) -> None:
    """对每个电极 body 加 Voltage BC。PEC body 整体等势。

    pyaedt 0.26 的 assign_voltage 签名:
        assign_voltage(assignment, amplitude=1, name=None, ...)
    amplitude 默认单位 mV,所以要传 V 必须 ×1000 或者用字符串带单位。
    BC 名走 name=,不是 boundary=。
    """
    for electrode_name, v_volts in voltage_map.items():
        bc_name = f"V_{electrode_name}"
        # 删旧
        for bc in list(maxwell.boundaries):
            if bc.name == bc_name:
                with contextlib.suppress(Exception):
                    bc.delete()
        # 加新:amplitude 用 mV 单位,V → mV ×1000
        maxwell.assign_voltage(
            electrode_name,
            amplitude=v_volts * 1000.0,   # V → mV
            name=bc_name,
        )


def setup_matrix_solution(maxwell, electrode_names: list[str]) -> None:
    """
    注册一个 Matrix 求解,让 Maxwell 在 real_field setup 中输出 N×N 电容矩阵。

    pyaedt 0.26 的 assign_matrix(args: MaxwellMatrixSchema)。Electrostatic 用
    MatrixElectric dataclass。sources = Voltage BC 名(不是 body 名)。
    """
    from ansys.aedt.core.modules.boundary.maxwell_boundary import MatrixElectric

    # 删旧
    with contextlib.suppress(Exception):
        for m in list(maxwell.matrices):
            if m.name == MATRIX_NAME:
                m.delete()

    sources = [f"V_{n}" for n in electrode_names]
    matrix_args = MatrixElectric(
        signal_sources=sources,
        ground_sources=[],
        matrix_name=MATRIX_NAME,
    )
    maxwell.assign_matrix(matrix_args)


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
    setup_sweep: str = f"{SETUP_NAME} : LastAdaptive",
) -> dict[str, Path]:
    """
    导出 phi / E_x / E_y / E_z / E_mag 五个量到 .fld。

    pyaedt 0.26 的 export_field_file_on_grid 在 quantity="Phi" 上(silently)
    挂掉 → 回落到 CopyNamedExprToStack 而炸。绕开它,直接用 oFieldsReporter
    构造 calculator stack 然后 ExportOnGrid。

    quantity 类型 (Electrostatic):
      - "Phi"   标量势(直接 EnterQty)
      - "Mag_E" |E|(EnterQty E 后取 Mag)
      - "E_x" "E_y" "E_z" 分量(EnterQty E 后 ScalarX/Y/Z)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fcalc = maxwell.ofieldsreporter

    # 构造 ExportOnGrid 通用参数
    units = "mm"
    grid_start_wu = [f"{v}{units}" for v in grid.start]
    grid_stop_wu = [f"{v}{units}" for v in grid.stop]
    grid_step_wu = [f"{grid.step}{units}"] * 3
    grid_center = ["0mm", "0mm", "0mm"]
    export_options = [
        "NAME:ExportOption",
        "IncludePtInOutput:=", True,
        "RefCSName:=", "Global",
        "PtInSI:=", True,
        "FieldInRefCS:=", True,
    ]
    # solution 名:Setup1 : LastAdaptive
    # variation:对无 sweep 项目就空列表

    def _do_export(out_path: Path, stack_ops: list[tuple[str, ...]]) -> None:
        fcalc.CalcStack("clear")
        for op in stack_ops:
            method = getattr(fcalc, op[0])
            method(*op[1:])
        fcalc.ExportOnGrid(
            str(out_path),
            grid_start_wu, grid_stop_wu, grid_step_wu,
            setup_sweep, [], export_options,
            "Cartesian", grid_center, False,
        )

    paths: dict[str, Path] = {}

    # Maxwell 3D Electrostatic 的 NamedExpression 名(经探测确认):
    #   "Voltage"     → 电势 (V),即 README 里的 phi
    #   "Mag_E"       → |E|  (V/m)
    #   "<Ex,Ey,Ez>"  → E 向量,配 CalcOp("ScalarX/Y/Z") 取分量
    # EnterQty / "Phi" / "E" 都不可用。

    # 1. phi(用 Voltage)
    p = out_dir / "phi.fld"
    _do_export(p, [("CopyNamedExprToStack", "Voltage")])
    paths["phi"] = p

    # 2-4. E 分量
    for our_name, comp_op in [("E_x", "ScalarX"), ("E_y", "ScalarY"), ("E_z", "ScalarZ")]:
        p = out_dir / f"{our_name}.fld"
        _do_export(p, [("CopyNamedExprToStack", "<Ex,Ey,Ez>"), ("CalcOp", comp_op)])
        paths[our_name] = p

    # 5. |E|
    p = out_dir / "E_mag.fld"
    _do_export(p, [("CopyNamedExprToStack", "Mag_E")])
    paths["E_mag"] = p

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

    # pyaedt 0.26 的 duplicate_design 只接受 target name,源 = 当前 active design
    # 所以必须先确保 real_field 是 active
    if maxwell_real.design_name != REAL_FIELD_DESIGN:
        maxwell_real.set_active_design(REAL_FIELD_DESIGN)
    maxwell_real.duplicate_design(dst_design)

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
