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


def match_step_names_to_bodies(
    maxwell, step_path: Path, log_fn=print
) -> list[str]:
    """
    AEDT import_3d_cad 不保留 STEP 的 PRODUCT 名(已确认是 AEDT 行为,无 import
    选项可救)。这个函数做替代:

    1. 从 STEP 文本解析所有 PRODUCT 名,过滤到合法叶子名(Crystal/PPlus_NN/NPlus_NN)
    2. 校验数量:AEDT body 数 == STEP 叶子 PRODUCT 数
    3. 按"体积单调"启发式匹配:
       - 体积最大 AEDT body → Crystal
       - 剩余按体积升序:小的归 PPlus_*,大的归 NPlus_*
       - 编号按 STEP 里出现顺序(PPlus_01, PPlus_02, ...)
    4. 歧义 abort(例:多个 P+ 体积差 < 5% 时无法区分编号)

    这样名字真正来自 STEP(等于你 SolidWorks 里命的名),只是匹配靠体积。
    对单 P+ + 单 N+ 的 ICPC 总是无歧义;多电极对称分段需要后续真几何匹配。

    返回重命名后的 object 名列表。
    """
    from step_parser import extract_product_geometry, filter_leaf_products

    obj_names = list(maxwell.modeler.object_names)
    if not obj_names:
        raise RuntimeError("STEP 导入后没有 body")

    # ── 已被保留名字的情况(未来 AEDT 修了这个 bug)
    already_ok = (
        "Crystal" in obj_names
        and any(n.startswith("PPlus_") for n in obj_names)
        and any(n.startswith("NPlus_") for n in obj_names)
    )
    if already_ok:
        log_fn("AEDT 保留了 STEP PRODUCT 名,无需重命名")
        return obj_names

    # ── 从 STEP 取期望的名字列表
    log_fn(f"AEDT 丢失了 STEP 名字(变成 {obj_names[:1]} ...),从 STEP 文本恢复")
    step_products = filter_leaf_products(extract_product_geometry(step_path))
    expected_names = sorted(step_products.keys())
    expected_crystal = [n for n in expected_names if n == "Crystal"]
    expected_pplus = sorted(n for n in expected_names if n.startswith("PPlus_"))
    expected_nplus = sorted(n for n in expected_names if n.startswith("NPlus_"))

    log_fn(f"STEP 中的叶子 PRODUCT: Crystal={expected_crystal}, "
           f"PPlus={expected_pplus}, NPlus={expected_nplus}")

    if not expected_crystal:
        raise RuntimeError(
            "STEP 文件里没有名为 'Crystal' 的 PRODUCT。"
            "请在 SolidWorks 里把晶体本体零件命名为 'Crystal'(ASCII,无前后缀)。"
        )
    if not (expected_pplus or expected_nplus):
        raise RuntimeError(
            "STEP 文件里没有 PPlus_NN 或 NPlus_NN 模式的电极零件。"
        )
    if len(obj_names) != 1 + len(expected_pplus) + len(expected_nplus):
        raise RuntimeError(
            f"AEDT 导入了 {len(obj_names)} 个 body,但 STEP 里只有 "
            f"{1 + len(expected_pplus) + len(expected_nplus)} 个叶子 PRODUCT。"
            f"可能 SolidWorks 装配体里有未命名的额外零件。"
        )

    # ── 按体积排序
    vol_pairs = []
    for name in obj_names:
        try:
            vol = float(maxwell.modeler[name].volume)
        except Exception:
            vol = 0.0
        vol_pairs.append((name, vol))
    vol_pairs.sort(key=lambda t: -t[1])  # 大→小
    log_fn(f"AEDT body 体积排序: {[(n, round(v, 2)) for n, v in vol_pairs]}")

    # ── 匹配
    # 体积最大 = Crystal
    crystal_old = vol_pairs[0][0]
    _safe_rename(maxwell, crystal_old, "Crystal", log_fn)

    # 剩余:升序 = (P+ 集合) + (N+ 集合)
    # 假设:|PPlus| 平均比 |NPlus| 小(点电极物理特征)
    # 取最小 len(expected_pplus) 个归 P+,其余归 N+
    rest = sorted(vol_pairs[1:], key=lambda t: t[1])  # 小→大
    pplus_aedt = rest[: len(expected_pplus)]
    nplus_aedt = rest[len(expected_pplus):]

    # 歧义检查:P+ 和 N+ 体积带相邻时不可靠
    if pplus_aedt and nplus_aedt:
        max_pplus_vol = pplus_aedt[-1][1]
        min_nplus_vol = nplus_aedt[0][1]
        if max_pplus_vol > 0.5 * min_nplus_vol:
            raise RuntimeError(
                f"无法用体积启发式区分 P+ 和 N+:最大 P+ 体积 = {max_pplus_vol:.2f}, "
                f"最小 N+ 体积 = {min_nplus_vol:.2f},比值 > 50%。"
                "请联系开发者实现真几何匹配,或在 SolidWorks 里调整命名/形状。"
            )

    # 同类内部歧义:P+ 之间体积近似时无法编号
    for label, group, names in [
        ("PPlus", pplus_aedt, expected_pplus),
        ("NPlus", nplus_aedt, expected_nplus),
    ]:
        if len(group) > 1:
            vols = [v for _, v in group]
            spread = (max(vols) - min(vols)) / max(vols)
            if spread < 0.05:
                raise RuntimeError(
                    f"多个 {label} 电极体积差 < 5%(从 {min(vols):.2f} 到 {max(vols):.2f}),"
                    f"无法用体积启发式区分 {label}_01/02/...。"
                    "请用真几何匹配(待实现),或目前用单 {label} 测试。"
                )
        # 按 STEP 排序赋名:体积升序对应名字编号升序
        for (aedt_name, _vol), expected_name in zip(group, names):
            _safe_rename(maxwell, aedt_name, expected_name, log_fn)

    return list(maxwell.modeler.object_names)


# 向后兼容别名(老代码可能还在调)
rename_bodies_by_heuristic = match_step_names_to_bodies


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

    AEDT 2025 R2 + pyaedt 0.26 的实际表达式名(经探测):
        Matrix1.CplCoef(<src_i>, <src_j>)  ← 不是 C(...)
    sources 名 = Voltage BC 名 = "V_<electrode>"。
    单位:Maxwell 默认 F,转 pF。
    """
    sources = [f"V_{n}" for n in electrode_names]
    N = len(electrode_names)
    mat = np.zeros((N, N), dtype=np.float64)

    # 实测发现:一次性把 N² 个 expressions 传给 get_solution_data 时返回的
    # SolutionData 对部分 expr 找不到值。一个一个查更稳。
    for i, si in enumerate(sources):
        for j, sj in enumerate(sources):
            expr = f"{MATRIX_NAME}.C({si},{sj})"
            try:
                sol = maxwell.post.get_solution_data(
                    expressions=[expr],
                    setup_sweep_name=setup_sweep_name,
                    report_category="Matrix",
                )
                if sol is None or isinstance(sol, bool):
                    mat[i, j] = float("nan")
                    continue
                vals = sol.data_real(expr)
                c_raw = float(vals[-1]) if hasattr(vals, "__iter__") else float(vals)
            except Exception:
                c_raw = float("nan")
            # AEDT 2025 R2 的 Matrix1.C 返回值,经物理量级反推应是 aF (1e-18 F)。
            # 对于 ~10mm 间距的小 Ge 电极:期望 ~10-100 pF,实测 raw ≈ 4e7;
            # 4e7 aF = 4e7 × 1e-18 F = 4e-11 F = 40 pF ✓ 物理合理。
            # README 约定 pF,所以 aF → pF 除以 1e6 (1 pF = 1e6 aF)。
            # TODO(verify): 用一个已知电容的标定测试在真 ICPC 上确认。
            mat[i, j] = c_raw / 1.0e6
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
