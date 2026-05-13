"""
ICPC HPGe 静电场全套求解（基于已有 ICPC.aedt）
- 真实电场（带空间电荷，HV=2.5kV）
- 权重势 / 权重场（点电极读出 & HV 端读出，无空间电荷）
- 电容矩阵
所有输出都放在脚本所在目录的 ICPC_out/ 子目录里。
"""
from pathlib import Path
from ansys.aedt.core import Maxwell3d, Desktop

HERE = Path(__file__).resolve().parent
PROJECT_PATH = HERE / "ICPC.aedt"
OUT_DIR      = HERE / "ICPC_out"
OUT_DIR.mkdir(exist_ok=True)

AEDT_VERSION  = "2025.2"
NON_GRAPHICAL = False

SRC_DESIGN   = "Maxwell3DDesign1"
WP_PC_DESIGN = "WP_PointContact"
WP_HV_DESIGN = "WP_HV"

BND_GROUND = "ground"
BND_HV     = "HV"
BND_RHO    = "VolumeChargeDensity1"

GRID_START = ["-37mm", "-37mm", "0mm"]
GRID_STOP  = [ "37mm",  "37mm", "65mm"]
GRID_STEP  = ["0.5mm", "0.5mm", "0.5mm"]


def export_grid_fields(m, prefix: str):
    sub = OUT_DIR / prefix
    sub.mkdir(exist_ok=True)
    for q, fname in [("Phi",   "phi.fld"),
                     ("Mag_E", "E_mag.fld"),
                     ("E_x",   "Ex.fld"),
                     ("E_y",   "Ey.fld"),
                     ("E_z",   "Ez.fld")]:
        m.post.export_field_file(
            quantity=q,
            output_file=str(sub / fname),
            grid_type="Cartesian",
            grid_start=GRID_START,
            grid_stop=GRID_STOP,
            grid_step=GRID_STEP,
            assignment="Cylinder1",
        )
    print(f"  导出 → {sub}")


def set_voltage(m, bnd_name: str, value: str):
    for b in m.boundaries:
        if b.name == bnd_name:
            b.props["Voltage"] = value
            b.update()
            return
    raise RuntimeError(f"找不到边界 {bnd_name}")


def delete_boundary(m, bnd_name: str):
    for b in m.boundaries:
        if b.name == bnd_name:
            b.delete()
            return True
    return False


def main():
    desk = Desktop(version=AEDT_VERSION,
                   non_graphical=NON_GRAPHICAL,
                   new_desktop=False)

    # 1. 真实电场
    m_real = Maxwell3d(project=str(PROJECT_PATH), design=SRC_DESIGN)
    print("[1/3] 真实电场（带空间电荷，HV=2.5kV）...")
    m_real.analyze_setup("Setup1")
    export_grid_fields(m_real, "real_field")

    print("    电容矩阵：")
    sol = m_real.post.get_solution_data(
        expressions=["C(ground,ground)", "C(HV,HV)", "C(ground,HV)"],
        setup_sweep_name="Setup1 : LastAdaptive",
        report_category="Matrix",
    )
    if sol is not None:
        for e in ["C(ground,ground)", "C(HV,HV)", "C(ground,HV)"]:
            try:
                print(f"      {e} = {sol.data_real(e)}")
            except Exception as ex:
                print(f"      {e} 读取失败: {ex}")

    # 2. 点电极权重势
    print("[2/3] 点电极权重势...")
    m_real.duplicate_design(SRC_DESIGN, WP_PC_DESIGN)
    m_pc = Maxwell3d(design=WP_PC_DESIGN)
    delete_boundary(m_pc, BND_RHO)
    set_voltage(m_pc, BND_GROUND, "1V")
    set_voltage(m_pc, BND_HV,     "0V")
    m_pc.analyze_setup("Setup1")
    export_grid_fields(m_pc, "wpot_PC")

    # 3. HV 端权重势（自检用：WP_PC + WP_HV ≡ 1）
    print("[3/3] HV 端权重势...")
    m_real.duplicate_design(SRC_DESIGN, WP_HV_DESIGN)
    m_hv = Maxwell3d(design=WP_HV_DESIGN)
    delete_boundary(m_hv, BND_RHO)
    set_voltage(m_hv, BND_GROUND, "0V")
    set_voltage(m_hv, BND_HV,     "1V")
    m_hv.analyze_setup("Setup1")
    export_grid_fields(m_hv, "wpot_HV")

    m_real.save_project()
    desk.release_desktop(close_projects=False, close_desktop=False)
    print("完成。输出 →", OUT_DIR)


if __name__ == "__main__":
    main()
