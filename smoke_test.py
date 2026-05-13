"""Smoke test: 开 AEDT、开 ICPC.aedt、列设计/边界。不解算。"""
from pathlib import Path
from ansys.aedt.core import Maxwell3d, Desktop

HERE = Path(__file__).resolve().parent
PROJECT_PATH = HERE / "ICPC.aedt"

print("启动 AEDT 2025 R2 ...")
desk = Desktop(version="2025.2", non_graphical=False, new_desktop=True)
print("  Desktop OK, version =", desk.aedt_version_id)

print("打开工程：", PROJECT_PATH)
m = Maxwell3d(project=str(PROJECT_PATH), design="Maxwell3DDesign1")
print("  设计名:", m.design_name)
print("  求解器:", m.solution_type)
print("  对象列表:", m.modeler.object_names)
print("  边界:")
for b in m.boundaries:
    try:
        print(f"    - {b.name}  type={b.type}  props={dict(b.props)}")
    except Exception as ex:
        print(f"    - {b.name}  (读 props 失败: {ex})")
print("  Setup 列表:", [s.name for s in m.setups])
print("OK. 关闭 AEDT ...")
desk.release_desktop(close_projects=True, close_desktop=True)
