"""
HPGe 自动化仿真主入口。

用法 (详见 README.md §3):
    python run.py CRYSTAL.step --op operating.yaml [--force | --resume] [...]

完整规范 = README.md。本脚本仅按规范实现编排,不引入新的设计决策。
"""
from __future__ import annotations

import argparse
import logging
import re
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from spec import (
    CrystalSpec,
    OperatingSpec,
    SpecError,
    build_rho_expression,
)
import aedt_ops
import io_ops
from topology import detect_geometry, CrystalGeometry


# ============================================================================
# CLI
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="HPGe 探测器自动化静电仿真(STEP → Maxwell → .h5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="完整契约见 README.md。",
    )
    p.add_argument("step", type=Path, help="SolidWorks 导出的 STEP AP242 文件")
    p.add_argument("--op", type=Path, default=None, dest="op_path",
                   help="operating.yaml 路径")
    p.add_argument("--crystal", type=Path, default=None, dest="crystal_path",
                   help="crystal.yaml 路径(默认: <step>.yaml 同目录同名)")
    p.add_argument("--voltage", action="append", default=[], metavar="NAME=V",
                   help="单个电极电压覆盖,可多次")
    p.add_argument("--grid-step", type=float, default=None, metavar="MM",
                   help="覆盖 operating.yaml 中的网格步长 (mm)")
    p.add_argument("--output-root", type=Path, default=None, metavar="PATH",
                   help="输出根目录(默认: ./data_out 或 operating.yaml 中指定)")
    p.add_argument("--run-name", type=str, default=None, metavar="NAME",
                   help="子目录名覆盖(纯 CLI 电压模式下必填)")
    p.add_argument("--force", action="store_true",
                   help="允许覆盖已存在的输出目录")
    p.add_argument("--resume", action="store_true",
                   help="从 partial 状态恢复,跳过已完成的 design")
    p.add_argument("--gui", action="store_true",
                   help="显示 AEDT GUI(默认 headless)")
    p.add_argument("--reuse-aedt", action="store_true",
                   help="复用已存在的 .aedt 项目(STEP hash 必须匹配)")
    p.add_argument("--cleanup-aedt", action="store_true",
                   help="收尾时删除中间权重势 designs")
    p.add_argument("--aedt-version", default="2025.2",
                   help="AEDT 版本(默认 2025.2)")
    p.add_argument("--max-passes", type=int, default=10,
                   help="real_field 自适应最大 pass 数(默认 10)")
    p.add_argument("--percent-error", type=float, default=1.0,
                   help="real_field 自适应目标误差 %% (默认 1.0)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def parse_voltage_overrides(items: list[str]) -> dict[str, float]:
    """解析 --voltage NAME=V 列表 → {name: value}。"""
    out: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise SpecError(f"--voltage 格式错误: {item!r}(应为 NAME=V)")
        name, val = item.split("=", 1)
        name = name.strip()
        try:
            out[name] = float(val.strip())
        except ValueError:
            raise SpecError(f"--voltage {name}=...:电压必须是数字,实际 = {val!r}")
    return out


# ============================================================================
# 日志
# ============================================================================


def setup_logging(level: str, run_dir: Path | None) -> logging.Logger:
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(run_dir / "run.log", encoding="utf-8"))
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
    return logging.getLogger("run")


# ============================================================================
# Pre-flight 校验(README §5.1)
# ============================================================================


def preflight_validate(args, log: logging.Logger) -> tuple[CrystalSpec, OperatingSpec, Path, Path]:
    """
    Maxwell 启动前的校验。返回 (crystal_spec, operating_spec, run_dir, project_path)。
    任一失败抛 SpecError,主程序捕获后 abort。
    """
    # 1. STEP 存在
    if not args.step.is_file():
        raise SpecError(f"STEP 文件不存在: {args.step}")
    if args.step.suffix.lower() not in (".step", ".stp"):
        raise SpecError(f"STEP 扩展名应为 .step 或 .stp,实际: {args.step.suffix}")

    # 2. crystal.yaml
    crystal_path = args.crystal_path or args.step.with_suffix(".yaml")
    crystal_spec = CrystalSpec.from_yaml(crystal_path)

    # 3. operating.yaml(或纯 CLI)
    cli_overrides = parse_voltage_overrides(args.voltage)
    if args.op_path is not None:
        operating_spec = OperatingSpec.from_yaml(
            args.op_path,
            cli_overrides=cli_overrides,
            cli_grid_step_mm=args.grid_step,
            cli_output_root=args.output_root,
        )
        op_stem = args.op_path.stem
    else:
        if not cli_overrides:
            raise SpecError(
                "未提供 --op,且未提供 --voltage 覆盖。"
                "请二选一(完整运行参数走 yaml,临时覆盖走 CLI)。"
            )
        if not args.run_name:
            raise SpecError(
                "纯 --voltage CLI 模式必须提供 --run-name <NAME>,"
                "否则无法构造输出子目录名。"
            )
        # 构造一个最小 OperatingSpec(只走 cli_overrides)
        operating_spec = OperatingSpec(
            voltages_explicit={},
            defaults={},
            grid_step_mm=args.grid_step or 0.5,
            grid_padding_mm=0.5,
            output_root=args.output_root or Path("./data_out"),
            source_path=None,
            cli_overrides=cli_overrides,
            raw={},
        )
        op_stem = None

    # 4. 输出目录
    run_dir = io_ops.compute_run_dir(
        operating_spec.output_root,
        step_stem=args.step.stem,
        op_stem=op_stem,
        run_name_override=args.run_name,
    )
    log.info(f"输出目录: {run_dir}")

    if run_dir.exists():
        manifest_path = run_dir / io_ops.MANIFEST_FILENAME
        if args.resume:
            if not manifest_path.is_file():
                raise SpecError(
                    f"--resume 指定,但找不到 manifest.yaml: {manifest_path}"
                )
            existing = io_ops.Manifest.load(manifest_path)
            if existing.status == "success":
                raise SpecError(
                    f"输出目录已是 success 状态,无需 resume: {run_dir}"
                )
            if existing.status == "failed":
                raise SpecError(
                    "输出目录处于 failed 状态,--resume 不接受。"
                    "请用 --force 重跑(会清空目录)。"
                )
            log.info(f"--resume:已完成 design = {existing.completed_designs}")
        elif args.force:
            log.warning(f"--force:清空已存在的输出目录 {run_dir}")
            shutil.rmtree(run_dir)
        else:
            raise SpecError(
                f"输出目录已存在: {run_dir}\n"
                f"加 --force 覆盖,或加 --resume 从断点续跑。"
            )

    # 5. 项目文件路径
    project_path = run_dir / f"{args.step.stem}.aedt"

    # 6. 磁盘空间(粗估;真实 grid_shape 要等 AEDT 拿 bbox 后才知)
    #    这里跳过精确检查,只在 AEDT 启动后再做一次。

    log.info(f"Pre-flight OK: crystal={crystal_spec.source_path}, "
             f"op={'-' if operating_spec.source_path is None else operating_spec.source_path}, "
             f"step={args.step}")
    return crystal_spec, operating_spec, run_dir, project_path


# ============================================================================
# Signal handler
# ============================================================================


_current_manifest: io_ops.Manifest | None = None
_current_manifest_path: Path | None = None
_current_desktop = None


def _signal_handler(signum, frame):
    log = logging.getLogger("run")
    log.warning(f"收到信号 {signum},尝试优雅关闭...")
    if _current_manifest is not None and _current_manifest_path is not None:
        _current_manifest.status = "partial"
        _current_manifest.add_error(f"中断 (signal {signum})")
        try:
            _current_manifest.write(_current_manifest_path)
            log.warning(f"manifest 状态已写为 partial: {_current_manifest_path}")
        except Exception as e:
            log.error(f"写 manifest 失败: {e}")
    aedt_ops.close_aedt(_current_desktop, save=True)
    sys.exit(130)


def install_signal_handler() -> None:
    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)


# ============================================================================
# 主流水线
# ============================================================================


def main(argv: list[str] | None = None) -> int:
    global _current_manifest, _current_manifest_path, _current_desktop

    args = build_parser().parse_args(argv)
    log = setup_logging(args.log_level, run_dir=None)

    try:
        crystal_spec, operating_spec, run_dir, project_path = preflight_validate(args, log)
    except SpecError as e:
        log.error(f"Pre-flight 失败: {e}")
        return 2

    log = setup_logging(args.log_level, run_dir=run_dir)
    log.info("====== HPGe 仿真开始 ======")

    install_signal_handler()

    # 准备 manifest
    timestamp = io_ops.now_iso()
    run_id = run_dir.name
    manifest = io_ops.Manifest(
        run_id=run_id,
        timestamp=timestamp,
        status="running",
        aedt_version=args.aedt_version,
        inputs={
            "step": {
                "path": str(args.step.resolve()),
                "sha256": io_ops.sha256_of_file(args.step),
            },
            "crystal_yaml": crystal_spec.raw,
            "operating_yaml": operating_spec.raw,
            "cli_voltage_overrides": dict(operating_spec.cli_overrides),
        },
    )
    manifest_path = run_dir / io_ops.MANIFEST_FILENAME
    _current_manifest = manifest
    _current_manifest_path = manifest_path

    # --resume:加载已有 manifest 的 completed_designs
    completed = set()
    if args.resume and manifest_path.is_file():
        old = io_ops.Manifest.load(manifest_path)
        completed = set(old.completed_designs)
        manifest.completed_designs = list(completed)
        log.info(f"resume:跳过已完成 design = {sorted(completed)}")

    manifest.write(manifest_path)

    # ─────────────────────────────────────────────────────────────
    # AEDT 流程
    # ─────────────────────────────────────────────────────────────
    desktop = None
    try:
        log.info(f"启动 AEDT {args.aedt_version} (headless={not args.gui})...")
        desktop, maxwell_real = aedt_ops.start_aedt(
            aedt_version=args.aedt_version,
            non_graphical=(not args.gui),
            project_path=project_path,
            new_desktop=True,
        )
        _current_desktop = desktop

        # 导入 STEP
        log.info(f"导入 STEP: {args.step}")
        all_obj_names = aedt_ops.import_step(maxwell_real, args.step.resolve())
        log.info(f"导入完成,objects = {all_obj_names}")

        # 拓扑识别
        log.info("拓扑识别中...")
        geom = detect_geometry(maxwell_real)
        electrode_names = geom.pplus_names + geom.nplus_names
        log.info(
            f"Crystal vol = {geom.crystal_volume_mm3:.1f} mm³, "
            f"electrodes = {electrode_names}"
        )
        log.info(f"P+ 端面质心 = {geom.pplus_face_centroid_mm.tolist()} mm")
        log.info(f"对面端面质心 = {geom.opposite_face_centroid_mm.tolist()} mm")

        manifest.geometry_detected = {
            "crystal_volume_mm3": geom.crystal_volume_mm3,
            "electrodes": electrode_names,
            "pplus_face_centroid_mm": geom.pplus_face_centroid_mm.tolist(),
            "opposite_face_centroid_mm": geom.opposite_face_centroid_mm.tolist(),
            "bbox_mm": {
                "min": geom.bbox_min_mm.tolist(),
                "max": geom.bbox_max_mm.tolist(),
            },
        }
        manifest.write(manifest_path)

        # 电压 resolve(覆盖 + 默认 + CLI)
        voltages = operating_spec.resolve_voltages(electrode_names)
        log.info(f"resolved voltages: {voltages}")

        # 网格
        grid = aedt_ops.build_grid_from_geometry(
            geom, operating_spec.grid_step_mm, operating_spec.grid_padding_mm,
        )
        log.info(f"导出网格: shape={grid.shape}, step={grid.step}mm")
        manifest.geometry_detected["grid_shape"] = list(grid.shape)

        # 物理建模
        log.info("设材料(Crystal + PEC 电极)...")
        aedt_ops.setup_materials(maxwell_real, geom, crystal_spec.crystal_material)

        log.info("设 ρ Volume Charge Density(线性梯度表达式)...")
        rho_expr = build_rho_expression(
            geom.pplus_face_centroid_mm,
            geom.opposite_face_centroid_mm,
            crystal_spec.rho_pplus_si,
            crystal_spec.rho_opposite_si,
        )
        log.info(f"ρ(x,y,z) = {rho_expr}")
        aedt_ops.setup_charge_density(maxwell_real, geom.crystal_name, rho_expr)

        log.info("设电压 BC...")
        aedt_ops.setup_voltages(maxwell_real, voltages)

        log.info("设 Matrix solution(电容矩阵)...")
        aedt_ops.setup_matrix_solution(maxwell_real, electrode_names)

        log.info(f"设自适应 Setup: max_passes={args.max_passes}, "
                 f"percent_error={args.percent_error}")
        aedt_ops.setup_adaptive(
            maxwell_real,
            max_passes=args.max_passes,
            percent_error=args.percent_error,
        )

        # ───── real_field 求解 ─────
        if "real_field" not in completed:
            log.info("====== 求解 real_field(adaptive)======")
            t0 = time.time()
            aedt_ops.solve(maxwell_real)
            log.info(f"real_field 求解完成,耗时 {time.time()-t0:.1f}s")

            # 导出 .fld → .h5
            log.info("导出 real_field 字段...")
            fld_dir = run_dir / "real_field" / "_fld_tmp"
            fld_paths = aedt_ops.export_field_files(
                maxwell_real, fld_dir, geom.crystal_name, grid
            )
            common_attrs = dict(
                source_step=args.step.name,
                step_sha256=manifest.inputs["step"]["sha256"],
                timestamp=timestamp,
                aedt_version=args.aedt_version,
                grid_assignment=geom.crystal_name,
            )
            for quantity, fld_path in fld_paths.items():
                h5_path = run_dir / "real_field" / f"{quantity}.h5"
                io_ops.fld_to_h5(fld_path, h5_path, quantity, **common_attrs)
                log.info(f"  {h5_path.relative_to(run_dir)}")
            shutil.rmtree(fld_dir, ignore_errors=True)

            # 电容矩阵
            log.info("提取电容矩阵...")
            cap_matrix = aedt_ops.extract_capacitance_matrix(
                maxwell_real, electrode_names
            )
            csv_path = run_dir / "capacitance_matrix.csv"
            io_ops.write_capacitance_csv(csv_path, electrode_names, cap_matrix)
            log.info(f"电容矩阵 → {csv_path.relative_to(run_dir)}")
            log.info(f"对角(自电容,pF): {np.diag(cap_matrix).tolist()}")

            manifest.mark_design_complete("real_field", manifest_path)
        else:
            log.info("[resume] real_field 已完成,跳过")

        # ───── 权重势 designs ─────
        for electrode in electrode_names:
            design_key = f"wpot_{electrode}"
            if design_key in completed:
                log.info(f"[resume] {design_key} 已完成,跳过")
                continue

            log.info(f"====== 权重势 design: {design_key} ======")
            t0 = time.time()
            m_w = aedt_ops.setup_weighting_design(
                maxwell_real, electrode, electrode_names
            )
            aedt_ops.solve(m_w)
            log.info(f"{design_key} 求解完成,耗时 {time.time()-t0:.1f}s")

            log.info(f"导出 {design_key} 字段...")
            fld_dir = run_dir / "weighting_potential" / electrode / "_fld_tmp"
            fld_paths = aedt_ops.export_field_files(
                m_w, fld_dir, geom.crystal_name, grid
            )
            for quantity, fld_path in fld_paths.items():
                h5_path = run_dir / "weighting_potential" / electrode / f"{quantity}.h5"
                io_ops.fld_to_h5(
                    fld_path, h5_path, quantity,
                    extra_attrs={"target_electrode": electrode},
                    **common_attrs,
                )
                log.info(f"  {h5_path.relative_to(run_dir)}")
            shutil.rmtree(fld_dir, ignore_errors=True)

            manifest.mark_design_complete(design_key, manifest_path)

        # ───── 收尾 ─────
        if args.cleanup_aedt:
            log.info("--cleanup-aedt:删除中间权重势 designs")
            for electrode in electrode_names:
                dn = f"wpot_{electrode}"
                try:
                    if dn in maxwell_real.design_list:
                        maxwell_real.delete_design(dn)
                except Exception as e:
                    log.warning(f"删除 {dn} 失败: {e}")

        manifest.status = "success"
        manifest.write(manifest_path)
        log.info("====== HPGe 仿真完成 ======")
        log.info(f"manifest: {manifest_path}")
        return 0

    except SpecError as e:
        log.error(f"规范错误: {e}")
        manifest.status = "failed"
        manifest.add_error(str(e))
        manifest.write(manifest_path)
        return 2

    except Exception as e:
        log.error(f"运行时异常: {e}")
        log.error(traceback.format_exc())
        manifest.status = "failed"
        manifest.add_error(f"{type(e).__name__}: {e}")
        manifest.write(manifest_path)
        return 1

    finally:
        aedt_ops.close_aedt(desktop, save=True)


if __name__ == "__main__":
    sys.exit(main())
