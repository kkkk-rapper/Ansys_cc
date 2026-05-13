"""
File IO operations.

- .fld(AEDT 原生 ASCII)→ .h5 转换
- manifest.yaml 原子读写 + 状态机
- capacitance_matrix.csv 写入
- 一些路径计算辅助

无 AEDT 依赖,可独立单元测试。
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import yaml

# 用户接受的默认压缩等级(README §4.3)
HDF5_COMPRESSION = "gzip"
HDF5_COMPRESSION_OPTS = 4

MANIFEST_FILENAME = "manifest.yaml"

# 字段量 → 单位映射
QUANTITY_UNITS = {
    "phi":   "V",
    "E_x":   "V/m",
    "E_y":   "V/m",
    "E_z":   "V/m",
    "E_mag": "V/m",
}


# ============================================================================
# .fld 解析 + .h5 写入
# ============================================================================


def parse_fld_grid(fld_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    解析 AEDT 导出的 .fld 文件(笛卡尔规则网格)。

    AEDT .fld 笛卡尔格式:
        头若干行(注释,有时含 grid 信息)
        数据行: x y z value     (每个格点一行,SI 单位 m / V / V·m⁻¹)

    返回 (x, y, z, values),其中 x/y/z 是 1-D 唯一轴 (mm),
    values 是 [Nx, Ny, Nz] 形状的 3-D 数组(SI 单位)。

    TODO(verify): 实际 .fld 文件首部格式可能因 AEDT 版本不同而变。
                  当前实现假设是 "x y z v" 4列纯数据(忽略以 '*' 或 '$' 起始的行)。
                  首次真跑后用真实文件 sanity check。
    """
    if not fld_path.is_file():
        raise FileNotFoundError(f"找不到 .fld 文件: {fld_path}")

    rows = []
    with fld_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(("*", "$", "#", "%")):
                continue
            parts = s.split()
            if len(parts) < 4:
                continue
            try:
                x, y, z, v = (float(parts[0]), float(parts[1]),
                              float(parts[2]), float(parts[3]))
            except ValueError:
                continue
            rows.append((x, y, z, v))

    if not rows:
        raise ValueError(f"{fld_path}: 未解析出任何数据点")

    arr = np.asarray(rows, dtype=np.float64)
    # AEDT 默认导出位置单位是 m;转回 mm 给元数据
    xs_m, ys_m, zs_m, vs = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

    xs = np.unique(np.round(xs_m, 9))
    ys = np.unique(np.round(ys_m, 9))
    zs = np.unique(np.round(zs_m, 9))
    Nx, Ny, Nz = xs.size, ys.size, zs.size
    if Nx * Ny * Nz != arr.shape[0]:
        # 数据点数不匹配规则网格;可能是非规则网格或丢点
        raise ValueError(
            f"{fld_path}: 数据点数 {arr.shape[0]} ≠ Nx*Ny*Nz "
            f"({Nx}*{Ny}*{Nz}={Nx*Ny*Nz}),网格非规则或导出不完整"
        )

    # 构 3-D 数组。AEDT 通常先 z 后 y 后 x 扫描;但为稳妥用索引重排。
    ix = np.searchsorted(xs, np.round(xs_m, 9))
    iy = np.searchsorted(ys, np.round(ys_m, 9))
    iz = np.searchsorted(zs, np.round(zs_m, 9))
    values = np.empty((Nx, Ny, Nz), dtype=np.float32)
    values[ix, iy, iz] = vs.astype(np.float32)

    # 轴单位 mm(README §4.3)
    return xs * 1e3, ys * 1e3, zs * 1e3, values


def write_h5_field(
    h5_path: Path,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    z_mm: np.ndarray,
    values: np.ndarray,
    quantity: str,
    *,
    source_step: str,
    step_sha256: str,
    timestamp: str,
    aedt_version: str,
    grid_assignment: str = "Crystal",
    extra_attrs: dict[str, Any] | None = None,
) -> None:
    """
    写一个标量场到 .h5。布局严格按 README §4.3。

    /grid/{x,y,z}     float32[N*]      mm
    /data             float32[Nx,Ny,Nz]
    attrs on root     quantity, unit, source_step, step_sha256, timestamp, ...
    """
    if quantity not in QUANTITY_UNITS:
        raise ValueError(f"未知 quantity: {quantity}; 期望 {list(QUANTITY_UNITS)}")

    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as f:
        g = f.create_group("grid")
        g.create_dataset("x", data=x_mm.astype(np.float32))
        g.create_dataset("y", data=y_mm.astype(np.float32))
        g.create_dataset("z", data=z_mm.astype(np.float32))

        f.create_dataset(
            "data",
            data=values.astype(np.float32),
            compression=HDF5_COMPRESSION,
            compression_opts=HDF5_COMPRESSION_OPTS,
        )

        f.attrs["quantity"] = quantity
        f.attrs["unit"] = QUANTITY_UNITS[quantity]
        f.attrs["grid_assignment"] = grid_assignment
        f.attrs["source_step"] = source_step
        f.attrs["step_sha256"] = step_sha256
        f.attrs["timestamp"] = timestamp
        f.attrs["aedt_version"] = aedt_version
        if extra_attrs:
            for k, v in extra_attrs.items():
                f.attrs[k] = v


def fld_to_h5(
    fld_path: Path,
    h5_path: Path,
    quantity: str,
    **attrs,
) -> None:
    """端到端:.fld → .h5,顺手删除 .fld(节省磁盘)。"""
    x, y, z, vals = parse_fld_grid(fld_path)
    write_h5_field(h5_path, x, y, z, vals, quantity, **attrs)


# ============================================================================
# Capacitance matrix CSV
# ============================================================================


def write_capacitance_csv(
    csv_path: Path,
    electrode_names: list[str],
    matrix_pF: np.ndarray,
) -> None:
    """
    写 Maxwell 电容矩阵 CSV。

    布局:
        ,PPlus_01,NPlus_01
        PPlus_01,12.34,-12.30
        NPlus_01,-12.30,12.45

    单位 pF。约定:Maxwell 电容矩阵(对角自电容,非对角负互电容)。
    """
    if matrix_pF.shape != (len(electrode_names), len(electrode_names)):
        raise ValueError(
            f"电容矩阵形状 {matrix_pF.shape} 与电极数 {len(electrode_names)} 不匹配"
        )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + electrode_names)
        for i, row_name in enumerate(electrode_names):
            row = [row_name] + [f"{matrix_pF[i, j]:.6e}" for j in range(len(electrode_names))]
            writer.writerow(row)


# ============================================================================
# Manifest 状态机
# ============================================================================


@dataclass
class Manifest:
    """
    manifest.yaml 的内存表示。
    所有更新通过 .write(path) 原子落盘(写 .tmp → rename)。
    """
    run_id: str
    timestamp: str
    status: str = "running"      # running | success | partial | failed
    aedt_version: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    geometry_detected: dict[str, Any] = field(default_factory=dict)
    completed_designs: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "status": self.status,
            "aedt_version": self.aedt_version,
            "inputs": self.inputs,
            "geometry_detected": self.geometry_detected,
            "completed_designs": list(self.completed_designs),
            "errors": list(self.errors),
        }

    def write(self, path: Path) -> None:
        """原子写:写到 .tmp,然后 rename。中断时 manifest 处于一致状态。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=".manifest.", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    self.to_dict(),
                    f,
                    allow_unicode=True,
                    sort_keys=False,
                    default_flow_style=False,
                )
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def mark_design_complete(self, design_name: str, path: Path) -> None:
        if design_name not in self.completed_designs:
            self.completed_designs.append(design_name)
            self.write(path)

    def add_error(self, message: str, design: str | None = None) -> None:
        self.errors.append({
            "message": message,
            "design": design,
            "timestamp": now_iso(),
        })

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        if not path.is_file():
            raise FileNotFoundError(f"manifest.yaml 不存在: {path}")
        with path.open("r", encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        return cls(
            run_id=d.get("run_id", ""),
            timestamp=d.get("timestamp", ""),
            status=d.get("status", "unknown"),
            aedt_version=d.get("aedt_version", ""),
            inputs=d.get("inputs", {}),
            geometry_detected=d.get("geometry_detected", {}),
            completed_designs=list(d.get("completed_designs", [])),
            errors=list(d.get("errors", [])),
        )


# ============================================================================
# 路径辅助
# ============================================================================


def today_yyyymmdd(now: dt.datetime | None = None) -> str:
    """本地时区的 YYYYMMDD。"""
    now = now or dt.datetime.now()
    return now.strftime("%Y%m%d")


def now_iso(now: dt.datetime | None = None) -> str:
    """带本地时区偏移的 ISO 8601 字符串。"""
    now = now or dt.datetime.now().astimezone()
    return now.isoformat(timespec="seconds")


def compute_run_dir(
    output_root: Path,
    step_stem: str,
    op_stem: str | None,
    run_name_override: str | None,
    now: dt.datetime | None = None,
) -> Path:
    """
    /data_out/YYYYMMDD/<step>__<op>/(README §4.1)。

    op_stem 为 None 时必须提供 run_name_override(README §3 关于 CLI 模式的硬性约束)。
    """
    date = today_yyyymmdd(now)
    if run_name_override:
        run_name = run_name_override
    elif op_stem is None:
        raise ValueError(
            "未提供 operating.yaml 且未提供 --run-name;无法构造输出子目录名"
        )
    else:
        run_name = f"{step_stem}__{op_stem}"
    return Path(output_root) / date / run_name


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def estimate_output_size_bytes(
    n_electrodes: int,
    grid_shape: tuple[int, int, int],
    compression_factor: float = 0.4,
) -> int:
    """
    估算输出目录大小。粗略上界:
      每个 design 5 个标量场 × Nx*Ny*Nz × 4 字节(float32)× 压缩因子
      design 总数 = 1 (real_field) + N (weighting)
    """
    Nx, Ny, Nz = grid_shape
    per_field = Nx * Ny * Nz * 4
    per_design = per_field * 5
    n_designs = 1 + n_electrodes
    return int(per_design * n_designs * compression_factor)


def safe_filename(name: str) -> str:
    """把任意字符串改成可以做文件名的形式(替换 / : 空格 等)。"""
    return re.sub(r"[^\w.-]+", "_", name)
