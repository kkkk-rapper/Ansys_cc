"""
Schema validation, yaml loading, ρ gradient math.

Pure Python (PyYAML + NumPy only). No AEDT dependencies, fully unit-testable.

参考 README.md §2.2、§2.3 的字段约定。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ELEMENTARY_CHARGE = 1.602176634e-19  # Coulomb

PART_NAME_RE = re.compile(r"^(Crystal|PPlus_\d{2}|NPlus_\d{2})$")
PPLUS_RE = re.compile(r"^PPlus_\d{2}$")
NPLUS_RE = re.compile(r"^NPlus_\d{2}$")

CONCENTRATION_RANGE_CM3 = (1e6, 1e13)
GRID_STEP_RANGE_MM = (0.05, 5.0)
GRID_PADDING_RANGE_MM = (0.0, 10.0)
VOLTAGE_WARN_THRESHOLD = 10_000.0


class SpecError(Exception):
    """规范校验失败。错误信息要可执行(行号、字段名、期望值)。"""


# ============================================================================
# CrystalSpec —— 物理参数(crystal.yaml)
# ============================================================================


@dataclass
class CrystalSpec:
    carrier_type: str                      # "p-type" | "n-type"
    pplus_concentration_cm3: float         # 永远 > 0
    opposite_concentration_cm3: float      # 永远 > 0
    temperature_K: float
    crystal_material: str
    source_path: Path                      # 这份 yaml 的路径,用于错误追溯
    raw: dict[str, Any]                    # 原始解析结果,manifest 用

    @property
    def rho_sign(self) -> int:
        return -1 if self.carrier_type == "p-type" else +1

    @property
    def rho_pplus_si(self) -> float:
        return cm3_to_rho_si(self.pplus_concentration_cm3, self.carrier_type)

    @property
    def rho_opposite_si(self) -> float:
        return cm3_to_rho_si(self.opposite_concentration_cm3, self.carrier_type)

    @classmethod
    def from_yaml(cls, path: Path) -> "CrystalSpec":
        if not path.is_file():
            raise SpecError(f"crystal.yaml 不存在: {path}")
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise SpecError(f"{path}: 顶层必须是 mapping")

        physics = _require(raw, "physics", path, dict)
        material = _require(raw, "material", path, dict, optional_default={})

        carrier_type = _require(physics, "carrier_type", path, str)
        if carrier_type not in ("p-type", "n-type"):
            raise SpecError(
                f"{path}: physics.carrier_type 必须是 'p-type' 或 'n-type',"
                f"实际 = {carrier_type!r}"
            )

        conc = _require(physics, "impurity_concentration", path, dict)
        pplus = _require(conc, "PPlus_end", path, (int, float))
        opposite = _require(conc, "opposite_end", path, (int, float))
        for name, val in (("PPlus_end", pplus), ("opposite_end", opposite)):
            if val <= 0:
                raise SpecError(
                    f"{path}: impurity_concentration.{name} 必须 > 0(单位 cm⁻³),"
                    f"实际 = {val}"
                )
            lo, hi = CONCENTRATION_RANGE_CM3
            if not (lo <= val <= hi):
                raise SpecError(
                    f"{path}: impurity_concentration.{name} = {val:.3e} 超出合理范围 "
                    f"[{lo:.0e}, {hi:.0e}] cm⁻³"
                )

        temperature_raw = physics.get("temperature", 77.0)
        try:
            temperature = float(temperature_raw)
        except (TypeError, ValueError):
            raise SpecError(f"{path}: physics.temperature 必须为数字 (K),实际 = {temperature_raw!r}")
        if temperature <= 0:
            raise SpecError(f"{path}: physics.temperature 必须 > 0 K,实际 = {temperature}")

        crystal_material = material.get("crystal", "Germanium")
        if not isinstance(crystal_material, str) or not crystal_material:
            raise SpecError(f"{path}: material.crystal 必须为非空字符串")

        return cls(
            carrier_type=carrier_type,
            pplus_concentration_cm3=float(pplus),
            opposite_concentration_cm3=float(opposite),
            temperature_K=float(temperature),
            crystal_material=crystal_material,
            source_path=path,
            raw=raw,
        )


# ============================================================================
# OperatingSpec —— 运行参数(operating.yaml + CLI 覆盖)
# ============================================================================


@dataclass
class OperatingSpec:
    voltages_explicit: dict[str, float]    # voltages: 段
    defaults: dict[str, float]              # 通配符 pattern → V
    grid_step_mm: float
    grid_padding_mm: float
    output_root: Path
    source_path: Path | None                # None 表示纯 CLI 模式
    cli_overrides: dict[str, float] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(
        cls,
        path: Path,
        cli_overrides: dict[str, float] | None = None,
        cli_grid_step_mm: float | None = None,
        cli_output_root: Path | None = None,
    ) -> "OperatingSpec":
        if not path.is_file():
            raise SpecError(f"operating.yaml 不存在: {path}")
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise SpecError(f"{path}: 顶层必须是 mapping")

        voltages = raw.get("voltages", {}) or {}
        if not isinstance(voltages, dict):
            raise SpecError(f"{path}: voltages 必须是 mapping")
        voltages_explicit = {}
        for k, v in voltages.items():
            if not isinstance(k, str):
                raise SpecError(f"{path}: voltages 的 key 必须是字符串,见 {k!r}")
            try:
                voltages_explicit[k] = float(v)
            except (TypeError, ValueError):
                raise SpecError(f"{path}: voltages[{k!r}] 必须是数字,实际 = {v!r}")

        defaults_raw = raw.get("defaults", {}) or {}
        if not isinstance(defaults_raw, dict):
            raise SpecError(f"{path}: defaults 必须是 mapping")
        defaults = {}
        for pattern, v in defaults_raw.items():
            if not isinstance(pattern, str):
                raise SpecError(f"{path}: defaults 的 key 必须是字符串")
            try:
                defaults[pattern] = float(v)
            except (TypeError, ValueError):
                raise SpecError(f"{path}: defaults[{pattern!r}] 必须是数字,实际 = {v!r}")

        grid = raw.get("grid", {}) or {}
        if not isinstance(grid, dict):
            raise SpecError(f"{path}: grid 必须是 mapping")
        grid_step_mm = float(grid.get("step_mm", 0.5))
        grid_padding_mm = float(grid.get("padding_mm", 0.5))

        if cli_grid_step_mm is not None:
            grid_step_mm = float(cli_grid_step_mm)
        lo, hi = GRID_STEP_RANGE_MM
        if not (lo <= grid_step_mm <= hi):
            raise SpecError(
                f"{path}: grid.step_mm = {grid_step_mm} 超出 [{lo}, {hi}] mm"
            )
        lo, hi = GRID_PADDING_RANGE_MM
        if not (lo <= grid_padding_mm <= hi):
            raise SpecError(
                f"{path}: grid.padding_mm = {grid_padding_mm} 超出 [{lo}, {hi}] mm"
            )

        if cli_output_root is not None:
            output_root = cli_output_root
        else:
            output_root = Path(raw.get("output_root", "./data_out"))

        return cls(
            voltages_explicit=voltages_explicit,
            defaults=defaults,
            grid_step_mm=grid_step_mm,
            grid_padding_mm=grid_padding_mm,
            output_root=output_root,
            source_path=path,
            cli_overrides=dict(cli_overrides or {}),
            raw=raw,
        )

    def resolve_voltages(self, electrode_names: list[str]) -> dict[str, float]:
        """
        展开 defaults 通配符 → 覆盖 voltages_explicit → 覆盖 cli_overrides。
        必须为 electrode_names 中每个名字给出最终电压;缺失 → SpecError。
        """
        resolved: dict[str, float] = {}
        for name in electrode_names:
            # 1. defaults: 通配符,任一匹配即生效。多匹配时取定义顺序中靠后的(dict 保序)。
            for pattern, v in self.defaults.items():
                if fnmatch(name, pattern):
                    resolved[name] = v
            # 2. voltages_explicit 覆盖
            if name in self.voltages_explicit:
                resolved[name] = self.voltages_explicit[name]
            # 3. CLI 覆盖
            if name in self.cli_overrides:
                resolved[name] = self.cli_overrides[name]

        missing = [n for n in electrode_names if n not in resolved]
        if missing:
            raise SpecError(
                "以下电极在 operating.yaml / CLI 中都没有电压定义:\n  - "
                + "\n  - ".join(missing)
                + "\n请在 voltages: 段显式给值,或加 defaults 通配符,或用 --voltage NAME=V 覆盖。"
            )

        for name, v in resolved.items():
            if abs(v) > VOLTAGE_WARN_THRESHOLD:
                # warning, 不 abort
                import warnings
                warnings.warn(
                    f"电极 {name} 电压 = {v} V,绝对值 > {VOLTAGE_WARN_THRESHOLD} V,"
                    "请确认是否符合预期",
                    UserWarning,
                )
        return resolved


# ============================================================================
# 零件命名校验
# ============================================================================


def classify_electrode(name: str) -> str:
    """返回 'pplus' / 'nplus' / 'crystal' / 'invalid'。"""
    if name == "Crystal":
        return "crystal"
    if PPLUS_RE.match(name):
        return "pplus"
    if NPLUS_RE.match(name):
        return "nplus"
    return "invalid"


def validate_part_names(names: list[str]) -> tuple[list[str], list[str]]:
    """
    校验导入后的零件名集合。返回 (pplus_list, nplus_list),排序后。
    失败抛 SpecError。
    """
    crystals = [n for n in names if classify_electrode(n) == "crystal"]
    pplus = sorted(n for n in names if classify_electrode(n) == "pplus")
    nplus = sorted(n for n in names if classify_electrode(n) == "nplus")
    invalid = [n for n in names if classify_electrode(n) == "invalid"]

    if len(crystals) != 1:
        raise SpecError(
            f"必须恰好有一个 'Crystal' 零件,实际找到 {len(crystals)} 个: {crystals}"
        )
    if not pplus:
        raise SpecError("没有找到任何 PPlus_NN 电极(至少需要一个)")
    if not nplus:
        raise SpecError("没有找到任何 NPlus_NN 电极(至少需要一个)")
    if invalid:
        raise SpecError(
            "以下零件名不符合命名约定(应为 Crystal / PPlus_NN / NPlus_NN):\n  - "
            + "\n  - ".join(invalid)
            + "\n参见 README §2.1 零件命名约定。"
        )
    return pplus, nplus


# ============================================================================
# ρ 数学
# ============================================================================


def cm3_to_rho_si(concentration_cm3: float, carrier_type: str) -> float:
    """
    杂质浓度 (cm⁻³) → SI 空间电荷密度 (C/m³),按 carrier_type 注入符号。

    ρ = sign × e × N (cm⁻³) × 1e6 (cm⁻³ → m⁻³)
    """
    sign = -1.0 if carrier_type == "p-type" else +1.0
    return sign * ELEMENTARY_CHARGE * concentration_cm3 * 1e6


def build_rho_expression(
    pplus_centroid_mm: np.ndarray,
    opposite_centroid_mm: np.ndarray,
    rho_pplus_si: float,
    rho_opposite_si: float,
) -> str:
    """
    构造 AEDT Volume Charge Density 用的位置依赖表达式。

    几何:沿 (C_O - C_P) 方向线性插值。展开为 ρ(x,y,z) = α + βX + γY + δZ。
    AEDT 表达式中 X/Y/Z 是 SI 单位 (m);质心传入是 mm。

    返回字符串如 "(-0.00032) + (0.0)*X + (0.0)*Y + (-2.46e-05)*Z"。
    """
    cp = np.asarray(pplus_centroid_mm, dtype=float) * 1e-3   # mm → m
    co = np.asarray(opposite_centroid_mm, dtype=float) * 1e-3
    axis = co - cp
    L = float(np.linalg.norm(axis))
    if L < 1e-9:
        raise SpecError(
            "P+ 端面质心与对面端面质心重合,无法建立梯度方向。"
            "几何可能病态(零件不接触 Crystal,或 Crystal 退化)。"
        )
    axis_unit = axis / L
    slope = (rho_opposite_si - rho_pplus_si) / L
    beta = slope * axis_unit[0]
    gamma = slope * axis_unit[1]
    delta = slope * axis_unit[2]
    alpha = rho_pplus_si - slope * float(np.dot(cp, axis_unit))
    return f"({alpha:.6e}) + ({beta:.6e})*X + ({gamma:.6e})*Y + ({delta:.6e})*Z"


# ============================================================================
# helpers
# ============================================================================


_MISSING = object()


def _require(d: dict, key: str, path: Path, types, optional_default=_MISSING):
    if key not in d:
        if optional_default is not _MISSING:
            return optional_default
        raise SpecError(f"{path}: 缺少必填字段 {key!r}")
    val = d[key]

    # 容错:YAML 1.1 对 "2.0e10"(无显式正号)解析为字符串;
    # 数字字段接受能转浮点的字符串。
    if isinstance(types, tuple) and (int in types or float in types):
        if isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                pass
    elif types is int or types is float:
        if isinstance(val, str):
            try:
                return types(val)
            except ValueError:
                pass

    if not isinstance(val, types):
        type_names = (
            types.__name__ if isinstance(types, type) else " | ".join(t.__name__ for t in types)
        )
        raise SpecError(
            f"{path}: 字段 {key!r} 应为 {type_names},实际 = {type(val).__name__}"
        )
    return val
