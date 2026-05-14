"""
最小 STEP 解析器:从 STEP 文件读出每个 PRODUCT 的名字 + bbox。

只用 stdlib (re),无需 pythonocc/cadquery。

策略:
- 解析 DATA 段所有 entity 为 {id: (type, args_str)}
- 对每个 PRODUCT,沿引用链找到对应的 SHAPE_REPRESENTATION:
    PRODUCT
      ← PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE / PRODUCT_DEFINITION_FORMATION
      ← PRODUCT_DEFINITION
      ← PRODUCT_DEFINITION_SHAPE
      ← SHAPE_DEFINITION_REPRESENTATION
        (其第二个参数指向 SHAPE_REPRESENTATION)
- 从该 SHAPE_REPRESENTATION 递归遍历所有引用,收集 CARTESIAN_POINT
- 这些点的坐标范围即为 PRODUCT 的几何 bbox

匹配 AEDT body 时,用 bbox 中心距离最近的产品名作为该 body 的真实角色。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

_ENTITY_RE = re.compile(r"^#(\d+)\s*=\s*(\w+)\s*\((.*)\)$", re.DOTALL)
_REF_RE = re.compile(r"#(\d+)")
_NAME_RE = re.compile(r"^\s*'([^']*)'")
_POINT_COORDS_RE = re.compile(
    r"\(\s*([-+\d.eE]+)\s*,\s*([-+\d.eE]+)\s*,\s*([-+\d.eE]+)\s*\)"
)


def parse_step_entities(path: Path) -> dict[int, tuple[str, str]]:
    """读 STEP 文件,返回 {entity_id: (TYPE, args_text)}。多行 entity 已合并。"""
    text = path.read_text(encoding="utf-8", errors="replace")
    # 截 DATA;...ENDSEC; 段
    try:
        data_section = text.split("DATA;", 1)[1].split("ENDSEC;", 1)[0]
    except IndexError:
        raise ValueError(f"STEP 文件 {path} 没有 DATA 段")

    entities: dict[int, tuple[str, str]] = {}
    # 按 ';' 切;STEP 实体以 ; 结尾
    for raw_block in data_section.split(";"):
        block = raw_block.strip()
        if not block or not block.startswith("#"):
            continue
        m = _ENTITY_RE.match(block)
        if not m:
            continue
        entities[int(m.group(1))] = (m.group(2), m.group(3))
    return entities


def _refs(args: str) -> list[int]:
    return [int(x) for x in _REF_RE.findall(args)]


def _reachable(entities: dict[int, tuple[str, str]], start: int) -> set[int]:
    """从 start 沿引用 BFS 可达的所有 id。"""
    seen: set[int] = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        if cur not in entities:
            continue
        for r in _refs(entities[cur][1]):
            if r not in seen:
                seen.add(r)
                stack.append(r)
    return seen


def _build_reverse(entities: dict[int, tuple[str, str]]) -> dict[int, list[int]]:
    """反向引用表:谁引用了我。"""
    rev: dict[int, list[int]] = {}
    for eid, (_etype, args) in entities.items():
        for r in _refs(args):
            rev.setdefault(r, []).append(eid)
    return rev


def _types_referencing(
    reverse: dict[int, list[int]],
    entities: dict[int, tuple[str, str]],
    target_id: int,
    type_names: Iterable[str],
) -> list[int]:
    """找反向引用 target_id 且类型在 type_names 内的 entity ids。"""
    type_set = set(type_names)
    return [r for r in reverse.get(target_id, [])
            if entities.get(r, ("", ""))[0] in type_set]


def _find_shape_reps_for_product(
    entities: dict[int, tuple[str, str]],
    reverse: dict[int, list[int]],
    product_id: int,
) -> list[int]:
    """沿 PRODUCT → PRODUCT_DEFINITION_FORMATION → PRODUCT_DEFINITION
       → PRODUCT_DEFINITION_SHAPE → SHAPE_DEFINITION_REPRESENTATION → SHAPE_REPRESENTATION
    """
    formations = _types_referencing(
        reverse, entities, product_id,
        ["PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE",
         "PRODUCT_DEFINITION_FORMATION"],
    )
    defs: list[int] = []
    for f in formations:
        defs.extend(_types_referencing(reverse, entities, f, ["PRODUCT_DEFINITION"]))
    def_shapes: list[int] = []
    for d in defs:
        def_shapes.extend(_types_referencing(
            reverse, entities, d, ["PRODUCT_DEFINITION_SHAPE"]))
    sdrs: list[int] = []
    for ds in def_shapes:
        sdrs.extend(_types_referencing(
            reverse, entities, ds, ["SHAPE_DEFINITION_REPRESENTATION"]))
    # SDR(prod_def_shape, shape_rep) — 第二个引用 = shape rep
    shape_reps = []
    for sdr in sdrs:
        refs = _refs(entities[sdr][1])
        if len(refs) >= 2:
            shape_reps.append(refs[1])
    return shape_reps


def _expand_via_shape_rep_relationships(
    entities: dict[int, tuple[str, str]],
    seed_shape_reps: list[int],
) -> set[int]:
    """从初始 SHAPE_REP 集合出发,沿 SHAPE_REPRESENTATION_RELATIONSHIP /
    REPRESENTATION_RELATIONSHIP[_WITH_TRANSFORMATION] 扩展,
    收集所有关联的 SHAPE_REP / ADVANCED_BREP_SHAPE_REPRESENTATION ids。

    REPRESENTATION_RELATIONSHIP 在 STEP 里有变体:
      - SHAPE_REPRESENTATION_RELATIONSHIP('','',#parent,#child)
      - REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION(...) —— 复合 entity
    args 里包含两个 SHAPE_REP 引用,我们把双向关系都加入。
    """
    related = set(seed_shape_reps)
    changed = True
    # 收集所有 RELATIONSHIP 类 entities
    relationship_types = (
        "SHAPE_REPRESENTATION_RELATIONSHIP",
        "REPRESENTATION_RELATIONSHIP",
    )
    rel_entities = []
    for eid, (etype, args) in entities.items():
        # etype 可能是复合(包含多个类型名);检查 args 字符串里是否含 RELATIONSHIP
        if any(t in etype for t in relationship_types) or "REPRESENTATION_RELATIONSHIP" in args:
            rel_entities.append((eid, args))

    while changed:
        changed = False
        for _eid, args in rel_entities:
            rs = _refs(args)
            if any(r in related for r in rs):
                for r in rs:
                    if r not in related and r in entities:
                        # 只跟随 SHAPE_REP / ADVANCED_BREP_SHAPE_REPRESENTATION 类
                        etype = entities[r][0]
                        if ("SHAPE_REPRESENTATION" in etype
                                or "BREP" in etype
                                or "REPRESENTATION" in etype):
                            related.add(r)
                            changed = True
    return related


def extract_product_geometry(step_path: Path) -> dict[str, dict]:
    """
    返回 {product_name: {bbox_min, bbox_max, centroid, n_points}}。

    坐标单位 = STEP 原生(SolidWorks STEP 默认 mm)。
    """
    entities = parse_step_entities(step_path)
    reverse = _build_reverse(entities)

    # 全部 CARTESIAN_POINT 坐标
    points: dict[int, tuple[float, float, float]] = {}
    for eid, (etype, args) in entities.items():
        if etype == "CARTESIAN_POINT":
            m = _POINT_COORDS_RE.search(args)
            if m:
                points[eid] = (float(m.group(1)), float(m.group(2)), float(m.group(3)))

    # 全部 PRODUCT
    products = []
    for eid, (etype, args) in entities.items():
        if etype == "PRODUCT":
            m = _NAME_RE.match(args)
            if m:
                products.append((eid, m.group(1)))

    out: dict[str, dict] = {}
    for product_id, name in products:
        # 1. PRODUCT 链到本身的 SHAPE_REP(通常是局部坐标系)
        seed_reps = _find_shape_reps_for_product(entities, reverse, product_id)
        # 2. 沿 SHAPE_REPRESENTATION_RELATIONSHIP 扩到 ADVANCED_BREP_SHAPE_REPRESENTATION
        all_reps = _expand_via_shape_rep_relationships(entities, seed_reps)
        # 3. 从每个 rep 递归遍历,收 CARTESIAN_POINT
        all_points = []
        for sr in all_reps:
            for pid in _reachable(entities, sr):
                if pid in points:
                    all_points.append(points[pid])

        if not all_points:
            continue
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        zs = [p[2] for p in all_points]
        bbox_min = (min(xs), min(ys), min(zs))
        bbox_max = (max(xs), max(ys), max(zs))
        centroid = (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))
        out[name] = {
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "centroid": centroid,
            "n_points": len(all_points),
        }
    return out


def filter_leaf_products(
    products: dict[str, dict],
    valid_name_pattern: re.Pattern[str] | None = None,
) -> dict[str, dict]:
    """
    剔除装配体外壳:外壳的 bbox 通常是所有子件 bbox 的并集。
    如果给定 valid_name_pattern,只保留匹配的 PRODUCT。
    """
    if valid_name_pattern is None:
        valid_name_pattern = re.compile(r"^(Crystal|PPlus_\d{2}|NPlus_\d{2})$")
    return {k: v for k, v in products.items() if valid_name_pattern.match(k)}
