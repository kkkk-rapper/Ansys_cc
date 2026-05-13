"""
AEDT 几何拓扑识别。

读取已导入 Maxwell 设计的 body 集合,自动识别:
- Crystal / PPlus_* / NPlus_* 角色
- P+ 端面 vs 对面端面(用于 ρ 梯度方向)
- Crystal 包围盒(用于网格范围自动化)

依赖 pyaedt(ansys-aedt-core)。所有几何查询走 pyaedt 标准 API。
不确定的 API 用 # TODO(verify) 标注。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from spec import (
    SpecError,
    classify_electrode,
    validate_part_names,
)


@dataclass
class CrystalGeometry:
    """Crystal 及其与电极的拓扑关系。"""
    crystal_name: str
    pplus_names: list[str]
    nplus_names: list[str]
    crystal_volume_mm3: float
    bbox_min_mm: np.ndarray            # [xmin, ymin, zmin]
    bbox_max_mm: np.ndarray            # [xmax, ymax, zmax]
    pplus_face_centroid_mm: np.ndarray # P+ 端面合并质心(mm)
    opposite_face_centroid_mm: np.ndarray
    pplus_face_ids: list[int]
    opposite_face_ids: list[int]


def detect_geometry(maxwell, pplus_face_tol_mm: float = 0.01) -> CrystalGeometry:
    """
    从 Maxwell 实例读取所有 body,识别 Crystal/电极,定位 P+ 端面与对面端面。

    Args:
        maxwell: pyaedt Maxwell3d 实例,几何已导入
        pplus_face_tol_mm: 判断"面共享"的距离容差

    Raises:
        SpecError: 命名不合规、拓扑异常、几何病态
    """
    object_names = list(maxwell.modeler.object_names)

    pplus_names, nplus_names = validate_part_names(object_names)
    crystal_name = "Crystal"

    # 各 body 体积(mm³)
    volumes = {name: float(maxwell.modeler[name].volume) for name in object_names}
    crystal_vol = volumes[crystal_name]
    other_vols = {n: v for n, v in volumes.items() if n != crystal_name}
    if any(v >= crystal_vol for v in other_vols.values()):
        raise SpecError(
            f"Crystal 不是体积最大的 body。Crystal={crystal_vol:.2f} mm³,"
            f"其它={ {k: round(v, 2) for k, v in other_vols.items()} }。"
            "可能零件命名贴错了 —— 请检查 SolidWorks 装配体。"
        )

    # Crystal bbox
    crystal_obj = maxwell.modeler[crystal_name]
    bbox = _get_bbox_mm(maxwell, crystal_obj)
    bbox_min = np.array(bbox[:3])
    bbox_max = np.array(bbox[3:])

    # 收集 Crystal 所有面的质心 + 法向 + 面积
    crystal_face_info = _collect_face_info(maxwell, crystal_obj)
    # crystal_face_info: list of dict {id, centroid, normal, area}

    # P+ 端面:与任一 PPlus_* 共享面的 Crystal 面
    pplus_face_ids = _find_shared_faces(
        maxwell, crystal_obj, [maxwell.modeler[n] for n in pplus_names],
        tol_mm=pplus_face_tol_mm,
    )
    if not pplus_face_ids:
        raise SpecError(
            "未找到任何 Crystal 面与 PPlus_* 共享(在容差 "
            f"{pplus_face_tol_mm} mm 内)。"
            "可能 P+ 与 Crystal 不接触,或几何容差太大。"
        )
    pplus_centroid, pplus_total_area = _merge_face_centroids(
        crystal_face_info, pplus_face_ids
    )

    # 对面端面:在所有非 P+ 触面的 Crystal 面中,
    # 选与 pplus_centroid 距离最远 + 法向几乎反平行的那个(组)。
    opposite_face_ids = _find_opposite_face(
        crystal_face_info, pplus_face_ids, pplus_centroid
    )
    if not opposite_face_ids:
        raise SpecError(
            "未找到合适的对面端面(法向接近反平行 + 距离最远)。"
            "Crystal 几何可能不规则,需手动指定梯度方向。"
        )
    opposite_centroid, opposite_total_area = _merge_face_centroids(
        crystal_face_info, opposite_face_ids
    )

    if float(np.linalg.norm(opposite_centroid - pplus_centroid)) < 1e-3:
        raise SpecError(
            "P+ 端面质心与对面端面质心几乎重合(<1μm)。Crystal 几何病态。"
        )

    return CrystalGeometry(
        crystal_name=crystal_name,
        pplus_names=pplus_names,
        nplus_names=nplus_names,
        crystal_volume_mm3=crystal_vol,
        bbox_min_mm=bbox_min,
        bbox_max_mm=bbox_max,
        pplus_face_centroid_mm=pplus_centroid,
        opposite_face_centroid_mm=opposite_centroid,
        pplus_face_ids=pplus_face_ids,
        opposite_face_ids=opposite_face_ids,
    )


# ============================================================================
# 内部 helpers —— pyaedt API 包装层
# ============================================================================


def _get_bbox_mm(maxwell, obj) -> list[float]:
    """
    返回 [xmin, ymin, zmin, xmax, ymax, zmax],mm。

    TODO(verify): pyaedt 0.10 中 obj.bounding_box 属性返回 [x_min, y_min, z_min,
                  x_max, y_max, z_max] 的列表。单位由 model_units 决定;典型 mm。
    """
    bb = obj.bounding_box
    # 防御:有些版本返回字符串(带单位),需要清洗
    cleaned = []
    for v in bb:
        if isinstance(v, str):
            # 去掉末尾单位
            s = v.strip().rstrip("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ").strip()
            cleaned.append(float(s))
        else:
            cleaned.append(float(v))
    if len(cleaned) != 6:
        raise SpecError(f"bbox 返回非预期格式: {bb}")
    return cleaned


def _collect_face_info(maxwell, obj) -> list[dict]:
    """
    对一个 body 的所有面,收集 id、质心(mm)、法向、面积。

    TODO(verify): pyaedt 0.10 中 obj.faces 是 face 对象列表,有 .center, .normal, .area。
                  实际属性名可能略有差异(.centroid? .face_normal?)。
    """
    out = []
    for face in obj.faces:
        fid = int(face.id)
        # 质心(米)→ 毫米
        center = np.array(face.center, dtype=float)
        # pyaedt 通常返回 SI 米;若 model_units = mm,可能直接是 mm。
        # 防御性:若 |bbox| 的尺度暗示 mm,则不再 × 1000。这里用 obj.bounding_box 的尺度推断。
        center_mm = _coords_to_mm(maxwell, center)
        try:
            normal = np.array(face.normal, dtype=float)
            # 单位化
            n_norm = float(np.linalg.norm(normal))
            if n_norm > 1e-12:
                normal = normal / n_norm
            else:
                normal = np.array([0.0, 0.0, 0.0])
        except Exception:
            normal = np.array([0.0, 0.0, 0.0])
        try:
            area = float(face.area)
            # 面积单位推断(mm² 或 m²)
            area_mm2 = _area_to_mm2(maxwell, area)
        except Exception:
            area_mm2 = 0.0
        out.append({
            "id": fid,
            "centroid": center_mm,
            "normal": normal,
            "area": area_mm2,
        })
    return out


def _coords_to_mm(maxwell, vec: np.ndarray) -> np.ndarray:
    """
    根据 model_units 把坐标转 mm。

    TODO(verify): maxwell.modeler.model_units 给出当前单位字符串("mm" / "meter" / ...)。
    """
    units = str(getattr(maxwell.modeler, "model_units", "mm")).lower()
    if units in ("mm", "millimeter"):
        return vec
    if units in ("cm", "centimeter"):
        return vec * 10.0
    if units in ("m", "meter"):
        return vec * 1000.0
    if units in ("um", "micrometer"):
        return vec / 1000.0
    # 未识别:默认认为已是 mm
    return vec


def _area_to_mm2(maxwell, area: float) -> float:
    units = str(getattr(maxwell.modeler, "model_units", "mm")).lower()
    if units in ("mm", "millimeter"):
        return area
    if units in ("cm", "centimeter"):
        return area * 100.0
    if units in ("m", "meter"):
        return area * 1e6
    if units in ("um", "micrometer"):
        return area * 1e-6
    return area


def _find_shared_faces(
    maxwell, crystal_obj, electrode_objs: list, tol_mm: float = 0.01
) -> list[int]:
    """
    找 crystal_obj 上与任一 electrode_obj 共享的面 ID 列表。

    共享面判据:两面质心距离 < tol_mm 且法向几乎反平行(电极外法向 = 晶体内法向反向)。

    TODO(verify): AEDT 没有直接的 "shared face" API。这里用质心 + 法向重合判据近似。
                  另一种方法是用 oEditor.GetFaceByPosition + obj.GetFaceCenter 比对。
                  对真实 SolidWorks STEP 应该足够,因为共面零件的对应面在 STEP 里
                  几乎是数值同一。
    """
    crystal_faces = _collect_face_info(maxwell, crystal_obj)
    shared_ids: list[int] = []
    for electrode in electrode_objs:
        electrode_faces = _collect_face_info(maxwell, electrode)
        for cf in crystal_faces:
            for ef in electrode_faces:
                dist = float(np.linalg.norm(cf["centroid"] - ef["centroid"]))
                # 法向应当反平行:cos < -0.95
                cos = float(np.dot(cf["normal"], ef["normal"]))
                if dist < tol_mm and cos < -0.95:
                    if cf["id"] not in shared_ids:
                        shared_ids.append(cf["id"])
    return shared_ids


def _merge_face_centroids(
    face_infos: list[dict], face_ids: list[int]
) -> tuple[np.ndarray, float]:
    """面积加权合并质心,返回 (centroid_mm, total_area_mm2)。"""
    picked = [fi for fi in face_infos if fi["id"] in face_ids]
    if not picked:
        return np.zeros(3), 0.0
    total_area = sum(fi["area"] for fi in picked)
    if total_area <= 0:
        # 退化:简单平均
        return np.mean([fi["centroid"] for fi in picked], axis=0), 0.0
    weighted = sum(fi["centroid"] * fi["area"] for fi in picked)
    return weighted / total_area, total_area


def _find_opposite_face(
    crystal_face_infos: list[dict],
    pplus_face_ids: list[int],
    pplus_centroid: np.ndarray,
) -> list[int]:
    """
    在 Crystal 非 P+ 触面中,挑距离 pplus_centroid 最远、且法向与 P+ 平均法向反平行的面。

    返回一个 face id 列表(允许多个共面的面被一起当作"对面端")。
    """
    # P+ 端面平均外法向(指向 Crystal 之外)
    pplus_faces = [fi for fi in crystal_face_infos if fi["id"] in pplus_face_ids]
    if not pplus_faces:
        return []
    pplus_normal = np.mean([fi["normal"] for fi in pplus_faces], axis=0)
    n_norm = float(np.linalg.norm(pplus_normal))
    if n_norm > 1e-12:
        pplus_normal = pplus_normal / n_norm

    # 对面端面的法向应该和 P+ 法向**反**平行(都是 Crystal 的外法向)
    candidates = [fi for fi in crystal_face_infos if fi["id"] not in pplus_face_ids]
    if not candidates:
        return []

    scored = []
    for fi in candidates:
        if fi["area"] <= 0:
            continue
        cos = float(np.dot(fi["normal"], pplus_normal))
        # 反平行 ⇒ cos 接近 -1
        if cos > -0.85:
            continue
        dist = float(np.linalg.norm(fi["centroid"] - pplus_centroid))
        scored.append((fi, dist, cos))

    if not scored:
        return []

    # 选距离最大、面积排在 top-5
    scored.sort(key=lambda t: t[1], reverse=True)
    # 取离 pplus 最远的那一组 —— 法向相似且距离接近最大值的面合并为一组
    top_face = scored[0][0]
    top_dist = scored[0][1]
    top_normal = top_face["normal"]
    grouped = []
    for fi, dist, cos in scored:
        if (
            abs(dist - top_dist) < 0.5  # mm,同一端面的不同子面距离差不大
            and float(np.dot(fi["normal"], top_normal)) > 0.95
        ):
            grouped.append(fi["id"])

    return grouped or [top_face["id"]]
