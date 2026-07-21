#!/usr/bin/env python3
"""M3ED 1k-event TTC 调试集的几何与标签生成工具。

相机标定、位姿插值、深度重投影和 TTC 计算口径参考
`EV-TTC-main/TTCEF/calc_gt.py`，但封装在本文件中，避免修改原 EV-TTC 源码。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, Tuple

import cv2
import h5py
import numpy as np
from scipy.linalg import logm

try:  # Prefer the same Sophus interpolation used by EV-TTC.
    from projectaria_tools.core.sophus import SE3, interpolate
except Exception:  # pragma: no cover - only used when projectaria_tools is absent.
    SE3 = None
    interpolate = None


US_TO_S = 1e-6
SPOT_SPEED_THRESHOLD = 0.25
CAR_SPEED_THRESHOLD = 1.3
OMEGA_THRESHOLD = 0.18


# 双线性四邻域的像素偏移，顺序与 EV-TTC create_exp.py 中的权重一致。
ROI_NEIGHBOR_DX = np.array([0, 1, 0, 1], dtype=np.int32)
ROI_NEIGHBOR_DY = np.array([0, 0, 1, 1], dtype=np.int32)


@dataclass
class CameraGeometry:
    x_map: np.ndarray
    y_map: np.ndarray
    K: np.ndarray
    dist_map: np.ndarray
    proj_pts: np.ndarray
    K_flow: np.ndarray


def calc_dist_map_ds(x_map: np.ndarray, y_map: np.ndarray) -> np.ndarray:
    """生成 EV-TTC 使用的 720x720 裁剪区域到 360x360 的双线性映射表。"""

    dist_map = np.zeros((720, 720, 6), dtype=np.float32)
    yy, xx = np.mgrid[0:720, 280:1000]
    id1 = x_map[yy, xx] / 2.0 - 140.0
    id2 = y_map[yy, xx] / 2.0
    valid = (id1 >= 0) & (id2 >= 0) & (id1 < 359) & (id2 < 359)

    nc1 = np.floor(id1[valid]).astype(np.int32)
    nc2 = np.floor(id2[valid]).astype(np.int32)
    fx = id1[valid] - nc1
    fy = id2[valid] - nc2
    out_y = yy[valid]
    out_x = xx[valid] - 280

    dist_map[out_y, out_x, 0] = nc1
    dist_map[out_y, out_x, 1] = nc2
    dist_map[out_y, out_x, 2] = (1.0 - fx) * (1.0 - fy)
    dist_map[out_y, out_x, 3] = fx * (1.0 - fy)
    dist_map[out_y, out_x, 4] = (1.0 - fx) * fy
    dist_map[out_y, out_x, 5] = fx * fy
    return dist_map


def load_camera_geometry(f_data: h5py.File) -> CameraGeometry:
    """读取事件相机标定，并构造后续重投影和事件计数需要的几何缓存。"""

    D = f_data["prophesee"]["left"]["calib"]["distortion_coeffs"][:]
    intr = np.asarray(f_data["prophesee"]["left"]["calib"]["intrinsics"][:], dtype=np.float64)
    K = np.array([[intr[0], 0.0, intr[2]], [0.0, intr[1], intr[3]], [0.0, 0.0, 1.0]])
    res = np.array([720, 1280])
    new_mtx, _ = cv2.getOptimalNewCameraMatrix(K, D, res[::-1], 0)
    x_map, y_map = cv2.initUndistortRectifyMap(
        K, D, np.eye(3), new_mtx, res[::-1], cv2.CV_32FC1
    )
    K = new_mtx.astype(np.float32)
    proj_pts = np.linalg.inv(K) @ np.stack(
        [x_map.flatten(), y_map.flatten(), np.ones_like(x_map.flatten())]
    )
    proj_pts /= proj_pts[2]

    K_flow = K.copy()
    K_flow[0, 2] -= 280
    K_flow /= 2.0
    K_flow[2, 2] = 1.0

    return CameraGeometry(
        x_map=x_map,
        y_map=y_map,
        K=K,
        dist_map=calc_dist_map_ds(x_map, y_map),
        proj_pts=proj_pts.astype(np.float32),
        K_flow=K_flow.astype(np.float32),
    )


def accumulate_event_count(
    x_ev: np.ndarray,
    y_ev: np.ndarray,
    p_ev: np.ndarray,
    dist_map: np.ndarray,
    height: int = 360,
    width: int = 360,
) -> np.ndarray:
    """把一个 1k-event step 累积成正/负极性两通道事件计数图。"""

    event_cnt = np.zeros((2, height, width), dtype=np.float32)
    # 与 EV-TTC 一致，只保留中间 720 像素宽区域，然后去畸变下采样到 360x360。
    in_crop = (x_ev >= 280) & (x_ev <= 999) & (y_ev < 720)
    if not np.any(in_crop):
        return event_cnt

    x = x_ev[in_crop].astype(np.int64)
    y = y_ev[in_crop].astype(np.int64)
    p = p_ev[in_crop]
    update = dist_map[y, x - 280]
    x_c = update[:, 0].astype(np.int64)
    y_c = update[:, 1].astype(np.int64)
    valid = (update[:, 2:].sum(axis=1) > 0) & (x_c >= 0) & (x_c < width - 1) & (y_c >= 0) & (y_c < height - 1)
    if not np.any(valid):
        return event_cnt

    x_c = x_c[valid]
    y_c = y_c[valid]
    weights = update[valid, 2:]
    channels = np.where(p[valid] > 0, 0, 1)

    # 一个原始事件按双线性权重分配到最多 4 个下采样像素。
    np.add.at(event_cnt, (channels, y_c, x_c), weights[:, 0])
    np.add.at(event_cnt, (channels, y_c, x_c + 1), weights[:, 1])
    np.add.at(event_cnt, (channels, y_c + 1, x_c), weights[:, 2])
    np.add.at(event_cnt, (channels, y_c + 1, x_c + 1), weights[:, 3])
    return event_cnt


def map_raw_events_to_roi(
    x_ev: np.ndarray,
    y_ev: np.ndarray,
    p_ev: np.ndarray,
    t_ev: np.ndarray,
    raw_indices: np.ndarray,
    dist_map: np.ndarray,
    roi_x0: int,
    roi_y0: int,
    roi_size: int = 128,
) -> Dict[str, np.ndarray]:
    """筛出双线性贡献进入固定 ROI 的源事件，并保留其 ROI 内权重。

    这里的“1 个 ROI 事件”仍然指 1 条源事件；它可能向 ROI 内 1-4 个像素贡献
    小数权重，但源事件计数只加 1。
    """

    in_crop = (x_ev >= 280) & (x_ev <= 999) & (y_ev < 720)
    if not np.any(in_crop):
        return {
            "raw_indices": np.empty((0,), dtype=np.int64),
            "t": np.empty((0,), dtype=np.int64),
            "p": np.empty((0,), dtype=np.int8),
            "base_x": np.empty((0,), dtype=np.int32),
            "base_y": np.empty((0,), dtype=np.int32),
            "weights_in_roi": np.empty((0, 4), dtype=np.float32),
            "inside_mask": np.empty((0, 4), dtype=bool),
            "fully_inside": np.empty((0,), dtype=bool),
            "total_weight": np.empty((0,), dtype=np.float32),
        }

    x = x_ev[in_crop].astype(np.int64)
    y = y_ev[in_crop].astype(np.int64)
    update = dist_map[y, x - 280]
    valid_map = update[:, 2:].sum(axis=1) > 0
    if not np.any(valid_map):
        return {
            "raw_indices": np.empty((0,), dtype=np.int64),
            "t": np.empty((0,), dtype=np.int64),
            "p": np.empty((0,), dtype=np.int8),
            "base_x": np.empty((0,), dtype=np.int32),
            "base_y": np.empty((0,), dtype=np.int32),
            "weights_in_roi": np.empty((0, 4), dtype=np.float32),
            "inside_mask": np.empty((0, 4), dtype=bool),
            "fully_inside": np.empty((0,), dtype=bool),
            "total_weight": np.empty((0,), dtype=np.float32),
        }

    kept_raw = raw_indices[in_crop][valid_map].astype(np.int64)
    kept_t = t_ev[in_crop][valid_map].astype(np.int64)
    kept_p = p_ev[in_crop][valid_map].astype(np.int8)
    update = update[valid_map]
    base_x = update[:, 0].astype(np.int32)
    base_y = update[:, 1].astype(np.int32)
    weights = update[:, 2:].astype(np.float32)

    nbr_x = base_x[:, None] + ROI_NEIGHBOR_DX[None, :]
    nbr_y = base_y[:, None] + ROI_NEIGHBOR_DY[None, :]
    inside = (
        (nbr_x >= roi_x0)
        & (nbr_x < roi_x0 + roi_size)
        & (nbr_y >= roi_y0)
        & (nbr_y < roi_y0 + roi_size)
        & (weights > 0)
    )
    total_weight = np.where(inside, weights, 0.0).sum(axis=1).astype(np.float32)
    roi_event = total_weight > 0

    # fully_inside 只考虑实际有正权重的邻域；零权重邻域不影响是否完全落入 ROI。
    active = weights > 0
    fully_inside = np.all(~active | inside, axis=1) & roi_event

    return {
        "raw_indices": kept_raw[roi_event],
        "t": kept_t[roi_event],
        "p": kept_p[roi_event],
        "base_x": base_x[roi_event],
        "base_y": base_y[roi_event],
        "weights_in_roi": np.where(inside[roi_event], weights[roi_event], 0.0).astype(np.float32),
        "inside_mask": inside[roi_event],
        "fully_inside": fully_inside[roi_event],
        "total_weight": total_weight[roi_event],
    }


def accumulate_roi_event_count(
    base_x: np.ndarray,
    base_y: np.ndarray,
    p_ev: np.ndarray,
    weights_in_roi: np.ndarray,
    roi_x0: int,
    roi_y0: int,
    roi_size: int = 128,
) -> np.ndarray:
    """把固定 ROI 内的 1000 条源事件累积成 [2, roi_size, roi_size] 浮点权重图。"""

    event_cnt = np.zeros((2, roi_size, roi_size), dtype=np.float32)
    channels = np.where(p_ev > 0, 0, 1)
    for nbr in range(4):
        weight = weights_in_roi[:, nbr]
        valid = weight > 0
        if not np.any(valid):
            continue
        x = base_x[valid] + ROI_NEIGHBOR_DX[nbr] - roi_x0
        y = base_y[valid] + ROI_NEIGHBOR_DY[nbr] - roi_y0
        inside = (x >= 0) & (x < roi_size) & (y >= 0) & (y < roi_size)
        if np.any(inside):
            np.add.at(event_cnt, (channels[valid][inside], y[inside], x[inside]), weight[valid][inside])
    return event_cnt


def interpolate_poses(pose_mats: np.ndarray, pose_ts: np.ndarray, query_ts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """在指定时间戳插值 Cn_T_C0 位姿，并返回是否处于可插值范围。"""

    pose_ts = pose_ts.astype(np.float64)
    query_ts = np.asarray(query_ts, dtype=np.float64)
    idx = np.searchsorted(pose_ts, query_ts, side="left")
    valid = (idx > 0) & (idx < len(pose_ts))
    out = np.zeros((len(query_ts), 4, 4), dtype=np.float64)
    out[:, 3, 3] = 1.0

    if SE3 is not None and interpolate is not None:
        # 优先使用 EV-TTC 原始实现依赖的 Sophus 插值，保证几何口径一致。
        soph_pose = [SE3.from_matrix(pose_mats[i]) for i in range(len(pose_mats))]
        for i, ok in enumerate(valid):
            if not ok:
                continue
            j = int(idx[i])
            alpha = float((query_ts[i] - pose_ts[j - 1]) / (pose_ts[j] - pose_ts[j - 1]))
            out[i] = np.asarray(interpolate(soph_pose[j - 1], soph_pose[j], alpha).to_matrix())
        return out, valid

    # 兜底方案：环境缺少 projectaria_tools 时使用 scipy 的旋转插值。
    from scipy.spatial.transform import Rotation, Slerp

    rots = Rotation.from_matrix(pose_mats[:, :3, :3])
    for i, ok in enumerate(valid):
        if not ok:
            continue
        j = int(idx[i])
        alpha = float((query_ts[i] - pose_ts[j - 1]) / (pose_ts[j] - pose_ts[j - 1]))
        slerp = Slerp([0.0, 1.0], rots[[j - 1, j]])
        out[i, :3, :3] = slerp([alpha]).as_matrix()[0]
        out[i, :3, 3] = (1.0 - alpha) * pose_mats[j - 1, :3, 3] + alpha * pose_mats[j, :3, 3]
    return out, valid


def compute_relative_motion(
    pose_mats: np.ndarray,
    pose_ts: np.ndarray,
    t_start: np.ndarray,
    t_end: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """根据 t_start 到 t_end 的相对位姿计算平移速度 T 和角速度 Omega。"""

    start_pose, start_valid = interpolate_poses(pose_mats, pose_ts, t_start)
    end_pose, end_valid = interpolate_poses(pose_mats, pose_ts, t_end)
    valid = start_valid & end_valid & (t_end > t_start)
    rel = np.zeros((len(t_start), 4, 4), dtype=np.float64)
    rel[:, 3, 3] = 1.0
    T = np.zeros((len(t_start), 3), dtype=np.float32)
    Omega = np.zeros((len(t_start), 3), dtype=np.float32)

    for i, ok in enumerate(valid):
        if not ok:
            continue
        dt_sec = float((t_end[i] - t_start[i]) * US_TO_S)
        rel_i = start_pose[i] @ np.linalg.inv(end_pose[i])
        rel[i] = rel_i
        T[i] = (rel_i[:3, 3] / dt_sec).astype(np.float32)
        w_hat = logm(rel_i[:3, :3], disp=False)[0]
        Omega[i] = (np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]]) / dt_sec).real.astype(np.float32)
    return T, Omega, valid


def _min_splat_depth(
    x: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    height: int = 360,
    width: int = 360,
    splat: int = 3,
) -> np.ndarray:
    """用 min 聚合把稀疏深度点 splat 到 360x360 深度图。"""

    depth_img = np.full((height, width), np.inf, dtype=np.float32)
    xi = np.floor(x).astype(np.int32)
    yi = np.floor(y).astype(np.int32)
    valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height) & np.isfinite(depth) & (depth > 0)
    xi = xi[valid]
    yi = yi[valid]
    depth = depth[valid].astype(np.float32)
    if len(depth) == 0:
        depth_img[:] = np.nan
        return depth_img

    for dy in range(-splat, splat + 1):
        yy = yi + dy
        y_ok = (yy >= 0) & (yy < height)
        if not np.any(y_ok):
            continue
        for dx in range(-splat, splat + 1):
            xx = xi + dx
            ok = y_ok & (xx >= 0) & (xx < width)
            if np.any(ok):
                np.minimum.at(depth_img, (yy[ok], xx[ok]), depth[ok])
    depth_img[~np.isfinite(depth_img)] = np.nan
    return depth_img


def project_depth_to_event_frame(
    depth_raw: np.ndarray,
    depth_pose: np.ndarray,
    start_pose: np.ndarray,
    geom: CameraGeometry,
    splat: int = 3,
) -> np.ndarray:
    """把 M3ED depth_gt 重投影到当前事件相机起始时刻的 360x360 图像平面。"""

    # depth_gt 原本在对应深度帧坐标下，需要先变换到当前事件帧坐标。
    CN_T_DN = start_pose @ np.linalg.inv(depth_pose)
    flat_depth = depth_raw.reshape(-1)
    valid_pts = np.isfinite(flat_depth) & (flat_depth > 0)
    if not np.any(valid_pts):
        return np.full((360, 360), np.nan, dtype=np.float32)

    valid_depth = flat_depth[valid_pts].astype(np.float32)
    depth_cord = geom.proj_pts[:, valid_pts] * valid_depth[None, :]
    depth_cord_aug = np.vstack([depth_cord, np.ones_like(depth_cord[0])])
    event_cord = CN_T_DN @ depth_cord_aug
    in_front = event_cord[2] > 0
    if not np.any(in_front):
        return np.full((360, 360), np.nan, dtype=np.float32)

    event_cord = event_cord[:, in_front]
    valid_depth = valid_depth[in_front]
    img_cord = geom.K @ event_cord[:3]
    img_cord /= img_cord[2]

    # 与 EV-TTC 保持一致：中心裁剪后缩小 2 倍。
    crop = (
        (img_cord[0] > 280)
        & (img_cord[0] < 1000)
        & (img_cord[1] > 0)
        & (img_cord[1] < 720)
    )
    if not np.any(crop):
        return np.full((360, 360), np.nan, dtype=np.float32)

    x = (img_cord[0, crop] - 280.0) / 2.0
    y = img_cord[1, crop] / 2.0
    depth_splat = valid_depth[crop]
    return _min_splat_depth(x, y, depth_splat, splat=splat)


def compute_labels_for_steps(
    f_depth: h5py.File,
    pose_mats: np.ndarray,
    pose_ts: np.ndarray,
    geom: CameraGeometry,
    t_start: np.ndarray,
    t_end: np.ndarray,
    sequence_name: str,
    splat: int = 3,
    min_valid_pixels: int = 500,
) -> Dict[str, np.ndarray]:
    """为每个 1k-event step 生成 depth、TTC、inverse TTC、mask 和监督标志。"""

    depth_ts = f_depth["ts"][:].astype(np.int64)
    depth_poses = f_depth["Cn_T_C0"][:].astype(np.float64)
    depth_dset = f_depth["depth"]["prophesee"]["left"]

    T, Omega, pose_valid = compute_relative_motion(pose_mats, pose_ts, t_start, t_end)
    start_pose, start_pose_valid = interpolate_poses(pose_mats, pose_ts, t_start)
    depth_idx = np.searchsorted(depth_ts, t_start, side="left")
    depth_valid = (depth_idx >= 0) & (depth_idx < len(depth_ts))
    geom_valid = pose_valid & start_pose_valid & depth_valid

    n = len(t_start)
    depth_start = np.full((n, 360, 360), np.nan, dtype=np.float32)
    ttc_start = np.full((n, 360, 360), np.nan, dtype=np.float32)
    inv_ttc_start = np.zeros((n, 360, 360), dtype=np.float32)
    valid_mask = np.zeros((n, 360, 360), dtype=bool)

    for i in range(n):
        if not geom_valid[i]:
            continue
        d_idx = int(depth_idx[i])
        depth_img = project_depth_to_event_frame(
            depth_dset[d_idx], depth_poses[d_idx], start_pose[i], geom, splat=splat
        )
        ttc = depth_img / (T[i, 2] + 1e-5)
        # signed inverse TTC 保留远离/靠近两个方向；正TTC区域由 positive_ttc_mask 单独表达。
        inv_ttc = T[i, 2] / np.maximum(depth_img, 1e-6)
        mask = np.isfinite(depth_img) & (depth_img > 0) & np.isfinite(inv_ttc)

        depth_start[i] = depth_img
        ttc_start[i] = ttc.astype(np.float32)
        inv_ttc_start[i] = np.nan_to_num(inv_ttc, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        valid_mask[i] = mask

    # 时间步级筛选只写入标志，不修改 valid_ttc_mask，便于后续排查边界样本。
    speed_threshold = CAR_SPEED_THRESHOLD if "car" in sequence_name else SPOT_SPEED_THRESHOLD
    speed_valid = np.linalg.norm(T, axis=1) > speed_threshold
    omega_valid = np.linalg.norm(Omega, axis=1) < OMEGA_THRESHOLD
    supervise_valid = pose_valid & speed_valid & omega_valid
    direct_supervision_valid = supervise_valid & (valid_mask.reshape(n, -1).sum(axis=1) > 0)

    return {
        "depth_start": depth_start,
        "ttc_start": ttc_start,
        "inverse_ttc_start": inv_ttc_start,
        "valid_ttc_mask": valid_mask,
        "positive_ttc_mask": valid_mask & (inv_ttc_start > 0),
        "T": T,
        "Omega": Omega,
        "speed_valid": speed_valid.astype(bool),
        "omega_valid": omega_valid.astype(bool),
        "supervise_valid": supervise_valid.astype(bool),
        "direct_supervision_valid": direct_supervision_valid.astype(bool),
        "pose_valid": pose_valid.astype(bool),
        "depth_valid": depth_valid.astype(bool),
        "depth_source_index": depth_idx.astype(np.int64),
    }
