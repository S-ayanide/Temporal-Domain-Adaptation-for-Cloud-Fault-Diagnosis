"""
Shared constants and helpers for data preparation (Alibaba-only and Google→Alibaba).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

WINDOW_SIZE = 20
WINDOW_STEP = 5
MIN_SEQ_LEN = WINDOW_SIZE

FEATURE_COLS = [
    "cpu_util_percent",
    "mem_util_percent",
    "mem_gps",
    "net_in",
    "net_out",
    "disk_io_percent",
]

FAULT_NAMES = {
    0: "Normal",
    1: "CPU_Overload",
    2: "Memory_Leak",
    3: "Disk_IO_Fault",
    4: "Network_Fault",
    5: "Mixed_Fault",
}
N_CLASSES = len(FAULT_NAMES)


def compute_thresholds(df: pd.DataFrame) -> dict:
    return {
        "cpu": float(np.percentile(df["cpu_util_percent"].clip(0, 100), 85)),
        "mem": float(np.percentile(df["mem_util_percent"].clip(0, 100), 85)),
        "disk": float(np.percentile(df["disk_io_percent"].clip(0, 100), 85)),
        "net": float(np.percentile((df["net_in"] + df["net_out"]).clip(0, 200), 80)),
    }


def assign_labels(df: pd.DataFrame, thr: dict) -> np.ndarray:
    cpu = df["cpu_util_percent"] > thr["cpu"]
    mem = df["mem_util_percent"] > thr["mem"]
    disk = df["disk_io_percent"] > thr["disk"]
    net = (df["net_in"] + df["net_out"]) > thr["net"]
    multi = cpu.astype(int) + mem.astype(int) + disk.astype(int) + net.astype(int) >= 2
    labels = np.zeros(len(df), dtype=np.int64)
    labels[multi] = 5
    labels[cpu & ~multi] = 1
    labels[mem & ~multi] = 2
    labels[disk & ~multi] = 3
    labels[net & ~multi] = 4
    return labels


def make_windows(
    df: pd.DataFrame,
    thr: dict,
    window_size: int = WINDOW_SIZE,
    step: int = WINDOW_STEP,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequences, labels, machines = [], [], []
    feat_arr = df[FEATURE_COLS].values.astype(np.float32)
    lab_arr = assign_labels(df, thr)
    mid_arr = df["machine_id"].values

    for mid in df["machine_id"].unique():
        mask = mid_arr == mid
        idx = np.where(mask)[0]
        if len(idx) < window_size:
            continue
        f_m = feat_arr[idx]
        l_m = lab_arr[idx]
        for start in range(0, len(idx) - window_size + 1, step):
            end = start + window_size
            sequences.append(f_m[start:end])
            labels.append(l_m[end - 1])
            machines.append(mid)

    if not sequences:
        return (
            np.empty((0, window_size, len(FEATURE_COLS)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.array([]),
        )
    return (
        np.stack(sequences),
        np.array(labels, dtype=np.int64),
        np.array(machines),
    )


def node_types_from_windows(X_win: np.ndarray) -> np.ndarray:
    means = X_win.mean(axis=1)
    df_tmp = pd.DataFrame(means, columns=FEATURE_COLS)
    cpu_p70 = df_tmp["cpu_util_percent"].quantile(0.70)
    mem_p70 = df_tmp["mem_util_percent"].quantile(0.70)
    disk_p60 = df_tmp["disk_io_percent"].quantile(0.60)
    nt = []
    for _, r in df_tmp.iterrows():
        if r["cpu_util_percent"] > cpu_p70:
            nt.append("cpu_heavy")
        elif r["mem_util_percent"] > mem_p70:
            nt.append("mem_heavy")
        elif r["disk_io_percent"] > disk_p60:
            nt.append("io_heavy")
        else:
            nt.append("mixed")
    return np.array(nt)
