"""译文两两盲评平台。

并发排名时会有多个小型 Bradley–Terry 拟合同时进行；让 BLAS/OpenMP 只开一个
计算线程，可避免小矩阵上的线程池过度订阅与 CPU 配额抖动。仅在外部未设置时
填默认值，且必须在 numpy/scipy 被导入前生效。
"""
from __future__ import annotations

import os

for _var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")
