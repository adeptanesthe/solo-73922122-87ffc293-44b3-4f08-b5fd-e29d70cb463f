"""GET /ranking 的一致数据快照、SQL 计票聚合与进程内缓存。

性能与并发约定
--------------
- 实收答卷在数据库端按 (左模型, 右模型, 编辑) ``GROUP BY`` 汇总左右胜票数，
  Python 不逐份加载答卷；每种组合在设计矩阵中只占一行，拟合时以总票数为权重，
  Bradley–Terry 似然仍按每一票计入（见 :func:`app.bt.fit_bradley_terry`）。
- 一次排名所用的名单、计数、秩与设计矩阵都在同一个只读事务中读出，保证来自
  同一份已提交数据；读出后立即结束事务，拟合期间不持有数据库读锁，
  收卷/登记/派发/撤回/重置可照常写入。
- 进程内数据未变时，按数据库引擎（bind）复用快照中的票数与秩；调整 ``l2``
  只重新拟合。任何 Session 成功提交后缓存自动失效（version 递增），
  因此提交完成后发起的查询一定读到新状态；拟合中的在途请求继续使用旧快照。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sqlalchemy import case, event, func, select
from sqlalchemy.orm import Session

from .bt import ParamSpace, build_param_space, design_matrix, matrix_rank
from .models import COMPLETED, ORIGIN_PLAN, PENDING, Editor, Match, Model

Row = tuple[int, int, int]


# ---------------------------------------------------------------- 计票聚合

@dataclass(frozen=True)
class VoteGroup:
    """一种 (左模型, 右模型, 编辑) 组合的汇总票数（数据库 GROUP BY 的一行）。"""

    left_model_id: int
    right_model_id: int
    editor_id: int
    left_wins: int
    right_wins: int
    n_votes: int

    @property
    def row(self) -> Row:
        return (self.left_model_id, self.right_model_id, self.editor_id)


@dataclass
class RankingSnapshot:
    """一次排名所需的全部只读数据，来自同一个已提交快照。"""

    ps: ParamSpace
    reference: int | None
    model_ids: list[int]
    editor_ids: list[int]
    model_names: list[tuple[int, str]]
    editor_names: list[tuple[int, str]]
    groups: list[VoteGroup]
    n_responses: int
    pending_rows: list[Row]
    free_manuscripts: list[str]
    X: np.ndarray = field(repr=False, default=None)  # 每种组合一行
    rank: int = 0

    @property
    def received_rows(self) -> list[Row]:
        """实收答卷的去重组合行（重复组合不改变秩与零空间）。"""
        return [g.row for g in self.groups]

    @property
    def n_pending(self) -> int:
        return len(self.pending_rows)


def free_manuscripts(db: Session) -> list[str]:
    """空闲原稿 = 无待交场次 且 未被补发过（每篇原稿最多补一场）。

    已撤回的补发不占名额（撤回后可重派）。用一条 GROUP BY/HAVING 在库端完成，
    只返回空闲原稿号，不把全部场次读到 Python。
    """
    pending_flags = func.sum(case((Match.status == PENDING, 1), else_=0))
    supplemented_flags = func.sum(
        case(
            ((Match.origin == ORIGIN_PLAN) & Match.status.in_((PENDING, COMPLETED)), 1),
            else_=0,
        )
    )
    stmt = (
        select(Match.manuscript_id)
        .group_by(Match.manuscript_id)
        .having(pending_flags == 0)
        .having(supplemented_flags == 0)
        .order_by(Match.manuscript_id)
    )
    return [str(ms) for ms in db.scalars(stmt).all()]


def _load_snapshot(db: Session) -> RankingSnapshot:
    """在当前事务内一次性读出排名所需的全部数据（调用方负责结束事务）。"""
    model_names = [
        (int(mid), name)
        for mid, name in db.execute(select(Model.id, Model.name).order_by(Model.id)).all()
    ]
    editor_names = [
        (int(eid), name)
        for eid, name in db.execute(select(Editor.id, Editor.name).order_by(Editor.id)).all()
    ]
    model_ids = [mid for mid, _ in model_names]
    editor_ids = [eid for eid, _ in editor_names]
    reference = model_ids[0] if model_ids else None  # 第一个登记的模型为 0 分参照
    ps = build_param_space(model_ids, editor_ids, reference)

    # 实收答卷：数据库端按 左模型/右模型/编辑 汇总左右胜票数，不逐份取答卷。
    left_wins = func.coalesce(func.sum(case((Match.winner == "left", 1), else_=0)), 0)
    right_wins = func.coalesce(func.sum(case((Match.winner == "right", 1), else_=0)), 0)
    stmt = (
        select(
            Match.left_model_id,
            Match.right_model_id,
            Match.editor_id,
            left_wins,
            right_wins,
            func.count(Match.id),
        )
        .where(Match.status == COMPLETED)
        .group_by(Match.left_model_id, Match.right_model_id, Match.editor_id)
        .order_by(Match.left_model_id, Match.right_model_id, Match.editor_id)
    )
    groups: list[VoteGroup] = []
    n_responses = 0
    for left_id, right_id, editor_id, lw, rw, n_votes in db.execute(stmt).all():
        g = VoteGroup(
            left_model_id=int(left_id),
            right_model_id=int(right_id),
            editor_id=int(editor_id),
            left_wins=int(lw),
            right_wins=int(rw),
            n_votes=int(n_votes),
        )
        groups.append(g)
        n_responses += g.n_votes

    # 待交场次（在途、数量小）：只取排程需要的三个 id。
    pending_rows = [
        (int(l), int(r), int(e))
        for l, r, e in db.execute(
            select(Match.left_model_id, Match.right_model_id, Match.editor_id)
            .where(Match.status == PENDING)
            .order_by(Match.id)
        ).all()
    ]

    X = design_matrix([g.row for g in groups], ps)
    rank = matrix_rank(X)

    # 空闲原稿只在缺秩构造补发计划时才需要；满秩热路径省去这次分组查询。
    # 仍在同一只读事务内读取，保证与上面的计数/秩来自同一份快照。
    deficient = ps.n_params > 0 and rank < ps.n_params
    free_ms = free_manuscripts(db) if deficient else []

    return RankingSnapshot(
        ps=ps,
        reference=reference,
        model_ids=model_ids,
        editor_ids=editor_ids,
        model_names=model_names,
        editor_names=editor_names,
        groups=groups,
        n_responses=n_responses,
        pending_rows=pending_rows,
        free_manuscripts=free_ms,
        X=X,
        rank=rank,
    )


# ---------------------------------------------------------------- 进程内缓存

@dataclass
class _Entry:
    version: int = 0
    snapshot: RankingSnapshot | None = None


class RankingCache:
    """按数据库引擎隔离的票数/秩快照缓存。

    写事务提交后 version 递增并清空快照；version 同时用于防止加载期间有写入
    提交时把旧快照重新发布回缓存。锁只保护字典读写，绝不在拟合期间持有。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[Any, _Entry] = {}
        self.hits = 0
        self.misses = 0

    def _entry(self, bind: Any) -> _Entry:
        entry = self._entries.get(bind)
        if entry is None:
            entry = _Entry()
            self._entries[bind] = entry
        return entry

    def get(self, bind: Any) -> tuple[RankingSnapshot | None, int]:
        with self._lock:
            entry = self._entries.get(bind)
            if entry is not None and entry.snapshot is not None:
                self.hits += 1
                return entry.snapshot, entry.version
            version = entry.version if entry is not None else 0
            self.misses += 1
            return None, version

    def publish(self, bind: Any, version: int, snapshot: RankingSnapshot) -> RankingSnapshot | None:
        """版本未变则发布快照；期间发生过写入（version 变了）则返回 None。"""
        with self._lock:
            entry = self._entry(bind)
            if entry.version != version:
                return None
            if entry.snapshot is None:
                entry.snapshot = snapshot
            return entry.snapshot

    def note_change(self, bind: Any) -> None:
        with self._lock:
            entry = self._entry(bind)
            entry.version += 1
            entry.snapshot = None

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"hits": self.hits, "misses": self.misses, "engines": len(self._entries)}


ranking_cache = RankingCache()


def get_ranking_snapshot(db: Session) -> RankingSnapshot:
    """返回当前请求可用的排名快照：优先复用进程内缓存，否则从事务一致读出。"""
    bind = db.bind
    cached, version = ranking_cache.get(bind)
    if cached is not None:
        return cached

    snapshot = _load_snapshot(db)
    # 只读事务到此结束：立即释放 SQLite 读锁，后面的 BT 拟合可能较久，
    # 不能让写操作等待本请求的读事务。
    db.rollback()

    published = ranking_cache.publish(bind, version, snapshot)
    return published if published is not None else snapshot


# 任何 Session 成功提交（API 的登记/收卷/派发/撤回/apply/reset 都在此列）后
# 自动令对应引擎的缓存失效。放在 after_commit 中可保证：调用方收到写入成功
# 的响应之前，新查询必然缓存未命中并读到已提交的新状态。
@event.listens_for(Session, "after_commit")
def _invalidate_ranking_cache(session: Session) -> None:
    ranking_cache.note_change(session.bind)
