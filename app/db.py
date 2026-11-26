"""数据库引擎与会话。"""
from __future__ import annotations

import os

from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./bt.db")

is_sqlite = DATABASE_URL.startswith("sqlite")
# timeout/busy_timeout：排名拟合期间虽不持锁，但并发写入在 SQLite 上仍可能短暂撞锁，
# 等待而不是立刻报 "database is locked"。
connect_args = {"check_same_thread": False, "timeout": 30.0} if is_sqlite else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
Base = declarative_base()

# 文件型 SQLite 开 WAL：读写互不阻塞、一个读事务内是一致快照，适合“拟合期间照常
# 收卷/写入”的并发要求。已有 SQLite 库可直接打开，切 WAL 是兼容的在线操作。
# 内存库不支持 WAL；若数据库所在目录只读导致 WAL 无法创建，退回默认回滚日志模式。
if is_sqlite and ":memory:" not in DATABASE_URL:

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA busy_timeout=30000")
            try:
                cursor.execute("PRAGMA journal_mode=WAL")
            except Exception:
                # 只读目录等场景下 WAL 建不起来时，保持默认日志模式仍可工作
                pass
        finally:
            cursor.close()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
