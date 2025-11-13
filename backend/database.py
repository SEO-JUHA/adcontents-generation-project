# backend/database.py
from __future__ import annotations

import os
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, JSON,
    TIMESTAMP, func, ForeignKey, LargeBinary
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

# -----------------------------
# DB 연결 (PyMySQL 드라이버 사용)
# -----------------------------
# .env 가 있다면 여기서 읽어도 됩니다. (없으면 기본값 사용)
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_USER = os.getenv("MYSQL_USER", "user1")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "4321")
MYSQL_DB = os.getenv("MYSQL_DB", "app")

# 반드시 pymysql 드라이버 사용
DATABASE_URL = (
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
    "?charset=utf8mb4"
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    future=True,
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)
Base = declarative_base()

# -----------------------------
# 인스타그램 포스트 모델 (기존)
# -----------------------------
class GeneratedPost(Base):
    __tablename__ = "generated_posts"

    id = Column(Integer, primary_key=True, index=True)
    brand_persona = Column(Text)
    product_info = Column(Text)
    store_address = Column(String(255))
    target_audience = Column(String(255))
    generated_captions = Column(JSON)
    generated_hashtags = Column(JSON)
    engagement_prediction = Column(JSON)

# -----------------------------
# users / logos 모델
#  - logos.user_id -> users.id (FK)
#  - users 테이블은 이미 존재(Show Create Table 기준)하므로 정의만 일치시키면 됩니다.
# -----------------------------
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(64), unique=True, nullable=False)
    # 실제 MySQL 타입은 VARBINARY(60) 이었으므로 LargeBinary(60)로 매핑
    password_hash = Column(LargeBinary(60), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=True)

    logos = relationship("Logo", back_populates="user", cascade="all, delete-orphan")

class Logo(Base):
    __tablename__ = "logos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    image_path = Column(String(512), nullable=False)  # 예: logos/3/uuid.png
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)

    user = relationship("User", back_populates="logos")


# ---cartoon---
class UsersData(Base):
    """
    ComfyUI 작업 기록 (client_id/prompt_id/status)
    """
    __tablename__ = "usersdata"
    id = Column(Integer, primary_key=True, autoincrement=True)
    client_id = Column(String(64), nullable=True, index=True)
    prompt_id = Column(String(64), nullable=True, index=True)
    status    = Column(String(32), nullable=True, index=True)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)

class FourcutsImage(Base):
    """
    생성된 4컷 이미지 파일 경로 기록
    """
    __tablename__ = "fourcuts_image"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL", onupdate="CASCADE"), nullable=True)
    image_path = Column(Text, nullable=False)  # 예: 'outputs/fourcuts/2025-10-10/uuid.png'
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    updated_at = Column(
        TIMESTAMP, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    user = relationship("User", backref="fourcuts_images")
