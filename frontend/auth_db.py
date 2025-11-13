# /home/uv-env/auth_db.py
import os
from contextlib import contextmanager
from typing import Optional, Tuple
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import pooling
from mysql.connector.errors import IntegrityError
import bcrypt

load_dotenv()

DB_CONFIG = {
    "host": os.environ.get("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.environ.get("MYSQL_PORT", "3306")),
    "user": os.environ.get("MYSQL_USER", "user1"),
    "password": os.environ.get("MYSQL_PASSWORD", "4321"),
    "database": os.environ.get("MYSQL_DB", "app"),
}
cnxpool = pooling.MySQLConnectionPool(pool_name="app_pool", pool_size=5, **DB_CONFIG)

@contextmanager
def get_conn():
    conn = cnxpool.get_connection()
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
          id INT AUTO_INCREMENT PRIMARY KEY,
          username VARCHAR(64) NOT NULL UNIQUE,
          password_hash VARBINARY(60) NOT NULL,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        conn.commit()

def username_exists(username: str) -> bool:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM users WHERE username=%s LIMIT 1", (username,))
        return cur.fetchone() is not None

def create_user(username: str, password: str) -> Tuple[bool, str]:
    if not username or not password:
        return False, "아이디/비밀번호를 입력하세요."
    if len(username) < 3:
        return False, "아이디는 3자 이상."
    if len(password) < 6:
        return False, "비밀번호는 6자 이상."
    if username_exists(username):
        return False, "이미 사용 중인 아이디."
    pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                (username, pw_hash),
            )
            conn.commit()
        return True, "회원가입 완료! 로그인하세요."
    except IntegrityError:
        return False, "이미 사용 중인 아이디."
    except Exception as e:
        return False, f"회원가입 오류: {e}"

def verify_user(username: str, password: str) -> Tuple[bool, Optional[int]]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, password_hash FROM users WHERE username=%s LIMIT 1", (username,))
        row = cur.fetchone()
        if not row:
            return False, None
        user_id, pw_hash = row
        try:
            ok = bcrypt.checkpw(password.encode("utf-8"), pw_hash)
        except Exception:
            ok = False
        return (ok, user_id if ok else None)
