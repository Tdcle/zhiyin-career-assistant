from __future__ import annotations

import psycopg2

try:
    from utils.auth import hash_password, verify_password
    from utils.logger import get_logger
except ImportError:
    from backend.utils.auth import hash_password, verify_password
    from backend.utils.logger import get_logger

logger = get_logger("database")


class UserMixin:
    def _seed_default_users(self):
        with self.get_cursor() as cur:
            cur.execute("SELECT count(*) FROM users")
            if cur.fetchone()[0] == 0:
                cur.execute(
                    "INSERT INTO users (user_id, username) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    ("admin", "管理员"),
                )
                logger.info("default user ensured")

    def _generate_next_user_id(self, cur):
        cur.execute(r"SELECT user_id FROM users WHERE user_id ~ '^\d+$'")
        rows = cur.fetchall()
        if not rows:
            return "00001"
        max_id = 0
        for row in rows:
            try:
                uid = int(row[0]) if not isinstance(row, dict) else int(row["user_id"])
                max_id = max(max_id, uid)
            except (ValueError, TypeError):
                continue
        return f"{max_id + 1:05d}"

    def create_user(self, username: str):
        if not username or not username.strip():
            return False, "用户名不能为空"
        try:
            with self.get_cursor() as cur:
                new_user_id = self._generate_next_user_id(cur)
                cur.execute(
                    "INSERT INTO users (user_id, username) VALUES (%s, %s)",
                    (new_user_id, username.strip()),
                )
                logger.info(f"✅ 用户创建成功: ID={new_user_id}, 昵称={username}")
                return True, f"用户创建成功! ID: {new_user_id} | 昵称: {username}"
        except psycopg2.IntegrityError:
            return False, "创建失败: 用户已存在"
        except Exception as e:
            logger.error(f"❌ 用户创建失败: {e}", exc_info=True)
            return False, f"创建失败: {str(e)}"

    def get_user_by_username(self, username: str):
        if not username or not username.strip():
            return None
        try:
            with self.get_cursor(dict_cursor=True) as cur:
                cur.execute(
                    """
                    SELECT user_id, username
                    FROM users
                    WHERE username = %s
                    ORDER BY created_at ASC
                    LIMIT 1
                    """,
                    (username.strip(),),
                )
                return cur.fetchone()
        except Exception as e:
            logger.error("get user by username failed: %s", e, exc_info=True)
            return None

    def create_user_with_password(self, username: str, password: str):
        username = (username or "").strip()
        password = (password or "").strip()
        if not username:
            return False, "username is required"
        if len(password) < 6:
            return False, "password must be at least 6 characters"

        try:
            with self.get_cursor(dict_cursor=True) as cur:
                cur.execute(
                    """
                    SELECT u.user_id, u.username, a.user_id AS auth_user_id
                    FROM users u
                    LEFT JOIN auth_users a ON a.user_id = u.user_id
                    WHERE u.username = %s
                    ORDER BY u.created_at ASC
                    LIMIT 1
                    """,
                    (username,),
                )
                existing = cur.fetchone()
                if existing and existing.get("auth_user_id"):
                    return False, "username already exists"

                if existing:
                    user_id = existing["user_id"]
                    cur.execute(
                        """
                        INSERT INTO auth_users (user_id, password_hash, is_active)
                        VALUES (%s, %s, TRUE)
                        ON CONFLICT (user_id)
                        DO UPDATE SET
                            password_hash = EXCLUDED.password_hash,
                            is_active = TRUE,
                            updated_at = CURRENT_TIMESTAMP
                        """,
                        (user_id, hash_password(password)),
                    )
                    return True, {"user_id": user_id, "username": existing["username"]}

                user_id = self._generate_next_user_id(cur)
                cur.execute(
                    "INSERT INTO users (user_id, username) VALUES (%s, %s)",
                    (user_id, username),
                )
                cur.execute(
                    """
                    INSERT INTO auth_users (user_id, password_hash, is_active)
                    VALUES (%s, %s, TRUE)
                    """,
                    (user_id, hash_password(password)),
                )
                return True, {"user_id": user_id, "username": username}
        except Exception as e:
            logger.error("create user with password failed: %s", e, exc_info=True)
            return False, str(e)

    def authenticate_user(self, username: str, password: str):
        username = (username or "").strip()
        if not username or not password:
            return None

        try:
            with self.get_cursor(dict_cursor=True) as cur:
                cur.execute(
                    """
                    SELECT u.user_id, u.username, a.password_hash, a.is_active
                    FROM users u
                    JOIN auth_users a ON a.user_id = u.user_id
                    WHERE u.username = %s
                    ORDER BY u.created_at ASC
                    LIMIT 1
                    """,
                    (username,),
                )
                row = cur.fetchone()
        except Exception as e:
            logger.error("authenticate user query failed: %s", e, exc_info=True)
            return None

        if not row or not row.get("is_active"):
            return None
        if not verify_password(password, row.get("password_hash", "")):
            return None
        return {"user_id": row["user_id"], "username": row["username"]}

    def change_user_password(self, user_id: str, current_password: str, new_password: str):
        user_id = (user_id or "").strip()
        current_password = (current_password or "").strip()
        new_password = (new_password or "").strip()

        if not user_id:
            return False, "user not found"
        if len(current_password) < 6:
            return False, "current password is invalid"
        if len(new_password) < 6:
            return False, "new password must be at least 6 characters"
        if current_password == new_password:
            return False, "new password must be different from current password"

        try:
            with self.get_cursor(dict_cursor=True) as cur:
                cur.execute(
                    """
                    SELECT password_hash, is_active
                    FROM auth_users
                    WHERE user_id = %s
                    LIMIT 1
                    """,
                    (user_id,),
                )
                row = cur.fetchone()
                if not row or not row.get("is_active"):
                    return False, "user account is unavailable"

                if not verify_password(current_password, row.get("password_hash", "")):
                    return False, "current password is incorrect"

                cur.execute(
                    """
                    UPDATE auth_users
                    SET password_hash = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = %s
                    """,
                    (hash_password(new_password), user_id),
                )
                return True, "password changed"
        except Exception as e:
            logger.error("change user password failed: %s", e, exc_info=True)
            return False, str(e)

    def get_all_users(self):
        try:
            with self.get_cursor(dict_cursor=True) as cur:
                cur.execute("SELECT user_id, username FROM users ORDER BY user_id DESC")
                return cur.fetchall()
        except Exception as e:
            logger.error(f"failed to get users: {e}", exc_info=True)
            return []

    def get_all_users_list(self):
        try:
            with self.get_cursor() as cur:
                cur.execute("SELECT user_id, username FROM users ORDER BY user_id DESC")
                return [f"{u[0]} ({u[1]})" for u in cur.fetchall()]
        except Exception as e:
            logger.error(f"❌ 获取用户列表失败: {e}")
            return []
