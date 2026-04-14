from __future__ import annotations

from psycopg2.extras import Json

try:
    from utils.logger import get_logger
except ImportError:
    from backend.utils.logger import get_logger

from .common import _make_json_safe

logger = get_logger("database")


class ConversationMixin:
    def get_conversation_state(self, thread_id: str):
        try:
            with self.get_cursor(dict_cursor=True) as cur:
                cur.execute("""
                    SELECT thread_id, user_id, scene, summary, recent_messages, extra_state, updated_at
                    FROM conversation_states
                    WHERE thread_id = %s
                """, (thread_id,))
                return cur.fetchone()
        except Exception as e:
            logger.error(f"❌ 获取会话状态失败: {e}", exc_info=True)
            return None

    def get_conversation_state_for_user(self, user_id: str, thread_id: str, scene: str = "chat"):
        try:
            with self.get_cursor(dict_cursor=True) as cur:
                cur.execute("""
                    SELECT thread_id, user_id, scene, summary, recent_messages, extra_state, updated_at
                    FROM conversation_states
                    WHERE thread_id = %s AND user_id = %s AND scene = %s
                """, (thread_id, user_id, scene))
                return cur.fetchone()
        except Exception as e:
            logger.error("get conversation state for user failed: %s", e, exc_info=True)
            return None

    def list_conversation_states(self, user_id: str, scene: str = "chat", limit: int = 100):
        try:
            with self.get_cursor(dict_cursor=True) as cur:
                cur.execute("""
                    SELECT thread_id, user_id, scene, summary, recent_messages, extra_state, updated_at
                    FROM conversation_states
                    WHERE user_id = %s AND scene = %s
                    ORDER BY updated_at DESC
                    LIMIT %s
                """, (user_id, scene, limit))
                return cur.fetchall()
        except Exception as e:
            logger.error("list conversation states failed: %s", e, exc_info=True)
            return []

    def upsert_conversation_state(
        self,
        thread_id: str,
        user_id: str,
        scene: str,
        summary: str,
        recent_messages: list[dict],
        extra_state: dict | None = None,
    ) -> bool:
        try:
            with self.get_cursor() as cur:
                cur.execute("""
                    INSERT INTO conversation_states
                        (thread_id, user_id, scene, summary, recent_messages, extra_state, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (thread_id)
                    DO UPDATE SET
                        user_id = EXCLUDED.user_id,
                        scene = EXCLUDED.scene,
                        summary = EXCLUDED.summary,
                        recent_messages = EXCLUDED.recent_messages,
                        extra_state = EXCLUDED.extra_state,
                        updated_at = CURRENT_TIMESTAMP;
                """, (
                    thread_id,
                    user_id,
                    scene,
                    summary or "",
                    Json(_make_json_safe(recent_messages or [])),
                    Json(_make_json_safe(extra_state or {})),
                ))
                return True
        except Exception as e:
            logger.error(f"❌ 保存会话状态失败: {e}", exc_info=True)
            return False

    def delete_conversation_state(self, thread_id: str) -> bool:
        try:
            with self.get_cursor() as cur:
                cur.execute("DELETE FROM conversation_states WHERE thread_id = %s", (thread_id,))
                return True
        except Exception as e:
            logger.error(f"❌ 删除会话状态失败: {e}", exc_info=True)
            return False

    def delete_conversation_state_for_user(self, user_id: str, thread_id: str, scene: str = "chat") -> bool:
        try:
            with self.get_cursor() as cur:
                cur.execute(
                    "DELETE FROM conversation_states WHERE thread_id = %s AND user_id = %s AND scene = %s",
                    (thread_id, user_id, scene),
                )
                return cur.rowcount > 0
        except Exception as e:
            logger.error("delete conversation state for user failed: %s", e, exc_info=True)
            return False
