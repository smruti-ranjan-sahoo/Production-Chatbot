import sqlite3
from datetime import datetime


class ChatDatabase:
    def __init__(self, db_path="chat.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            title TEXT,
            created_at TEXT
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            role TEXT,
            content TEXT,
            created_at TEXT
        )
        """)

        self.conn.commit()

    # -------------------------
    # Conversation
    # -------------------------

    def create_conversation(self, conversation_id, user_id):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO conversations VALUES (?, ?, ?, ?)",
            (conversation_id, user_id, "New Chat", datetime.utcnow().isoformat())
        )
        self.conn.commit()

    def get_conversations(self, user_id):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, title FROM conversations WHERE user_id=? ORDER BY created_at DESC",
            (user_id,)
        )
        return cursor.fetchall()

    # -------------------------
    # Messages
    # -------------------------

    def add_message(self, conversation_id, role, content):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, datetime.utcnow().isoformat())
        )
        self.conn.commit()

    def get_messages(self, conversation_id):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id",
            (conversation_id,)
        )
        return cursor.fetchall()