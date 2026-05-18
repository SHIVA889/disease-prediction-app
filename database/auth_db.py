import sqlite3
from datetime import datetime

from werkzeug.security import check_password_hash, generate_password_hash

from database.db import get_connection


class AuthError(Exception):
    pass


def init_auth_table():
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.commit()


def create_user(username, password):
    username = username.strip()

    if len(username) < 3:
        raise AuthError("Username must be at least 3 characters long.")

    if len(password) < 4:
        raise AuthError("Password must be at least 4 characters long.")

    password_hash = generate_password_hash(password)

    try:
        with get_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO users (username, password_hash, created_at)
                VALUES (?, ?, ?)
                """,
                (
                    username,
                    password_hash,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
            connection.commit()
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError as error:
        raise AuthError("Username already exists. Please choose another one.") from error

    return get_user_by_id(user_id)


def authenticate_user(username, password):
    username = username.strip()
    if not username or not password:
        return None

    row = get_user_auth_record(username)

    if row is None:
        return None

    if not check_password_hash(row["password_hash"], password):
        return None

    return {
        "id": row["id"],
        "username": row["username"],
        "created_at": row["created_at"],
    }


def get_user_auth_record(username):
    with get_connection() as connection:
        return connection.execute(
            "SELECT id, username, password_hash, created_at FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()


def get_user_by_id(user_id):
    with get_connection() as connection:
        row = connection.execute(
            "SELECT id, username, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()

    if row is None:
        return None

    return {
        "id": row["id"],
        "username": row["username"],
        "created_at": row["created_at"],
    }
