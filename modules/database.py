import sqlite3
import pandas as pd
from datetime import datetime

DB_FILE = "fairmeet.db"


def get_connection():
    return sqlite3.connect(DB_FILE, check_same_thread=False)


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS meetings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meeting_name TEXT,
            analysis_date TEXT,
            fairness_score REAL,
            agenda_score REAL,
            engagement_score REAL,
            verdict TEXT
        )
        """
    )

    cur.execute("PRAGMA table_info(meetings)")
    columns = [row[1] for row in cur.fetchall()]

    required_columns = {
        "meeting_name",
        "analysis_date",
        "fairness_score",
        "agenda_score",
        "engagement_score",
        "verdict",
    }

    if not required_columns.issubset(set(columns)):
        cur.execute("DROP TABLE IF EXISTS meetings")
        cur.execute(
            """
            CREATE TABLE meetings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_name TEXT,
                analysis_date TEXT,
                fairness_score REAL,
                agenda_score REAL,
                engagement_score REAL,
                verdict TEXT
            )
            """
        )

    conn.commit()
    conn.close()


def save_meeting(name, fairness, agenda, engagement, verdict):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO meetings (
            meeting_name, analysis_date, fairness_score,
            agenda_score, engagement_score, verdict
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            name,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            float(fairness),
            float(agenda),
            float(engagement),
            verdict,
        ),
    )

    conn.commit()
    conn.close()


def get_history():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM meetings ORDER BY id DESC", conn)
    conn.close()
    return df


def delete_meeting(meeting_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
    conn.commit()
    conn.close()