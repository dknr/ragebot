#!/usr/bin/env python3
"""
SQLite storage for RageBot statistics
"""

import sqlite3
import os
import logging
from datetime import datetime, timedelta
from contextlib import contextmanager

DB_PATH = "ragebot.db"
_db_connection = None

def get_db_connection():
    """Get a database connection (singleton pattern)"""
    global _db_connection
    if _db_connection is None:
        _db_connection = sqlite3.connect(DB_PATH, check_same_thread=False)
        _db_connection.row_factory = sqlite3.Row
        logging.info("Database connection initialized")
    return _db_connection

@contextmanager
def get_db_cursor():
    """Context manager for database operations with automatic commit/rollback"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        yield cursor
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"Database transaction failed: {e}")
        raise
    finally:
        cursor.close()

def close_db_connection():
    """Close the database connection"""
    global _db_connection
    if _db_connection is not None:
        _db_connection.close()
        _db_connection = None
        logging.info("Database connection closed")

def init_db():
    """Initialize database schema"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER NOT NULL,
        message_id INTEGER NOT NULL,
        sentiment_label TEXT NOT NULL,
        sentiment_score REAL NOT NULL,
        reaction TEXT,
        lol_count INTEGER DEFAULT 0,
        is_neat INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Migrate: add is_neat column if it doesn't exist
    cursor.execute("PRAGMA table_info(messages)")
    columns = [row[1] for row in cursor.fetchall()]
    if 'is_neat' not in columns:
        cursor.execute("ALTER TABLE messages ADD COLUMN is_neat INTEGER DEFAULT 0")
        logging.info("Added is_neat column to messages table")

    # Migrate: add lol_count column if it doesn't exist
    cursor.execute("PRAGMA table_info(messages)")
    columns = [row[1] for row in cursor.fetchall()]
    if 'lol_count' not in columns:
        cursor.execute("ALTER TABLE messages ADD COLUMN lol_count INTEGER DEFAULT 0")
        logging.info("Added lol_count column to messages table")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS thresholds (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sentiment TEXT NOT NULL,
        threshold REAL NOT NULL,
        reaction TEXT NOT NULL,
        UNIQUE(sentiment, threshold)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS flags (
        flag TEXT PRIMARY KEY,
        value INTEGER NOT NULL
    )
    """)

    # Initialize thresholds on first run
    value = get_flag('thresholds_firstrun')
    if value == 0 or value is None:
        # Get default thresholds from defaults module
        try:
            from defaults import EMOJI_THRESHOLDS

            for sentiment, thresholds in EMOJI_THRESHOLDS.items():
                for threshold, reaction in thresholds:
                    cursor.execute("""
                    INSERT OR IGNORE INTO thresholds (sentiment, threshold, reaction)
                    VALUES (?, ?, ?)
                    """, (sentiment, threshold, reaction))
        except ImportError:
            pass  # EMOJI_THRESHOLDS not available yet

        # Set first run flag
        cursor.execute("""
        INSERT OR REPLACE INTO flags (flag, value)
        VALUES ('thresholds_firstrun', 1)
        """)

    conn.commit()

def add_message(chat_id, message_id, sentiment_label, sentiment_score, reaction=None, lol_count=0, is_neat=False):
    """Add a message to the database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO messages (chat_id, message_id, sentiment_label, sentiment_score, reaction, lol_count, is_neat)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (chat_id, message_id, sentiment_label, sentiment_score, reaction, lol_count, 1 if is_neat else 0))
        conn.commit()
    finally:
        cursor.close()

def get_vibecheck(chat_id, hours=24):
    """Get statistics for the last N hours"""
    conn = get_db_connection()
    cursor = conn.cursor()

    since = datetime.now() - timedelta(hours=hours)

    cursor.execute("""
    SELECT
        COUNT(*) as message_count,
        SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive,
        SUM(CASE WHEN sentiment_label = 'neutral' THEN 1 ELSE 0 END) as neutral,
        SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative,
        SUM(lol_count) as total_lol_count,
        SUM(CASE WHEN lol_count > 0 THEN 1 ELSE 0 END) as messages_with_lol,
        SUM(CASE WHEN is_neat = 1 THEN 1 ELSE 0 END) as neat_count
    FROM messages
    WHERE chat_id = ?
        AND created_at >= ?
    """, (chat_id, since.strftime('%Y-%m-%d %H:%M:%S')))

    result = cursor.fetchone()
    cursor.close()

    return {
        'messageCount': result['message_count'],
        'positive': result['positive'],
        'neutral': result['neutral'],
        'negative': result['negative'],
        'total_lol_count': result['total_lol_count'],
        'messages_with_lol': result['messages_with_lol'],
        'neat_count': result['neat_count']
    }

def format_vibecheck(vibecheck_data):
    """Format vibecheck statistics"""
    lines = []
    lines.append("📊 Vibecheck for last 24 hours")

    if vibecheck_data['messageCount'] > 0:
        total = vibecheck_data['messageCount']
        positive_pct = (vibecheck_data['positive'] / total) * 100 if total > 0 else 0
        neutral_pct = (vibecheck_data['neutral'] / total) * 100 if total > 0 else 0
        negative_pct = (vibecheck_data['negative'] / total) * 100 if total > 0 else 0
        total_lol_count = vibecheck_data.get('total_lol_count', 0)
        messages_with_lol = vibecheck_data.get('messages_with_lol', 0)
        messages_with_lol_pct = (messages_with_lol / total) * 100 if total > 0 else 0
        neat_count = vibecheck_data.get('neat_count', 0)
        neat_pct = (neat_count / total) * 100 if total > 0 else 0

        lines.append("")
        lines.append(f"Messages: {total}")
        lines.append("")
        lines.append(f"Positive: {positive_pct:.1f}%")
        lines.append(f"Neutral: {neutral_pct:.1f}%")
        lines.append(f"Negative: {negative_pct:.1f}%")
        lines.append(f"LOL: {total_lol_count} total ({messages_with_lol} messages, {messages_with_lol_pct:.1f}%)")
        lines.append(f"NEAT: {neat_count} messages ({neat_pct:.1f}%)")
    else:
        lines.append("")
        lines.append("No messages in the last 24 hours")

    return "\n".join(lines)

def get_thresholds():
    """Get all thresholds from database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        SELECT sentiment, threshold, reaction
        FROM thresholds
        ORDER BY sentiment, threshold DESC
        """)
        thresholds = {}
        for row in cursor.fetchall():
            sentiment = row['sentiment']
            if sentiment not in thresholds:
                thresholds[sentiment] = []
            thresholds[sentiment].append((row['threshold'], row['reaction']))
        return thresholds
    finally:
        cursor.close()

def update_threshold(sentiment, threshold, reaction):
    """Update or insert a threshold"""
    with get_db_cursor() as cursor:
        cursor.execute("""
        INSERT OR REPLACE INTO thresholds (sentiment, threshold, reaction)
        VALUES (?, ?, ?)
        """, (sentiment, threshold, reaction))

def update_flag(flag, value):
    """Update or insert a flag value"""
    with get_db_cursor() as cursor:
        cursor.execute("""
        INSERT OR REPLACE INTO flags (flag, value)
        VALUES (?, ?)
        """, (flag, value))

def get_flag(flag):
    """Get a flag value"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT value FROM flags WHERE flag = ?", (flag,))
    result = cursor.fetchone()
    cursor.close()

    return result['value'] if result else None

def delete_old_messages(days=30):
    """Delete messages older than N days"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        DELETE FROM messages
        WHERE created_at < datetime('now', '-' || ? || ' days')
        """, (days,))
        deleted_count = cursor.rowcount
        conn.commit()
        return deleted_count
    finally:
        cursor.close()

def get_sentiment_trend(chat_id, hours=168, interval_hours=1):
    """Get sentiment counts over time for trend analysis
    Returns list of dicts with timestamp_bucket, positive, neutral, negative, total
    Uses 1-hour buckets; X-axis labels formatted at noon/midnight intervals
    Includes all buckets in time range, even those with zero messages
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Calculate the time range
    since = datetime.now() - timedelta(hours=hours)
    
    # Get actual data from database
    cursor.execute("""
    SELECT
        strftime('%Y-%m-%d %H:00:00', created_at) as bucket,
        SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive,
        SUM(CASE WHEN sentiment_label = 'neutral' THEN 1 ELSE 0 END) as neutral,
        SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative,
        COUNT(*) as total
    FROM messages
    WHERE chat_id = ?
        AND created_at >= ?
    GROUP BY bucket
    ORDER BY bucket
    """, (chat_id, since.strftime('%Y-%m-%d %H:%M:%S')))

    # Build a dict of bucket -> data for quick lookup
    data_by_bucket = {}
    for row in cursor.fetchall():
        data_by_bucket[row['bucket']] = {
            'timestamp': row['bucket'],
            'positive': row['positive'],
            'neutral': row['neutral'],
            'negative': row['negative'],
            'total': row['total']
        }

    cursor.close()
    
    # Generate all hourly buckets in the time range
    results = []
    current = since.replace(minute=0, second=0, microsecond=0)
    while current <= datetime.now():
        bucket_str = current.strftime('%Y-%m-%d %H:00:00')
        if bucket_str in data_by_bucket:
            results.append(data_by_bucket[bucket_str])
        else:
            # Add zero bucket
            results.append({
                'timestamp': bucket_str,
                'positive': 0,
                'neutral': 0,
                'negative': 0,
                'total': 0
            })
        current += timedelta(hours=1)
    
    return results