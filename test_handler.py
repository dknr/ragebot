#!/usr/bin/env python3
"""
Test RageBot handler directly without Telegram connection
"""

import logging
import sys
from unittest.mock import Mock

from telegram import Update
from telegram.ext import ContextTypes

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
    stream=sys.stdout
)

async def minimal_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Minimal handler that logs minimal info"""
    logging.info("=" * 60)
    logging.info("Message received")
    logging.info(f"User: {update.effective_user.username}")
    logging.info(f"Chat ID: {update.effective_chat.id}")
    logging.info("=" * 60)

async def test_handler():
    """Test handler with mock update"""
    logging.info("Creating mock Telegram update...")

    # Create mock update with message
    mock_update = Mock(spec=Update)
    mock_update.message = Mock()
    mock_update.message.text = "Test message from user"
    mock_update.effective_chat = Mock()
    mock_update.effective_chat.id = 123456
    mock_update.effective_user = Mock()
    mock_update.effective_user.username = "testuser"

    logging.info("Calling handler with mock update...")
    await minimal_handler(mock_update, ContextTypes.DEFAULT_TYPE)

    logging.info("Test complete!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_handler())