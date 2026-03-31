#!/usr/bin/env python3
"""
RageBot - Telegram bot with message logging
"""

import asyncio
import logging
import os
import re
import sys
from datetime import datetime, timedelta
from collections import defaultdict
from io import BytesIO

from dotenv import load_dotenv
load_dotenv()

from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes, CommandHandler

# Import for sentiment analysis
try:
    from transformers import pipeline
    ROBERTA_AVAILABLE = True
except ImportError:
    ROBERTA_AVAILABLE = False
    logging.warning("transformers library not installed. Sentiment analysis unavailable.")

# Import for database storage
try:
    from store import init_db, add_message, get_vibecheck, format_vibecheck, get_thresholds, delete_old_messages, get_sentiment_trend
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logging.warning("store module not available. Statistics will not be saved.")

# Import for graph generation
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False
    logging.warning("matplotlib not installed. Graph generation unavailable.")

# Import default thresholds
from defaults import EMOJI_THRESHOLDS

def load_neat_gifs():
    """Load all neat GIF files from assets folder"""
    neat_gifs = []
    assets_dir = "assets"

    try:
        if os.path.exists(assets_dir):
            for filename in os.listdir(assets_dir):
                if filename.startswith("neat-") and filename.endswith(".gif"):
                    gif_path = os.path.join(assets_dir, filename)
                    neat_gifs.append(gif_path)
                    logging.info(f"Found neat GIF: {filename}")
        else:
            logging.warning(f"Assets directory not found: {assets_dir}")

        if not neat_gifs:
            logging.warning("No neat GIF files found in assets folder")

        return neat_gifs
    except Exception as e:
        logging.error(f"Error loading neat GIFs: {e}", exc_info=True)
        return []

def get_emoji_for_sentiment(label, sentiment_score, irony_score=0):
    """Get emoji based on sentiment, irony, and score"""
    logging.debug(f"get_emoji_for_sentiment: label={label}, score={sentiment_score}, irony={irony_score}")

    # Check irony first (highest priority)
    try:
        thresholds = get_thresholds()
        if 'irony' in thresholds:
            for threshold, emoji in thresholds['irony']:
                if irony_score >= threshold:
                    logging.debug(f"Matched irony: {irony_score} >= {threshold}, emoji={emoji}")
                    return emoji
    except Exception as e:
        logging.warning(f"Error getting irony thresholds from database: {e}")

    # Try to get thresholds from database
    try:
        thresholds = get_thresholds()
        if label in thresholds:
            # Check thresholds from high to low
            for threshold, emoji in thresholds[label]:
                if sentiment_score >= threshold:
                    logging.debug(f"Matched database threshold: {label} {sentiment_score} >= {threshold}, emoji={emoji}")
                    return emoji
    except Exception as e:
        logging.warning(f"Error getting thresholds from database: {e}")

    logging.debug(f"No emoji matched for score {label} {sentiment_score}, irony {irony_score}")
    return None

def is_command_message(update: Update) -> bool:
    """Check if message is a command"""
    return update.message and update.message.text and update.message.text.startswith("/")

def format_sentiment_debug(emoji, sentiment_label, sentiment_score, irony_score):
    """Format sentiment analysis results for debug output"""
    lines = ["<pre><code>"]
    lines.append("🤖 Sentiment Analysis Results\n")
    lines.append(f"Reaction: {emoji}\n\n")
    lines.append("Raw Output:\n")
    lines.append("-" * 30 + "\n")
    lines.append(f"Label: {sentiment_label}\n")
    lines.append(f"Score: {sentiment_score:.4f}\n")
    if irony_score > 0:
        lines.append(f"Irony Score: {irony_score:.4f}\n")
    lines.append("</code></pre>")
    return "\n".join(lines)

# Create log directory
os.makedirs("log", exist_ok=True)

# Set up logging to file with timestamp
log_filename = f"log/ragebot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.DEBUG)

# Set up logging to console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.WARNING)

# Configure root logger
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
    handlers=[file_handler, console_handler]
)

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")



async def debug_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Toggle debug logging"""
    try:
        if "debug_mode" not in context.application.bot_data:
            context.application.bot_data["debug_mode"] = False

        context.application.bot_data["debug_mode"] = not context.application.bot_data["debug_mode"]
        debug_mode = context.application.bot_data["debug_mode"]

        if debug_mode:
            await update.message.reply_text("🐛 RageBot debug mode enabled.")
        else:
            await update.message.reply_text("🔇 RageBot debug mode disabled.")
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")

async def mute_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Toggle mute mode"""
    try:
        if "mute_mode" not in context.application.bot_data:
            context.application.bot_data["mute_mode"] = False

        context.application.bot_data["mute_mode"] = not context.application.bot_data["mute_mode"]
        mute_mode = context.application.bot_data["mute_mode"]

        if mute_mode:
            await update.message.reply_text("🔇 RageBot muted. Emoji reactions disabled.")
        else:
            await update.message.reply_text("🔊 RageBot unmuted. Emoji reactions enabled.")
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Reset error counter and mute mode"""
    try:
        context.application.bot_data["sentiment_error_count"] = 0
        context.application.bot_data["mute_mode"] = False
        await update.message.reply_text("✅ Error counter reset. Mute mode disabled.")
        logging.info("RageBot reset completed")
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")
        logging.error(f"Error in reset command: {e}", exc_info=True)

async def thresholds_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show sentiment thresholds"""
    lines = ["<pre><code>"]
    lines.append("RageBot Emoji Thresholds:\n")

    # Read thresholds from database
    try:
        thresholds = get_thresholds()

        # Sort by sentiment and threshold descending
        sorted_thresholds = []
        for sentiment, threshold_list in sorted(thresholds.items()):
            for threshold, emoji in sorted(threshold_list, key=lambda x: x[0], reverse=True):
                sorted_thresholds.append((sentiment, threshold, emoji))

        # Print table header
        lines.append(f"{'Sentiment':<12} {'Threshold':<12} {'Reaction':<8}\n")
        lines.append("-" * 35 + "\n")

        # Print threshold rows
        for sentiment, threshold, emoji in sorted_thresholds:
            lines.append(f"{sentiment:<12} {threshold:<12.2f} {emoji:<8}\n")

    except Exception as e:
        lines.append(f"Error loading thresholds: {e}\n")

    lines.append("</code></pre>")

    await update.message.reply_text("".join(lines), parse_mode="HTML")

async def vibecheck_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show last 24h statistics"""
    if DB_AVAILABLE:
        try:
            chat_id = update.effective_chat.id
            vibecheck_data = get_vibecheck(chat_id, hours=24)
            body = format_vibecheck(vibecheck_data)
            await update.message.reply_text(body)
            logging.info(f"RageBot vibecheck for chat {chat_id} sent")
        except Exception as e:
            await update.message.reply_text(f"Error getting stats: {e}")
            logging.error(f"Error in vibecheck command: {e}")
    else:
        await update.message.reply_text("Statistics feature not available.")
        logging.warning("vibecheck command attempted but DB not available")

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show current bot status"""
    status_lines = ["📊 RageBot Status\n"]

    # Debug mode
    debug_mode = context.application.bot_data.get("debug_mode", False)
    status_lines.append(f"Debug mode: {'✅ ON' if debug_mode else '❌ OFF'}")

    # Mute mode
    mute_mode = context.application.bot_data.get("mute_mode", False)
    status_lines.append(f"Mute mode: {'🔇 ON' if mute_mode else '🔊 OFF'}")

    # Sentiment error count
    error_count = context.application.bot_data.get("sentiment_error_count", 0)
    status_lines.append(f"Sentiment errors: {error_count}")

    # Sentiment pipeline
    if "sentiment_pipeline" in context.application.bot_data:
        status_lines.append("Sentiment analysis: ✅ Active (RoBERTa)")
    else:
        status_lines.append("Sentiment analysis: ❌ Not loaded")

    # Database
    if DB_AVAILABLE:
        status_lines.append("Database: ✅ Active (SQLite)")
    else:
        status_lines.append("Database: ❌ Not available")

    # Thresholds
    try:
        thresholds = get_thresholds()
        if thresholds:
            status_lines.append(f"Thresholds: ✅ {len(thresholds)} sentiment types")
        else:
            status_lines.append("Thresholds: ❌ Not configured")
    except Exception as e:
        status_lines.append(f"Thresholds: ❌ Error ({e})")

    await update.message.reply_text("\n".join(status_lines))


async def clear_old_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear old messages from database"""
    try:
        # Parse days argument (default: 30 days)
        days = 30
        if context.args and len(context.args) > 0:
            try:
                days = int(context.args[0])
                if days < 1:
                    await update.message.reply_text("Error: Days must be at least 1")
                    return
            except ValueError:
                await update.message.reply_text("Error: Invalid number format")
                return

        deleted_count = delete_old_messages(days)

        if deleted_count > 0:
            await update.message.reply_text(f"🗑️ Deleted {deleted_count} old message(s) older than {days} days")
        else:
            await update.message.reply_text("✅ No old messages found to delete")

    except Exception as e:
        await update.message.reply_text(f"Error: {e}")
        logging.error(f"Error in clear_old command: {e}", exc_info=True)


async def test_emoji_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Test all configured emoji reactions"""
    try:
        all_emojis = []
        # Collect all emojis from database thresholds
        thresholds = get_thresholds()
        for sentiment, threshold_list in thresholds.items():
            for threshold, emoji in threshold_list:
                all_emojis.append(emoji)

        if not all_emojis:
            await update.message.reply_text("No emojis configured to test!")
            return

        results = []
        await update.message.reply_text(f"Testing {len(all_emojis)} emoji(s)...")

        # Test each emoji
        for emoji in all_emojis:
            try:
                await update.message.set_reaction(emoji)
                results.append(f"✅ {emoji}")
                # Wait 1 second before next emoji
                await asyncio.sleep(1)
            except Exception as e:
                results.append(f"❌ {emoji} - {e}")
                # Log warning for unsupported emoji
                if "Reaction emoji" in str(e) or "can't be used" in str(e):
                    logging.warning(f"Telegram rejected emoji {emoji}: {e}")

        # Send results report
        await update.message.reply_text("Emoji test results:\n" + "\n".join(results))

    except Exception as e:
        await update.message.reply_text(f"Error during emoji test: {e}")
        logging.error(f"Emoji test error: {e}", exc_info=True)


async def trends_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate and send sentiment trend graph"""
    if not DB_AVAILABLE:
        await update.message.reply_text("Statistics feature not available.")
        return
    
    if not GRAPH_AVAILABLE:
        await update.message.reply_text("Graph generation not available (matplotlib not installed).")
        return
    
    try:
        chat_id = update.effective_chat.id
        
        # Parse arguments
        hours = 168  # 7 days default
        interval_hours = 1  # 1 hour default
        
        if context.args and len(context.args) > 0:
            try:
                hours = int(context.args[0])
                if hours < 1:
                    await update.message.reply_text("Error: Hours must be at least 1")
                    return
            except ValueError:
                await update.message.reply_text("Error: Invalid number format")
                return
        
        if len(context.args) > 1:
            try:
                interval_hours = int(context.args[1])
                if interval_hours < 1:
                    await update.message.reply_text("Error: Interval must be at least 1")
                    return
            except ValueError:
                await update.message.reply_text("Error: Invalid interval format")
                return
        
        # Get trend data
        trend_data = get_sentiment_trend(chat_id, hours=hours, interval_hours=interval_hours)
        
        if not trend_data:
            await update.message.reply_text("No message data available for the specified time period.")
            return
        
        # Convert UTC timestamps to Arizona time (UTC-7)
        timestamps_utc = [datetime.strptime(entry['timestamp'], '%Y-%m-%d %H:%M:%S') for entry in trend_data]
        timestamps_az = [ts.replace(tzinfo=None) - timedelta(hours=7) for ts in timestamps_utc]
        
        positive = [entry['positive'] for entry in trend_data]
        neutral = [entry['neutral'] for entry in trend_data]
        negative = [entry['negative'] for entry in trend_data]
        total = [entry['total'] for entry in trend_data]
        
        # Create bar positions
        x_pos = range(len(timestamps_az))
        
        # Generate stacked bar graph
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Bottom: negative (red)
        ax.bar(x_pos, negative, color='red', label='Negative', alpha=0.7)
        # Middle: positive (green) - stacked on top of negative
        ax.bar(x_pos, positive, color='green', label='Positive', alpha=0.7, bottom=negative)
        # Top: neutral (blue) - stacked on top of negative+positive
        ax.bar(x_pos, neutral, color='blue', label='Neutral', alpha=0.7, 
               bottom=[n+p for n,p in zip(negative, positive)])
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Message Count')
        ax.set_title(f'Sentiment Trend ({hours//24} days, {interval_hours}h buckets)')
        ax.legend()
        
        # Add shaded background for night time (6pm-6am = hours 18-6)
        # Group consecutive night hours into single spans to avoid overlap
        night_ranges = []
        start_idx = None
        for i, ts in enumerate(timestamps_az):
            hour = ts.hour
            is_night = hour >= 18 or hour < 6
            if is_night and start_idx is None:
                start_idx = i
            elif not is_night and start_idx is not None:
                # End of night range
                night_ranges.append((start_idx, i - 1))
                start_idx = None
        # Handle night that extends to the end
        if start_idx is not None:
            night_ranges.append((start_idx, len(timestamps_az) - 1))
        
        # Draw single axvspan for each night range
        for start_idx, end_idx in night_ranges:
            ax.axvspan(start_idx, end_idx + 1, color='lightgray', alpha=0.3, zorder=0)
        
        # Show hour labels for every 3rd bucket (two-digit hour)
        step = 3
        ax.set_xticks(x_pos[::step])
        ax.set_xticklabels([ts.strftime('%H') for ts in timestamps_az[::step]], rotation=0, ha='center')
        
        # Add horizontal separator lines and weekday labels for day boundaries
        day_boundaries = defaultdict(list)
        for i, ts in enumerate(timestamps_az):
            day_key = ts.strftime('%Y-%m-%d')
            day_boundaries[day_key].append(i)
        
        for day_key, indices in day_boundaries.items():
            if indices:
                # Draw horizontal line across the day
                start_idx = indices[0]
                end_idx = indices[-1]
                ax.axhline(y=-0.15, xmin=start_idx/len(timestamps_az), xmax=(end_idx+1)/len(timestamps_az),
                          color='gray', linewidth=1, clip_on=False)
                
                # Add weekday label centered on the day
                center_idx = (start_idx + end_idx) // 2
                dow = timestamps_az[start_idx].strftime('%a')
                month = timestamps_az[start_idx].strftime('%b')
                day_num = timestamps_az[start_idx].day
                day_label = f"{dow} {month} {day_num}"
                ax.text(center_idx, -0.20, day_label, ha='center', va='top', 
                       fontsize=9, weight='bold')
        
        ax.grid(True, axis='both', alpha=0.3)
        plt.tight_layout()
        
        # Save to bytes buffer
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        # Send as photo
        await update.message.reply_photo(photo=buf, caption=f"📊 Sentiment Trend - {hours//24} days (stacked bars)")
        logging.info(f"RageBot trends graph sent for chat {chat_id}")
        
    except Exception as e:
        await update.message.reply_text(f"Error generating trend graph: {e}")
        logging.error(f"Error in trends command: {e}", exc_info=True)


async def minimal_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Minimal handler that logs and reacts"""
    # Skip command messages
    if is_command_message(update):
        return

    try:
        logging.debug(f"Message received from {update.effective_user.username} in chat {update.effective_chat.id}")

        # Count lol occurrences (case-insensitive)
        text = update.message.text or ""
        lol_matches = re.findall(r'\b(lol|lmao|rofl|haha)\b', text, re.IGNORECASE)
        lol_count = len(lol_matches)
        is_neat = bool(re.search(r'\bneat[!\.]*\b', text, re.IGNORECASE))
        truncated_text = text[:160] if len(text) > 160 else text


        # Check for NEAT! pattern (case-insensitive, catches "neat", "neat!", "neat.", etc.)
        if is_neat:
            logging.info(f"NEAT! pattern detected in message: {text}")

            # Load neat GIFs if not already loaded
            if "neat_gifs" not in context.application.bot_data:
                neat_gifs = load_neat_gifs()
                if neat_gifs:
                    context.application.bot_data["neat_gifs"] = neat_gifs

            # Randomly select and send a GIF if available
            if "neat_gifs" in context.application.bot_data:
                import random
                neat_gifs = context.application.bot_data["neat_gifs"]
                random_gif = random.choice(neat_gifs)

                try:
                    await update.message.reply_animation(animation=open(random_gif, 'rb'))
                    logging.info(f"NEAT! GIF sent: {os.path.basename(random_gif)}")
                except Exception as e:
                    logging.error(f"Error sending NEAT! GIF: {e}")
            else:
                logging.info("No neat GIFs available, skipping")
        if "sentiment_pipeline" in context.application.bot_data:
            try:
                sentiment_result = context.application.bot_data["sentiment_pipeline"](truncated_text)
                sentiment_label = sentiment_result[0]["label"]
                sentiment_score = sentiment_result[0]["score"]
                logging.debug(f"Sentiment: {sentiment_label} (score: {sentiment_score:.4f})")

                # Get irony score
                irony_score = 0
                if "irony_pipeline" in context.application.bot_data:
                    try:
                        irony_result = context.application.bot_data["irony_pipeline"](truncated_text)
                        # Get highest irony score
                        irony_scores = [r['score'] for r in irony_result]
                        irony_score = max(irony_scores) if irony_scores else 0
                        logging.debug(f"Irony score: {irony_score:.4f}")
                    except Exception as e:
                        logging.debug(f"Irony analysis error: {e}")

                # Get emoji based on sentiment and irony
                emoji = get_emoji_for_sentiment(sentiment_label, sentiment_score, irony_score)
                mute_mode = context.application.bot_data.get("mute_mode", False)

                # Send debug output if debug mode is enabled
                debug_mode = context.application.bot_data.get("debug_mode", False)
                if debug_mode:
                    await update.message.reply_text(format_sentiment_debug(emoji, sentiment_label, sentiment_score, irony_score), parse_mode="HTML")

                # Add emoji reaction if not muted
                if emoji and not mute_mode:
                    try:
                        await update.message.set_reaction(emoji)
                        logging.debug(f"Replied with {emoji}")
                    except Exception as e:
                        logging.warning(f"Telegram rejected emoji {emoji}: {e}")

                # Add to database
                if DB_AVAILABLE:
                    try:
                        add_message(
                            chat_id=update.effective_chat.id,
                            message_id=update.message.message_id,
                            sentiment_label=sentiment_label,
                            sentiment_score=sentiment_score,
                            lol_count=lol_count,
                            is_neat=is_neat
                        )
                    except Exception as e:
                        logging.error(f"Failed to add message to database: {e}")
            except Exception as e:
                # Sentiment analysis failed
                logging.error(f"Sentiment analysis error: {e}", exc_info=True)

                # Add message to database with lol count even if sentiment failed
                if DB_AVAILABLE:
                    try:
                        add_message(
                            chat_id=update.effective_chat.id,
                            message_id=update.message.message_id,
                            sentiment_label='unknown',
                            sentiment_score=0,
                            lol_count=lol_count,
                            is_neat=is_neat
                        )
                    except Exception as db_e:
                        logging.error(f"Failed to add message to database: {db_e}")

                # Increment error counter
                error_count = context.application.bot_data.get("sentiment_error_count", 0) + 1
                context.application.bot_data["sentiment_error_count"] = error_count

                # React with 👾 if not muted
                mute_mode = context.application.bot_data.get("mute_mode", False)
                if not mute_mode:
                    try:
                        await update.message.set_reaction("👾")
                    except Exception as e:
                        logging.warning(f"Telegram rejected emoji 👾: {e}")

                # Check if error count exceeds 10
                if error_count >= 10:
                    context.application.bot_data["mute_mode"] = True
                    await update.message.reply_text(
                        "⚠️ Sentiment analysis has failed 10+ times. Mute mode enabled to prevent spam."
                    )
        else:
            # Fallback for when pipeline not loaded
            logging.debug("Sentiment pipeline not loaded.")
    except Exception as e:
        logging.error(f"Error in minimal_handler: {e}", exc_info=True)

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors"""
    logging.error(f"Error: {context.error}", exc_info=True)

def main():
    """Main entry point"""
    logging.info("Creating application...")
    app = Application.builder().token(TOKEN).build()

    # Initialize database
    if DB_AVAILABLE:
        try:
            init_db()
            logging.info("Database initialized successfully!")
        except Exception as e:
            logging.error(f"Failed to initialize database: {e}")

    # Ensure database connection is properly closed on exit
    import atexit
    atexit.register(lambda: close_db_connection() if DB_AVAILABLE else None)

    logging.info("RageBot started with emoji threshold system")
    logging.info("Loaded thresholds: Negative/Neutral/Positive with 3 levels each")
    logging.debug(f"EMOJI_THRESHOLDS: {EMOJI_THRESHOLDS}")

    # Initialize error counter
    if "sentiment_error_count" not in app.bot_data:
        app.bot_data["sentiment_error_count"] = 0

    # Load RoBERTa sentiment pipeline
    if ROBERTA_AVAILABLE:
        logging.info("Loading RoBERTa sentiment analysis pipeline...")
        try:
            sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            logging.info("RoBERTa sentiment pipeline loaded successfully!")
            app.bot_data["sentiment_pipeline"] = sentiment_pipeline

            logging.info("Loading RoBERTa irony detection pipeline...")
            try:
                irony_pipeline = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-irony")
                logging.info("RoBERTa irony pipeline loaded successfully!")
                app.bot_data["irony_pipeline"] = irony_pipeline
            except Exception as e:
                logging.error(f"Failed to load irony pipeline: {e}", exc_info=True)

        except Exception as e:
            logging.error(f"Failed to load RoBERTa pipeline: {e}", exc_info=True)
    else:
        logging.info("Skipping sentiment pipeline (transformers not installed)")

    logging.info("Adding handlers...")
    # Command handlers
    app.add_handler(CommandHandler("ragebot_debug", debug_command))
    app.add_handler(CommandHandler("ragebot_mute", mute_command))
    app.add_handler(CommandHandler("ragebot_reset", reset_command))
    app.add_handler(CommandHandler("ragebot_thresholds", thresholds_command))
    app.add_handler(CommandHandler("ragebot_vibecheck", vibecheck_command))
    app.add_handler(CommandHandler("ragebot_clear_old", clear_old_command))
    app.add_handler(CommandHandler("ragebot_status", status_command))
    app.add_handler(CommandHandler("ragebot_test_emoji", test_emoji_command))
    app.add_handler(CommandHandler("ragebot_trends", trends_command))

    # Message handler for sentiment analysis (skip command messages)
    app.add_handler(MessageHandler(filters.TEXT, minimal_handler))

    logging.info("Adding error handler...")
    app.add_error_handler(error_handler)

    logging.info("Starting minimal bot...")
    app.run_polling()

if __name__ == "__main__":
    main()
