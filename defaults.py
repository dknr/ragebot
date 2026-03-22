"""
Default RageBot configuration
"""

# Emoji threshold system - sentiment scores to emojis
EMOJI_THRESHOLDS = {
    'negative': [
        (0.95, '🤬'),  # Very negative
        (0.90, '😡'),  # Somewhat negative
        (0.80, '🙈'),  # Mildly negative
    ],
    'neutral': [
        (0.90, '😐'),  # Somewhat neutral
    ],
    'positive': [
        (0.95, '🎉'),  # Very positive
        (0.90, '👏'),  # Somewhat positive
        (0.80, '👍'),  # Mildly positive
    ],
}