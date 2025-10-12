"""Utility functions for datetime conversion in infrastructure layer."""

from datetime import datetime, timezone
from typing import Optional


def naive_to_utc_aware(dt: Optional[datetime]) -> Optional[datetime]:
    """Convert a naive datetime (from database) to a UTC-aware datetime.
    
    If the datetime is naive (from database), assume it's UTC and make it timezone-aware.
    If it's already timezone-aware, convert to UTC.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    else:
        return dt.astimezone(timezone.utc)
