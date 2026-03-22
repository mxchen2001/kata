import re


def parse_duration(duration: str) -> int:
    """
    Parse a human-readable duration string into total seconds.
    
    Supports units: w (weeks), d (days), h (hours), m (minutes), s (seconds)
    Examples: "2h30m", "1d12h", "45s", "1w2d"
    
    Args:
        duration: Human-readable duration string
        
    Returns:
        Total number of seconds as an integer
        
    Raises:
        ValueError: If the input string is invalid or empty
    """
    if not duration or not isinstance(duration, str):
        raise ValueError(f"Invalid duration: {duration!r}")
    
    unit_seconds: dict[str, int] = {
        'w': 7 * 24 * 3600,
        'd': 24 * 3600,
        'h': 3600,
        'm': 60,
        's': 1,
    }
    
    pattern = re.compile(r'^(\d+[wdhms])+$')
    
    if not pattern.match(duration):
        raise ValueError(f"Invalid duration format: {duration!r}")
    
    token_pattern = re.compile(r'(\d+)([wdhms])')
    tokens = token_pattern.findall(duration)
    
    if not tokens:
        raise ValueError(f"No valid duration tokens found in: {duration!r}")
    
    total_seconds: int = 0
    for value_str, unit in tokens:
        value = int(value_str)
        total_seconds += value * unit_seconds[unit]
    
    return total_seconds


def format_duration(seconds: int) -> str:
    """
    Format a duration in seconds back to a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Human-readable duration string
        
    Raises:
        ValueError: If seconds is negative
    """
    if seconds < 0:
        raise ValueError(f"Duration cannot be negative: {seconds}")
    
    if seconds == 0:
        return "0s"
    
    units: list[tuple[str, int]] = [
        ('w', 7 * 24 * 3600),
        ('d', 24 * 3600),
        ('h', 3600),
        ('m', 60),
        ('s', 1),
    ]
    
    parts: list[str] = []
    remaining = seconds
    
    for unit, unit_value in units:
        if remaining >= unit_value:
            count = remaining // unit_value
            remaining %= unit_value
            parts.append(f"{count}{unit}")
    
    return ''.join(parts)
