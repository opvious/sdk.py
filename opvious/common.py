from importlib import metadata
import urllib.parse


try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = ""


del metadata


def format_percent(val):
    if val == "Infinity":
        return "inf"
    return f"{int(val * 10_000) / 100}%"


def strip_nones(d):
    return {k: v for k, v in d.items() if v is not None}


def is_url(s: str) -> bool:
    """Checks if a string is a URL."""
    try:
        res = urllib.parse.urlparse(s)
        return bool(res.scheme and res.netloc)
    except ValueError:
        return False
