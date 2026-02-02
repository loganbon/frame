"""Module-level configuration for Frame defaults."""

import threading
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FrameConfig:
    """Configuration for Frame defaults."""

    default_cache_dir: Path | None = None  # None = use cwd/.frame_cache
    default_parent_cache_dirs: list[Path] = field(default_factory=list)


# Module-level singleton
_frame_config: FrameConfig | None = None
_config_lock = threading.Lock()


def get_frame_config() -> FrameConfig:
    """Get the global Frame configuration singleton."""
    global _frame_config
    if _frame_config is None:
        with _config_lock:
            if _frame_config is None:
                _frame_config = FrameConfig()
    return _frame_config


def configure_frame(
    default_cache_dir: Path | str | None = None,
    default_parent_cache_dirs: list[Path | str] | None = None,
) -> None:
    """Configure default Frame settings.

    Args:
        default_cache_dir: Default cache directory for new Frame instances.
            Pass None to reset to default behavior (cwd/.frame_cache).
        default_parent_cache_dirs: Default parent cache directories for new
            Frame instances. These are read-only fallback caches.

    Example:
        from frame import configure_frame

        # Set global defaults
        configure_frame(
            default_cache_dir="/data/frame_cache",
            default_parent_cache_dirs=["/shared/cache", "/archive/cache"]
        )

        # Now all Frame instances use these defaults
        f = Frame(my_func)  # Uses /data/frame_cache
    """
    config = get_frame_config()
    with _config_lock:
        if default_cache_dir is not None:
            config.default_cache_dir = Path(default_cache_dir)
        if default_parent_cache_dirs is not None:
            config.default_parent_cache_dirs = [
                Path(p) for p in default_parent_cache_dirs
            ]


def get_default_cache_dir() -> Path:
    """Get the default cache directory."""
    config = get_frame_config()
    if config.default_cache_dir is not None:
        return config.default_cache_dir
    return Path.cwd() / ".frame_cache"


def get_default_parent_cache_dirs() -> list[Path]:
    """Get the default parent cache directories."""
    return list(get_frame_config().default_parent_cache_dirs)


def reset_frame_config() -> None:
    """Reset configuration to defaults. Useful for testing."""
    global _frame_config
    with _config_lock:
        _frame_config = FrameConfig()
