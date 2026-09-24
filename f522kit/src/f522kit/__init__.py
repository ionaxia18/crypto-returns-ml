"""f522kit: nonlinear modeling on anonymous core980 features."""

from . import contract
from .calendar import Anchor, Schedule, build_schedule, group_dates_by_week, week_end
from .data import DataError, Dataset, DateShard
from .driver import resolve_schedule, run
from .metrics import DailyStats, aggregate, annualized_sharpe, clip_alpha, daily_stats, paired_diff
from .scoring import format_report, save_report, score
from .window import DateBlock, TrainWindow

__version__ = "0.2.3rc2"

__all__ = [
    "Anchor",
    "DataError",
    "DateBlock",
    "DailyStats",
    "Dataset",
    "DateShard",
    "Schedule",
    "TrainWindow",
    "aggregate",
    "annualized_sharpe",
    "build_schedule",
    "clip_alpha",
    "contract",
    "daily_stats",
    "format_report",
    "group_dates_by_week",
    "paired_diff",
    "resolve_schedule",
    "run",
    "save_report",
    "score",
    "week_end",
]
