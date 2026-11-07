#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import time
from datetime import timedelta
from dataclasses import dataclass
from typing import Optional, Mapping, Tuple
from numbers import Real

import pythonwrench as pw
from pythonwrench.datetime import ISO8601_HOUR_FORMAT_DOUBLE_DOT

logger = logging.getLogger(__name__)


@dataclass
class RemainingTimeInfo:
    elapsed_duration_s: float
    elapsed_duration_delta: timedelta
    estimated_total_duration_s: float
    estimated_remaining_duration_s: float
    estimated_end_time: float
    estimated_end_date: str
    estimated_duration_remaining_delta: timedelta


def log_completion_time(
    step: int,
    max_steps: Optional[int],
    epoch: int,
    max_epochs: Optional[int],
    fit_start_time: Optional[float],
    date_format: str = ISO8601_HOUR_FORMAT_DOUBLE_DOT,
) -> float:
    """Log completion time based on current step/epoch and estimated end time."""
    progress_dict = {
        "step": (step, max_steps),
        "epoch": (epoch, max_epochs),
    }
    return log_completion_time_from_progress_dict(
        progress_dict,
        fit_start_time,
        date_format,
    )


def log_completion_time_from_progress_dict(
    progress_dict: Mapping[str, Tuple[float, Optional[float]]],
    fit_start_time: Optional[float],
    date_format: str = ISO8601_HOUR_FORMAT_DOUBLE_DOT,
) -> float:
    completion_percents = {}
    for name, (current, maxvalue) in progress_dict.items():
        if isinstance(maxvalue, Real) and maxvalue > 0:
            completion_percent = current / max(maxvalue, 1) * 100.0
        else:
            completion_percent = 0.0

        completion_percents[name] = completion_percent

    max_completion_percent = max(completion_percents.values())
    details = (
        f"{name}: {progress_dict[name][0]}/{progress_dict[name][1]} ({completion_percent:.1f}%)"
        for name, completion_percent in completion_percents.items()
    )
    msg = f"Completion: {max_completion_percent:.1f}% ({' '.join(details)})"
    logger.info(msg)

    if fit_start_time is None:
        msg = f"Cannot estimate end date because {fit_start_time=}."
        pw.warn_once(msg)
    elif max_completion_percent == 0.0:
        pass
    else:
        remaining = get_remaining_time(
            fit_start_time, max_completion_percent, date_format
        )
        msg = f"Estimated end: {remaining.estimated_end_date} (elapsed: {remaining.elapsed_duration_delta}, remaining: {remaining.estimated_duration_remaining_delta})"
        logger.info(msg)

    return max_completion_percent


def get_remaining_time(
    fit_start_time: float,
    completion_percent: float,
    date_format: str = ISO8601_HOUR_FORMAT_DOUBLE_DOT,
) -> RemainingTimeInfo:
    elapsed_duration_s = time.time() - fit_start_time
    estimated_total_duration_s = elapsed_duration_s * 100 / completion_percent
    estimated_remaining_duration_s = estimated_total_duration_s - elapsed_duration_s
    estimated_end_time = (
        fit_start_time + elapsed_duration_s + estimated_remaining_duration_s
    )
    estimated_end_date = time.strftime(date_format, time.localtime(estimated_end_time))
    elapsed_duration_delta = timedelta(seconds=elapsed_duration_s)
    estimated_duration_remaining_delta = timedelta(
        seconds=estimated_remaining_duration_s
    )
    return RemainingTimeInfo(
        elapsed_duration_s=elapsed_duration_s,
        elapsed_duration_delta=elapsed_duration_delta,
        estimated_total_duration_s=estimated_total_duration_s,
        estimated_remaining_duration_s=estimated_remaining_duration_s,
        estimated_end_time=estimated_end_time,
        estimated_end_date=estimated_end_date,
        estimated_duration_remaining_delta=estimated_duration_remaining_delta,
    )
