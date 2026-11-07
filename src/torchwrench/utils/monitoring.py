#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import time
from datetime import timedelta
from typing import Optional
from numbers import Number

import pythonwrench as pw

logger = logging.getLogger(__name__)


def log_completion_time(
    step: int,
    max_steps: Optional[int],
    epoch: int,
    max_epochs: Optional[int],
    fit_start_time: Optional[float],
) -> None:
    """Log completion time based on current step/epoch and estimated end time."""
    if isinstance(max_steps, Number) and max_steps > 0:
        step_completion_percent = step / max(max_steps, 1) * 100.0
    else:
        step_completion_percent = 0.0

    if isinstance(max_epochs, Number) and max_epochs > 0:
        epoch_completion_percent = epoch / max(max_epochs, 1) * 100.0
    else:
        epoch_completion_percent = 0.0

    completion_percent = max(step_completion_percent, epoch_completion_percent)
    msg = f"Completion: {completion_percent:.1f}% (epoch: {epoch}/{max_epochs} ({epoch_completion_percent:.1f}%), step: {step}/{max_steps} ({step_completion_percent:.1f}%))"
    logger.info(msg)

    if fit_start_time is None:
        msg = f"Cannot estimate end date because {fit_start_time=}."
        pw.warn_once(msg)
    elif completion_percent == 0.0:
        pass
    else:
        duration_elapsed_s = time.time() - fit_start_time
        estimated_total_duration_s = duration_elapsed_s * 100 / completion_percent
        estimated_duration_remaining_s = estimated_total_duration_s - duration_elapsed_s
        estimated_fit_end_time = (
            fit_start_time + duration_elapsed_s + estimated_duration_remaining_s
        )
        estimated_fit_end_date = time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.localtime(estimated_fit_end_time)
        )
        duration_elapsed_delta = timedelta(seconds=duration_elapsed_s)
        estimated_duration_remaining_delta = timedelta(
            seconds=estimated_duration_remaining_s
        )

        msg = f"Estimated end: {estimated_fit_end_date} (elapsed: {duration_elapsed_delta}, remaining: {estimated_duration_remaining_delta})"
        logger.info(msg)
