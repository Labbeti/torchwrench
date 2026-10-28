#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import time
from datetime import timedelta
from typing import Optional

logger = logging.getLogger(__name__)


def log_completion_time(
    step: int,
    max_steps: int,
    epoch: int,
    max_epochs: Optional[int],
    fit_start_time: Optional[float],
) -> None:
    step_completion_percent = step / max(max_steps, 1) * 100.0

    if max_epochs is None:
        epoch_completion_percent = 0.0
    else:
        epoch_completion_percent = epoch / max(max_epochs, 1) * 100.0

    completion_percent = max(step_completion_percent, epoch_completion_percent)
    msg = f"Fit completion: {completion_percent:.1f}% (epoch: {epoch}/{max_epochs} ({epoch_completion_percent:.1f}%), step: {step}/{max_steps} ({step_completion_percent:.1f}%))"
    logger.info(msg)

    if fit_start_time is None:
        msg = f"Cannot estimate end date because {fit_start_time=}."
        logger.warning(msg)
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

        msg = f"Estimated fit end: {estimated_fit_end_date} (elapsed: {duration_elapsed_delta}, remaining: {estimated_duration_remaining_delta})"
        logger.info(msg)
