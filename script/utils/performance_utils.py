# utils/model_utils.py
from __future__ import annotations
import os
import time
import socket
import datetime as _dt
from typing import Any, Callable, Tuple

def timed_call(
    callback: Callable[..., Any],
    *args: Any,
    log_path: str | None = None,
    label: str | None = None,
    **kwargs: Any,
) -> Tuple[Any, float]:
    """
    Run `callback(*args, **kwargs)`, measure wall-clock seconds, log to a .txt, and return (result, seconds).

    Args:
        callback: Function to execute.
        *args, **kwargs: Passed through to `callback`.
        log_path: Where to append the timing log (default: 'results/runtime_log.txt').
        label: Optional label for the log (default: callback.__name__).

    Returns:
        (result, seconds): the callback's return value and elapsed wall-clock seconds.

    Raises:
        Re-raises any exception from `callback` after logging it.
    """
    if log_path is None:
        log_path = os.path.join("results", "runtime_log.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    name = label or getattr(callback, "__name__", "callback")
    start = time.perf_counter()
    try:
        result = callback(*args, **kwargs)
        ok = True
        err_msg = ""
        return_value = result
    except Exception as e:
        ok = False
        err_msg = f"{type(e).__name__}: {e}"
        return_value = None  # not used when re-raising
    finally:
        elapsed = time.perf_counter() - start
        ts = _dt.datetime.now().isoformat(timespec="seconds")
        host = socket.gethostname()
        line = f"{ts}\t{name}\tok={ok}\tsecs={elapsed:.3f}\thost={host}"
        if not ok:
            line += f"\terror=\"{err_msg}\""
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    if not ok:
        # Re-raise to preserve original behavior, after logging
        raise

    return return_value, elapsed


def timeit(log_path: str | None = None, label: str | None = None):
    """
    Decorator version of `timed_call`. Usage:

        @timeit(log_path="results/runtime_log.txt")
        def batch_inference(...):
            ...

        (res, secs) = batch_inference(...)  # returns (original_result, seconds)
    """
    def _decorator(func: Callable[..., Any]):
        def _wrapper(*args: Any, **kwargs: Any):
            return timed_call(func, *args, log_path=log_path, label=label or func.__name__, **kwargs)
        _wrapper.__name__ = func.__name__
        _wrapper.__doc__ = func.__doc__
        return _wrapper
    return _decorator
