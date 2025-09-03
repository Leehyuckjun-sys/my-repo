from __future__ import annotations
from time import perf_counter


class LatencyTimer:
"""Context manager to measure latency in ms."""
def __enter__(self):
self.t0 = perf_counter()
return self
def __exit__(self, exc_type, exc, tb):
self.dt_ms = (perf_counter() - self.t0) * 1000.0
