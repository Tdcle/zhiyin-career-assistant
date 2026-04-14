"""Lightweight runtime metrics and threshold-based alerting."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock

from config.config import config
from utils.logger import get_logger
from utils.redis_client import redis_is_ready
from utils.resume_task_queue import get_resume_queue_metrics


logger = get_logger("monitoring")


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


class RuntimeMetrics:
    def __init__(self) -> None:
        self.window_seconds = max(30, int(config.MONITOR_WINDOW_SECONDS))
        self.max_events = max(500, int(config.MONITOR_MAX_EVENTS))
        self.alert_min_samples = max(1, int(config.MONITOR_ALERT_MIN_SAMPLES))
        self.alert_check_interval_seconds = max(5, int(config.MONITOR_ALERT_CHECK_INTERVAL_SECONDS))
        self.alert_p95_ms = max(1, int(config.MONITOR_ALERT_P95_MS))
        self.alert_5xx_rate = max(0.0, float(config.MONITOR_ALERT_5XX_RATE))
        self.alert_resume_queue = max(1, int(config.MONITOR_ALERT_RESUME_QUEUE_LEN))

        self._events: deque[tuple[float, str, str, int, float]] = deque()
        self._lock = Lock()
        self._last_alert_check_ts = 0.0
        self._started_at = time.time()
        self._lifetime_total = 0
        self._lifetime_5xx = 0

    def record_request(self, *, method: str, path: str, status_code: int, latency_ms: float) -> None:
        now = time.time()
        safe_method = str(method or "").upper()[:16] or "GET"
        safe_path = str(path or "").strip()[:256] or "/"
        safe_status = int(status_code)
        safe_latency = max(0.0, float(latency_ms))

        with self._lock:
            self._events.append((now, safe_method, safe_path, safe_status, safe_latency))
            self._lifetime_total += 1
            if safe_status >= 500:
                self._lifetime_5xx += 1
            self._prune_locked(now)

    def _prune_locked(self, now_ts: float) -> None:
        cutoff = now_ts - self.window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()
        while len(self._events) > self.max_events:
            self._events.popleft()

    def _window_stats_locked(self) -> dict:
        latencies = [row[4] for row in self._events]
        total = len(self._events)
        status_counts = {"2xx": 0, "3xx": 0, "4xx": 0, "5xx": 0}
        route_map: dict[str, dict] = defaultdict(
            lambda: {
                "count": 0,
                "latencies": [],
                "errors_5xx": 0,
                "errors_4xx": 0,
            }
        )

        for _, method, path, status, latency in self._events:
            if 200 <= status < 300:
                status_counts["2xx"] += 1
            elif 300 <= status < 400:
                status_counts["3xx"] += 1
            elif 400 <= status < 500:
                status_counts["4xx"] += 1
            else:
                status_counts["5xx"] += 1

            key = f"{method} {path}"
            bucket = route_map[key]
            bucket["count"] += 1
            bucket["latencies"].append(latency)
            if status >= 500:
                bucket["errors_5xx"] += 1
            elif status >= 400:
                bucket["errors_4xx"] += 1

        routes = []
        for route_key, bucket in route_map.items():
            count = int(bucket["count"])
            routes.append(
                {
                    "route": route_key,
                    "count": count,
                    "p95_ms": round(_percentile(bucket["latencies"], 95), 2),
                    "avg_ms": round(sum(bucket["latencies"]) / max(count, 1), 2),
                    "error_rate_5xx": round(bucket["errors_5xx"] / max(count, 1), 4),
                    "error_rate_4xx": round(bucket["errors_4xx"] / max(count, 1), 4),
                }
            )
        routes.sort(key=lambda item: item["count"], reverse=True)

        error_rate_5xx = status_counts["5xx"] / max(total, 1)
        return {
            "count": total,
            "rps": round(total / float(self.window_seconds), 4),
            "p50_ms": round(_percentile(latencies, 50), 2),
            "p95_ms": round(_percentile(latencies, 95), 2),
            "p99_ms": round(_percentile(latencies, 99), 2),
            "avg_ms": round(sum(latencies) / max(total, 1), 2) if latencies else 0.0,
            "status_counts": status_counts,
            "error_rate_5xx": round(error_rate_5xx, 4),
            "routes": routes[:20],
        }

    def snapshot(self) -> dict:
        with self._lock:
            now = time.time()
            self._prune_locked(now)
            window = self._window_stats_locked()
            lifetime_total = self._lifetime_total
            lifetime_5xx = self._lifetime_5xx
            uptime_seconds = int(now - self._started_at)

        system = {
            "redis_ready": redis_is_ready(),
            "resume_queue": get_resume_queue_metrics(),
        }
        return {
            "timestamp": int(time.time()),
            "uptime_seconds": uptime_seconds,
            "window_seconds": self.window_seconds,
            "lifetime": {
                "request_total": lifetime_total,
                "request_5xx_total": lifetime_5xx,
                "error_rate_5xx": round(lifetime_5xx / max(lifetime_total, 1), 4),
            },
            "window": window,
            "system": system,
            "alert_thresholds": {
                "p95_ms": self.alert_p95_ms,
                "error_rate_5xx": self.alert_5xx_rate,
                "resume_queue_len": self.alert_resume_queue,
            },
        }

    def maybe_emit_alert(self) -> dict | None:
        now = time.time()
        with self._lock:
            if now - self._last_alert_check_ts < self.alert_check_interval_seconds:
                return None
            self._last_alert_check_ts = now
            self._prune_locked(now)
            window = self._window_stats_locked()

        if window["count"] < self.alert_min_samples:
            return None

        breaches: list[str] = []
        p95_ms = float(window["p95_ms"])
        if p95_ms >= float(self.alert_p95_ms):
            breaches.append(f"p95_ms={p95_ms}>=threshold({self.alert_p95_ms})")

        error_rate_5xx = float(window["error_rate_5xx"])
        if error_rate_5xx >= float(self.alert_5xx_rate):
            breaches.append(f"error_rate_5xx={error_rate_5xx:.4f}>=threshold({self.alert_5xx_rate})")

        queue_metrics = get_resume_queue_metrics()
        queue_len = int(queue_metrics.get("queue_length", 0) or 0)
        if queue_len >= int(self.alert_resume_queue):
            breaches.append(f"resume_queue_len={queue_len}>=threshold({self.alert_resume_queue})")

        if not breaches:
            return None

        payload = {
            "breaches": breaches,
            "window_count": window["count"],
            "p95_ms": window["p95_ms"],
            "error_rate_5xx": window["error_rate_5xx"],
            "resume_queue_len": queue_len,
        }
        logger.warning("runtime alert triggered: %s", payload)
        return payload


runtime_metrics = RuntimeMetrics()
