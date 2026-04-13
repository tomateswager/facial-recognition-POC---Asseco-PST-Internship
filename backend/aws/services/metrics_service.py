"""
metrics_service.py - Recolha e análise de métricas de performance
Regista latências, contadores e estatísticas das operações AWS Rekognition.
"""
import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class OperationMetric:
    operation: str
    latency_ms: float
    success: bool
    timestamp: float = field(default_factory=time.time)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "operation": self.operation,
            "latency_ms": round(self.latency_ms, 2),
            "success": self.success,
            "timestamp": self.timestamp,
            **self.extra,
        }


class MetricsCollector:
    def __init__(self):
        self._metrics: list[OperationMetric] = []
        self._counters: dict[str, int] = defaultdict(int)

    def record(self, operation: str, latency_ms: float, success: bool = True, **extra):
        metric = OperationMetric(
            operation=operation,
            latency_ms=latency_ms,
            success=success,
            extra=extra,
        )
        self._metrics.append(metric)
        self._counters[operation] += 1
        if not success:
            self._counters[f"{operation}_errors"] += 1
        logger.debug(
            "[METRICS] %s %.1fms success=%s",
            operation,
            latency_ms,
            success,
        )

    def get_summary(self) -> dict:
        if not self._metrics:
            return {"total_operations": 0, "operations": {}}

        by_op: dict[str, list[float]] = defaultdict(list)
        errors: dict[str, int] = defaultdict(int)

        for m in self._metrics:
            by_op[m.operation].append(m.latency_ms)
            if not m.success:
                errors[m.operation] += 1

        summary = {}
        for op, latencies in by_op.items():
            n = len(latencies)
            avg = sum(latencies) / n
            sorted_l = sorted(latencies)
            p50 = sorted_l[int(n * 0.50)]
            p95 = sorted_l[min(int(n * 0.95), n - 1)]
            p99 = sorted_l[min(int(n * 0.99), n - 1)]
            summary[op] = {
                "count": n,
                "errors": errors.get(op, 0),
                "avg_ms": round(avg, 2),
                "min_ms": round(sorted_l[0], 2),
                "max_ms": round(sorted_l[-1], 2),
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
                "p99_ms": round(p99, 2),
            }

        return {
            "total_operations": len(self._metrics),
            "operations": summary,
        }

    def get_recent(self, n: int = 20) -> list:
        return [m.to_dict() for m in self._metrics[-n:]]

    def reset(self):
        self._metrics.clear()
        self._counters.clear()


# Instância global partilhada entre pedidos
_global_collector = MetricsCollector()


def get_collector() -> MetricsCollector:
    return _global_collector
