from typing import Sequence, List

from neurips2022nmmo.env.metrics import Metrics


class Stat(dict):
    @classmethod
    def names(cls) -> List[str]:
        return ["sum", "max", "avg", "min"]

    @classmethod
    def from_metrices(cls, metrices: Sequence[Metrics]) -> "Stat":
        return Stat({
            "sum": Metrics.sum(metrices),
            "max": Metrics.max(metrices),
            "avg": Metrics.avg(metrices),
            "min": Metrics.min(metrices),
        })
