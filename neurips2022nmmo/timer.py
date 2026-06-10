import time
import contextlib
from loguru import logger
from typing import Dict, List
from collections import defaultdict


class timer:
    costs: Dict[str, List[float]] = defaultdict(lambda: [])

    @classmethod
    @contextlib.contextmanager
    def count(cls, name: str, printout: bool = False):
        start = time.time()
        yield
        cost = time.time() - start
        cls.costs[name].append(cost)
        msg = f"func {name} cost {cost} seconds"
        if printout:
            logger.opt(depth=2).info(msg)
        else:
            logger.opt(depth=2).trace(msg)

    @classmethod
    def reset(cls) -> Dict[str, Dict[str, float]]:
        stats = defaultdict(lambda: {})
        for name, history in cls.costs.items():
            stats[name]["max"] = max(history)
            stats[name]["min"] = min(history)
            stats[name]["avg"] = (sum(history) /
                                  len(history)) if history else 0
            stats[name]["50p"] = history[len(history) // 2] if history else 0
        cls.costs.clear()
        return stats
