from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from neurips2022nmmo.env.metrics import Metrics
from neurips2022nmmo.env.stat import Stat
from collections import defaultdict

ALIVE_SCORE = [10, 6, 5, 4, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
DEFEAT_SCORE = 0.5


@dataclass
class TeamResult:
    policy_id: str
    alive_score: int
    defeat_score: int
    total_score: int
    time_alive: int
    gold: int
    damage_taken: int
    n_timeout: Optional[int] = 0

    @classmethod
    def names(cls) -> List[str]:
        return [
            "total_score",
            "alive_score",
            "defeat_score",
            "time_alive",
            "gold",
            "damage_taken",
        ]


def gen_result(
    policy_id_by_team: Dict[int, str],
    metrics_by_team: Dict[int, Dict[int, Metrics]],
    n_timeout_by_team: Dict[int, int],
) -> Dict[int, TeamResult]:

    stat_by_team = {}
    for team_idx, metrics in metrics_by_team.items():
        stat_by_team[team_idx] = Stat.from_metrices(metrics.values())
    quads = [(
        team_idx,
        stat["max"]["TimeAlive"],
        count_max_alive_time(metrics_by_team[team_idx]),
        stat["avg"]["Profession"],
    ) for team_idx, stat in stat_by_team.items()]
    alive_scores = compute_alive_score(quads)

    results = {}
    for (team_idx, alive_score) in alive_scores:
        policy_id = policy_id_by_team[team_idx]
        time_alive = stat_by_team[team_idx]["max"]["TimeAlive"]
        defeat_score = stat_by_team[team_idx]["sum"][
            "PlayerDefeats"] * DEFEAT_SCORE
        total_score = alive_score + defeat_score
        gold = stat_by_team[team_idx]["sum"]["Gold"]
        damage_taken = stat_by_team[team_idx]["sum"]["DamageTaken"]
        n_timeout = n_timeout_by_team[team_idx]
        result = TeamResult(
            policy_id=policy_id,
            alive_score=alive_score,
            defeat_score=defeat_score,
            total_score=total_score,
            time_alive=time_alive,
            gold=gold,
            damage_taken=damage_taken,
            n_timeout=n_timeout,
        )
        results[team_idx] = result

    return results


def count_max_alive_time(metrics: Dict[int, Metrics]) -> int:
    cnt = 0
    max_alive_time = Metrics.max(metrics.values())["TimeAlive"]
    for m in metrics.values():
        if m["TimeAlive"] == max_alive_time:
            cnt += 1
    return cnt


def compute_alive_score(
    quads: List[Tuple[str, int, int, float]], ) -> List[Tuple[str, float]]:
    if not quads:
        return []
    quads = sorted(quads, key=lambda x: (-x[1], -x[2], -x[3]))
    ret = []
    idx, cnt, prev = 0, 1, quads[0][1:]
    for quad in quads[1:]:
        x = quad[1:]
        if x == prev:
            cnt += 1
        else:
            score = np.mean(ALIVE_SCORE[idx:idx + cnt])
            ret.extend([(quads[i][0], score) for i in range(idx, idx + cnt)])
            idx += cnt
            cnt = 1
            prev = x
    if x == prev:
        score = np.mean(ALIVE_SCORE[idx:idx + cnt])
        ret.extend([(quads[i][0], score) for i in range(idx, idx + cnt)])
    return ret


def avg_results(results: List[Dict[int, TeamResult]]) -> Dict[int, TeamResult]:
    if not results:
        return {}
    all_results: Dict[int, List[TeamResult]] = {}
    for result_by_team in results:
        for team_idx, result in result_by_team.items():
            if team_idx not in all_results:
                all_results[team_idx] = []
            all_results[team_idx].append(result)
    avg_result_by_team: Dict[int, TeamResult] = {}
    for team_idx in all_results:
        ss = {
            key: np.mean([getattr(r, key) for r in all_results[team_idx]])
            for key in TeamResult.names()
        }
        avg_result_by_team[team_idx] = TeamResult(
            all_results[team_idx][0].policy_id, **ss)
    return avg_result_by_team


def topn_team_inds(results: List[Dict[int, TeamResult]],
                   n=1) -> List[List[int]]:
    if not results:
        return []

    r = []
    for result_by_team in results:
        values = [result.total_score for result in result_by_team.values()]
        values = sorted(list(set(values)), reverse=True)
        topn = []
        for v in values:
            if len(topn) >= n:
                break
            topn.extend([
                i for i in result_by_team if result_by_team[i].total_score == v
            ])
        r.append(topn)

    return r


def topn_counts(results: List[Dict[int, TeamResult]], n=1) -> Dict[int, int]:
    if not results:
        return {}

    r = {i: 0 for i in results[0]}
    topn_inds = topn_team_inds(results, n)
    for topn in topn_inds:
        for i in topn:
            r[i] += 1

    return r


def topn_probs(results: List[Dict[int, TeamResult]], n=1) -> Dict[int, float]:
    if not results:
        return {}

    topn_cnts = topn_counts(results, n)
    return {i: cnt / len(results) for i, cnt in topn_cnts.items()}


def topn_count_by_policy(results: List[Dict[int, TeamResult]],
                         n=1) -> Dict[str, int]:
    if not results:
        return {}

    r = {v.policy_id: 0 for v in results[0].values()}
    topn_inds = topn_team_inds(results, n)
    for rind, topn in enumerate(topn_inds):
        policy_ids = list(set([results[rind][tind].policy_id
                               for tind in topn]))
        for p in policy_ids:
            r[p] += 1

    return r


def topn_prob_by_policy(results: List[Dict[int, TeamResult]],
                        n=1) -> Dict[str, float]:
    if not results:
        return {}

    topn_cnts = topn_count_by_policy(results, n)
    return {p: cnt / len(results) for p, cnt in topn_cnts.items()}


def avg_result_by_policy(
        result_by_team: Dict[int, TeamResult]) -> Dict[str, TeamResult]:
    d = defaultdict(lambda: [])
    for result in result_by_team.values():
        d[result.policy_id].append(result)

    ret = {}
    for policy_id, results in d.items():
        ss = {
            key: np.mean([getattr(r, key) for r in results])
            for key in TeamResult.names()
        }
        ret[policy_id] = TeamResult(policy_id=policy_id, **ss)

    return ret


def n_timeout(results: List[Dict[int, TeamResult]]) -> Dict[str, int]:
    d = defaultdict(lambda: 0)
    for result_by_team in results:
        for result in result_by_team.values():
            d[result.policy_id] += result.n_timeout
    return d
