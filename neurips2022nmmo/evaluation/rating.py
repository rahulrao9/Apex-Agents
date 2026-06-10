import openskill
import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional

from neurips2022nmmo.evaluation.team import Team


def _policy_ids(teams: List[Team]) -> List[str]:
    return list(set([team.policy_id for team in teams]))


class RatingSystem:
    ratings: Dict[str, openskill.Rating]
    baselines: Dict[str, openskill.Rating]

    def __init__(
        self,
        teams: List[Team],
        mu: float = 1000,
        sigma: float = 1000 / 3,
        baselines: Optional[Dict[str, float]] = None,
    ) -> None:
        policy_ids = _policy_ids(teams)
        self.ratings = {}
        self.baselines = {}
        for policy_id in policy_ids:
            if baselines and policy_id in baselines:
                rating = openskill.Rating(baselines[policy_id], sigma=1e-10)
                self.baselines[policy_id] = rating
            else:
                rating = openskill.Rating(mu, sigma)
            self.ratings[policy_id] = rating

    def update(self, teams: List[Team], scores: List[float]) -> None:
        policy_scores: Dict[str, List[float]] = defaultdict(lambda: [])
        for team, score in zip(teams, scores):
            policy_scores[team.policy_id].append(score)

        mean_scores = [np.mean(scores) for scores in policy_scores.values()]
        # from high to low
        sorted_scores = sorted(mean_scores, reverse=True)
        # same rank if scores are the same
        ranks = [sorted_scores.index(score) for score in mean_scores]

        ratings: List[List[openskill.Rating]] = [[self.ratings[p]]
                                                 for p in policy_scores]
        ratings: List[List[List[float]]] = openskill.rate(ratings, rank=ranks)
        ratings: List[openskill.Rating] = [
            openskill.create_rating(team[0]) for team in ratings
        ]

        for policy_id, rating in zip(policy_scores, ratings):
            self.ratings[policy_id] = rating

    def get_team_ratings(self,
                         teams: List[Team]) -> Dict[int, openskill.Rating]:
        return {
            i: self.ratings[team.policy_id]
            for i, team in enumerate(teams)
        }
