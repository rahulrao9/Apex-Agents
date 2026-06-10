import nmmo
from nmmo import entity
from typing import Dict, List, Tuple, Any

from neurips2022nmmo.env.metrics import Metrics
from neurips2022nmmo.env.stat import Stat
from loguru import logger


class TeamBasedEnv(object):
    players: Dict[int, entity.Player] = {}
    player_team_map: Dict[int, int] = {}
    team_players_map: Dict[int, List[int]] = {}

    def __init__(self, config: nmmo.config.Config) -> None:
        self._env = nmmo.Env(config)

    def __getattr__(self, __name: str) -> Any:
        if __name not in self.__dict__:
            return getattr(self._env, __name)

    def reset(self) -> Dict[int, Dict[int, dict]]:
        self.players.clear()
        self.player_team_map.clear()
        self.team_players_map.clear()
        self.obs_stat = None

        observations = self._env.reset(None, True)
        for player in self._env.realm.players.entities.values():
            player: entity.Player
            self.players[player.entID] = player
            self.player_team_map[player.entID] = player.pop
            if player.pop not in self.team_players_map:
                self.team_players_map[player.pop] = []
            self.team_players_map[player.pop].append(player.entID)

        return self._postprocess(observations)

    def step(
        self,
        actions_by_team: Dict[int, Dict[int, dict]],
    ) -> Tuple[Dict[int, Dict[int, dict]], Dict[int, Dict[int, int]], Dict[
            int, Dict[int, bool]], Dict[int, Dict[int, dict]]]:
        # merge actions
        actions = {}
        for team_idx, team_actions in actions_by_team.items():
            player_ids = self.team_players_map[team_idx]
            for i, action in team_actions.items():
                # avoid invalid player id
                if not isinstance(i, int):
                    logger.error(f"Invalid player index: {i}")
                    continue
                if i >= 0 and i < len(player_ids):
                    actions[player_ids[i]] = action

        observations, rewards, dones, infos = self._env.step(actions)

        # delete the observations of the done players
        for player_id, done in dones.items():
            if done and player_id in observations:
                del observations[player_id]

        return (
            self._postprocess(observations),
            self._split_by_team(rewards),
            self._split_by_team(dones),
            self._split_by_team(infos),
        )

    def _postprocess(self, observations):
        observations_by_team = self._split_by_team(observations)
        stat = self.stat_by_team()
        obs_stat = {}
        for team_idx in stat.keys():
            if team_idx in observations_by_team:
                alive = len(observations_by_team[team_idx])
            else:
                alive = 0
            defeats = stat[team_idx]["sum"]["PlayerDefeats"]
            profession = stat[team_idx]["avg"]["Profession"]
            obs_stat[team_idx] = {
                "AlivePlayers": alive,
                "PlayerDefeats": defeats,
                "Profession": profession
            }
        for team_idx in observations_by_team.keys():
            observations_by_team[team_idx]["stat"] = obs_stat
        self.obs_stat = obs_stat
        return observations_by_team

    def metrices_by_team(self) -> Dict[int, Dict[int, Metrics]]:
        metrices: Dict[int, Metrics] = {}
        for player in self.players.values():
            metrices[player.entID] = Metrics.collect(self, player)
        return self._split_by_team(metrices)

    def stat_by_team(self) -> Dict[int, Stat]:
        stat_by_team = {}
        for team_idx, metrices in self.metrices_by_team().items():
            stat_by_team[team_idx] = Stat.from_metrices(metrices.values())
        return stat_by_team

    def _split_by_team(self, xs: Dict[int, Any]) -> Dict[int, Dict[int, Any]]:
        xs_by_team = {}
        for player_id, x in xs.items():
            team_idx = self.player_team_map[player_id]
            if team_idx not in xs_by_team:
                xs_by_team[team_idx] = {}
            xs_by_team[team_idx][self.team_players_map[team_idx].index(
                player_id)] = x
        return xs_by_team
