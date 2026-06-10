from nmmo import config
from typing import Dict


class Team(object):
    id: str
    policy_id: str
    n_player: int
    env_config: config.Config
    n_timeout: int = 0

    def __init__(
        self,
        team_id: str,
        env_config: config.Config,
        **kwargs,
    ) -> None:
        self.id = team_id
        self.policy_id = kwargs.get("policy_id", team_id)
        self.n_player = env_config.PLAYER_N // len(env_config.PLAYERS)
        self.env_config = env_config

    def act(self, observations: Dict[int, dict]) -> Dict[int, dict]:
        raise NotImplementedError

    def reset(self) -> None:
        pass
