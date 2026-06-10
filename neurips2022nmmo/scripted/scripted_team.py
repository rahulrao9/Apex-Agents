import nmmo
from typing import Any, Dict, Type, List

from neurips2022nmmo.evaluation.team import Team
from neurips2022nmmo.scripted import baselines
import numpy as np


def setup():
    from nmmo.systems import item
    from nmmo.io.action import Price
    for itm in [
            item.Gold,
            item.Hat,
            item.Top,
            item.Bottom,
            item.Sword,
            item.Bow,
            item.Wand,
            item.Rod,
            item.Gloves,
            item.Pickaxe,
            item.Chisel,
            item.Arcane,
            item.Scrap,
            item.Shard,
            item.Shaving,
            item.Ration,
            item.Poultice,
    ]:

        item.ItemID.register(itm, itm.ITEM_ID)
    Price.init(None)


setup()


class ScriptedTeam(Team):
    agent_klass: Type = None
    agents: List[baselines.Scripted]

    def __init__(
        self,
        team_id: str,
        env_config: nmmo.config.Config,
        **kwargs,
    ) -> None:
        if "policy_id" not in kwargs:
            kwargs["policy_id"] = self.__class__.__name__
        super().__init__(team_id, env_config, **kwargs)
        self.reset()

    def reset(self):
        assert self.agent_klass
        self.agents = []
        for i in range(self.n_player):
            idx = i % len(self.agent_klass)
            agent = self.agent_klass[idx](self.env_config, i)
            self.agents.append(agent)

    def act(self, observations: Dict[Any, dict]) -> Dict[int, dict]:
        if "stat" in observations:
            stat = observations.pop("stat")
        actions = {i: self.agents[i](obs) for i, obs in observations.items()}
        for i in actions:
            for atn, args in actions[i].items():
                for arg, val in args.items():
                    if arg.argType == nmmo.action.Fixed:
                        actions[i][atn][arg] = arg.edges.index(val)
                    elif arg == nmmo.action.Target:
                        actions[i][atn][arg] = self.get_target_index(
                            val, self.agents[i].ob.agents)
                    elif atn in (nmmo.action.Sell,
                                 nmmo.action.Use) and arg == nmmo.action.Item:
                        actions[i][atn][arg] = self.get_item_index(
                            val, self.agents[i].ob.items)
                    elif atn == nmmo.action.Buy and arg == nmmo.action.Item:
                        actions[i][atn][arg] = self.get_item_index(
                            val, self.agents[i].ob.market)
        return actions

    @staticmethod
    def get_item_index(instance: int, items: np.ndarray) -> int:
        for i, itm in enumerate(items):
            id_ = nmmo.scripting.Observation.attribute(itm,
                                                       nmmo.Serialized.Item.ID)
            if id_ == instance:
                return i
        raise ValueError(f"Instance {instance} not found")

    @staticmethod
    def get_target_index(target: int, agents: np.ndarray) -> int:
        targets = [
            x for x in [
                nmmo.scripting.Observation.attribute(
                    agent, nmmo.Serialized.Entity.ID) for agent in agents
            ] if x
        ]
        return targets.index(target)


class RandomTeam(ScriptedTeam):
    agent_klass = [baselines.Random]


class FisherTeam(ScriptedTeam):
    agent_klass = [baselines.Fisher]


class HerbalistTeam(ScriptedTeam):
    agent_klass = [baselines.Herbalist]


class ProspectorTeam(ScriptedTeam):
    agent_klass = [baselines.Prospector]


class CarverTeam(ScriptedTeam):
    agent_klass = [baselines.Carver]


class AlchemistTeam(ScriptedTeam):
    agent_klass = [baselines.Alchemist]


class MeleeTeam(ScriptedTeam):
    agent_klass = [baselines.Melee]


class RangeTeam(ScriptedTeam):
    agent_klass = [baselines.Range]


class MageTeam(ScriptedTeam):
    agent_klass = [baselines.Mage]


class MixtureTeam(ScriptedTeam):
    agent_klass = [
        baselines.Fisher,
        baselines.Herbalist,
        baselines.Prospector,
        baselines.Carver,
        baselines.Alchemist,
        baselines.Melee,
        baselines.Range,
        baselines.Mage,
    ]


class CombatTeam(ScriptedTeam):
    agent_klass = [
        baselines.Melee,
        baselines.Melee,
        baselines.Melee,
        baselines.Range,
        baselines.Range,
        baselines.Range,
        baselines.Mage,
        baselines.Mage,
    ]
