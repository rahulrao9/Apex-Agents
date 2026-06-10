import nmmo
from nmmo import entity
from typing import List, Sequence, Callable

from neurips2022nmmo import tasks


class Metrics(dict):
    @classmethod
    def names(cls) -> List[str]:
        return [
            "PlayerDefeats",
            "TimeAlive",
            "DamageTaken",
            "Profession",
            "MeleeLevel",
            "RangeLevel",
            "MageLevel",
            "FishingLevel",
            "HerbalismLevel",
            "ProspectingLevel",
            "CarvingLevel",
            "AlchemyLevel",
            "HatLevel",
            "TopLevel",
            "BottomLevel",
            "HeldLevel",
            "AmmunitionLevel",
            "MeleeAttack",
            "RangeAttack",
            "MageAttack",
            "MeleeDefense",
            "RangeDefense",
            "MageDefense",
            "Equipment",
            "RationConsumed",
            "PoulticeConsumed",
            "RationLevelConsumed",
            "PoulticeLevelConsumed",
            "Gold",
            "Sells",
            "Buys",
        ]

    @classmethod
    def collect(cls, env: nmmo.Env, player: entity.Player) -> "Metrics":
        realm = env.realm
        return Metrics(
            **{
                "PlayerDefeats":
                float(tasks.player_kills(realm, player)),
                "TimeAlive":
                float(tasks.time_alive(realm, player)),
                "DamageTaken":
                float(tasks.damage_taken(realm, player)),
                "Profession":
                float(tasks.profession(realm, player)),
                "MeleeLevel":
                float(tasks.melee_level(realm, player)),
                "RangeLevel":
                float(tasks.range_level(realm, player)),
                "MageLevel":
                float(tasks.mage_level(realm, player)),
                "FishingLevel":
                float(tasks.fishing_level(realm, player)),
                "HerbalismLevel":
                float(tasks.herbalism_level(realm, player)),
                "ProspectingLevel":
                float(tasks.prospecting_level(realm, player)),
                "CarvingLevel":
                float(tasks.carving_level(realm, player)),
                "AlchemyLevel":
                float(tasks.alchemy_level(realm, player)),
                "HatLevel":
                float(tasks.hat_level(realm, player)),
                "TopLevel":
                float(tasks.top_level(realm, player)),
                "BottomLevel":
                float(tasks.bottom_level(realm, player)),
                "HeldLevel":
                float(tasks.held_level(realm, player)),
                "AmmunitionLevel":
                float(tasks.ammunition_level(realm, player)),
                "MeleeAttack":
                float(tasks.melee_attack(realm, player)),
                "RangeAttack":
                float(tasks.range_attack(realm, player)),
                "MageAttack":
                float(tasks.mage_attack(realm, player)),
                "MeleeDefense":
                float(tasks.melee_defense(realm, player)),
                "RangeDefense":
                float(tasks.range_defense(realm, player)),
                "MageDefense":
                float(tasks.mage_defense(realm, player)),
                "Equipment":
                float(tasks.equipment(realm, player)),
                "RationConsumed":
                float(tasks.ration_consumed(realm, player)),
                "PoulticeConsumed":
                float(tasks.poultice_consumed(realm, player)),
                "RationLevelConsumed":
                float(tasks.ration_level_consumed(realm, player)),
                "PoulticeLevelConsumed":
                float(tasks.poultice_level_consumed(realm, player)),
                "Gold":
                float(tasks.gold(realm, player)),
                "Sells":
                float(tasks.sells(realm, player)),
                "Buys":
                float(tasks.buys(realm, player)),
            })

    @classmethod
    def sum(cls, metrices: Sequence["Metrics"]) -> "Metrics":
        return cls.reduce(sum, metrices)

    @classmethod
    def max(cls, metrices: Sequence["Metrics"]) -> "Metrics":
        return cls.reduce(max, metrices)

    @classmethod
    def min(cls, metrices: Sequence["Metrics"]) -> "Metrics":
        return cls.reduce(min, metrices)

    @classmethod
    def avg(cls, metrices: Sequence["Metrics"]) -> "Metrics":
        return cls.reduce(lambda x: sum(x) / len(x) if len(x) else 0, metrices)

    @classmethod
    def reduce(cls, func: Callable,
               metrices: Sequence["Metrics"]) -> "Metrics":
        names = cls.names()
        values = [[m[name] for m in metrices] for name in names]
        return Metrics(**dict(zip(names, list(map(func, values)))))
