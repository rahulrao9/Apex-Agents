# monkey patch nmmo
from nmmo.entity import Player


def _packet(self):
    data = super(Player, self).packet()

    data["entID"] = self.entID
    data["annID"] = self.population

    data["base"] = self.base.packet()
    data["resource"] = self.resources.packet()
    data["skills"] = self.skills.packet()
    data["inventory"] = self.inventory.packet()

    # WebViewer only
    data["metrics"] = {
        "PlayerDefeats": self.history.playerKills,
        "TimeAlive": self.history.timeAlive.val,
        "Gold": self.inventory.gold.quantity.val,
        "DamageTaken": self.history.damage_received,
    }

    return data


def monkey_patch():
    Player.packet = _packet
