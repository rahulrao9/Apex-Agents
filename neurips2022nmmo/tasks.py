# Kills
def player_kills(realm, player):
    return player.history.playerKills


# Alive
def time_alive(realm, player):
    return player.history.timeAlive.val


# DamageTaken
def damage_taken(realm, player):
    return player.history.damage_received


# Profession
def melee_level(realm, player):
    return player.skills.melee.level.val


def range_level(realm, player):
    return player.skills.range.level.val


def mage_level(realm, player):
    return player.skills.mage.level.val


def fishing_level(realm, player):
    return player.skills.fishing.level.val


def herbalism_level(realm, player):
    return player.skills.herbalism.level.val


def prospecting_level(realm, player):
    return player.skills.prospecting.level.val


def carving_level(realm, player):
    return player.skills.carving.level.val


def alchemy_level(realm, player):
    return player.skills.alchemy.level.val


# Equipment
def hat_level(realm, player):
    x = player.equipment.hat
    if x is None:
        return 0
    else:
        return x.level.val


def top_level(realm, player):
    x = player.equipment.top
    if x is None:
        return 0
    else:
        return x.level.val


def bottom_level(realm, player):
    x = player.equipment.bottom
    if x is None:
        return 0
    else:
        return x.level.val


def held_level(realm, player):
    x = player.equipment.held
    if x is None:
        return 0
    else:
        return x.level.val


def ammunition_level(realm, player):
    x = player.equipment.ammunition
    if x is None:
        return 0
    else:
        return x.level.val


def melee_attack(realm, player):
    return player.equipment.melee_attack


def range_attack(realm, player):
    return player.equipment.range_attack


def mage_attack(realm, player):
    return player.equipment.mage_attack


def melee_defense(realm, player):
    return player.equipment.melee_defense


def range_defense(realm, player):
    return player.equipment.range_defense


def mage_defense(realm, player):
    return player.equipment.mage_defense


def equipment(realm, player):
    return player.equipment.total(lambda e: e.level)


# Item
def ration_consumed(realm, player):
    return player.ration_consumed


def poultice_consumed(realm, player):
    return player.poultice_consumed


def ration_level_consumed(realm, player):
    return player.ration_level_consumed


def poultice_level_consumed(realm, player):
    return player.poultice_level_consumed


# Gold
def gold(realm, player):
    return player.inventory.gold.quantity.val


def sells(realm, player):
    return player.sells


def buys(realm, player):
    return player.buys


def profession(realm, player):
    return max(player.skills.fishing.level.val,
               player.skills.herbalism.level.val,
               player.skills.prospecting.level.val,
               player.skills.carving.level.val,
               player.skills.alchemy.level.val, player.skills.mage.level.val,
               player.skills.range.level.val, player.skills.melee.level.val)
