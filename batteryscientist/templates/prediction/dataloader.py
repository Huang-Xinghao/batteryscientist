"""IC2ML loaders backed by BatteryLife PKLs and fixed cell splits."""

from data_provider.batterylife import BatteryLifeDataset, batterylife_trainloader


def NCA_trainloader(args):
    return batterylife_trainloader(args)


def NCM_trainloader(args):
    return batterylife_trainloader(args)


def NCMNCA_trainloader(args):
    return batterylife_trainloader(args)
