"""Second-life EV battery fleet at the 20 PV buses (extension of the RSS).

Each PV bus hosts a retired-EV battery with nominal power 35 % of the bus PV
capacity and a 4 h energy rating, derated by its state of health (SoH). For
the one-hour RSS the usable power is P_eff = SoH * P_nom in both directions
(an SoC window [20 %, 85 %] starting at 50 % covers one hour at P_eff).
The batteries enter the bilevel model as follower (dispatch) variables.
"""
import numpy as np

SOH = np.array([0.84, 0.82, 0.79, 0.81, 0.76, 0.78, 0.75, 0.80, 0.74, 0.72] * 2)
POWER_RATIO = 0.35


def bess_power(grid, power_ratio=POWER_RATIO, soh=SOH):
    caps = np.repeat(grid.profile[0, [2, 4, 6, 8, 10]], 4) * 0.25 / grid.base
    caps[15] *= 2.0
    return soh[:len(caps)] * power_ratio * caps
