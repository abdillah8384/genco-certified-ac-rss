"""THE GENCO CHALLENGE 2026 - extension: second-life EV batteries in the RSS.

Twenty retired-EV batteries (one per PV bus, 35 % of the bus PV capacity,
4 h, state of health 0.72-0.84) are added to the robust-secure-set bilevel as
dispatch (follower) resources. For every hour both RSS limits are solved with
the two-phase Benders decomposition with and without batteries, and the
worst-case operating points are verified with exact AC power flow
(gridfm-datakit + lightsim2grid) in the base case and both contingencies.

Output: results/e5_bess.csv, results/e5_bess_width.csv
Run:    python genco_ac_rssr_slbess.py
"""
import os
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("PYTHONPATH", HERE)
runpy.run_path(os.path.join(HERE, "experiments", "e5_bess.py"), run_name="__main__")
