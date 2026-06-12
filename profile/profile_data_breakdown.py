"""Breakdown of observation_adapter (data processing) hotspots via cProfile."""
import os
import sys

import fiona  # noqa: F401  (must precede torch, see profile_baseline.py)
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "profile"))

import profile_baseline as pb
from nuplan.common.actor_state.state_representation import StateSE2


def main():
    map_api = pb.get_maps_api(pb.MAP_ROOT, pb.MAP_VERSION, pb.MAP_NAME)
    pose, rb = pb.find_start_pose(map_api)
    route_ids = pb.build_route(rb)
    cfg = pb.build_config()
    proc = pb.DataProcessor(cfg)
    rng = np.random.default_rng(0)

    hists = []
    for k in range(33):
        off = k * pb.EGO_SPEED * 0.1
        p = StateSE2(pose.x + np.cos(pose.heading) * off, pose.y + np.sin(pose.heading) * off, pose.heading)
        hists.append(pb.make_history(p, 1e6 + k * 1e5, rng))

    for k in range(3):  # warm up map cache
        proc.observation_adapter(hists[k], [], map_api, list(route_ids), device="cpu")

    import cProfile
    import io
    import pstats

    pr = cProfile.Profile()
    pr.enable()
    for k in range(3, 33):
        proc.observation_adapter(hists[k], [], map_api, list(route_ids), device="cpu")
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(30)
    print(s.getvalue())


if __name__ == "__main__":
    main()
