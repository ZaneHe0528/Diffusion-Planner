"""
Profile Diffusion-Planner per-call latency in nuPlan closed-loop simulation,
split into 3 stages:
  1. data processing (observation_adapter: agents + map query, CPU)
  2. encoder forward (GPU)
  3. diffusion decoder sampling (GPU, DPM-Solver++)

No nuplan scenario DB / trained checkpoint required:
- real nuPlan map (las vegas) for the map-query path,
- synthetic ego history + agents (typical scene density),
- randomly initialized weights (identical FLOPs to trained ones).

Run with the navsim conda env (has nuplan-devkit + torch):
  /home/ubuntu/anaconda3/envs/navsim/bin/python profile/profile_baseline.py
"""
import json
import os
import sys
import time
import types

import numpy as np
# IMPORTANT: GDAL (fiona/geopandas, used by nuplan maps) must be imported BEFORE torch,
# otherwise it segfaults in this env.
import fiona  # noqa: F401

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

MAP_ROOT = os.path.expanduser("~/data/hezexiang/navsim_workspace/dataset/maps")
MAP_VERSION = "nuplan-maps-v1.0"
MAP_NAME = "us-nv-las-vegas-strip"

NUM_NEIGHBOR_VEHICLES = 30  # typical urban scene density
NUM_STATIC_OBJS = 5
EGO_SPEED = 5.0  # m/s
N_TIMED_CALLS = 30

# navsim env lacks mmengine (repo only uses mmengine.fileio for json/npz IO) -> minimal shim
try:
    import mmengine  # noqa: F401
except ImportError:
    _mm = types.ModuleType("mmengine")
    _fileio = types.ModuleType("mmengine.fileio")
    _fileio.get_text = lambda path: open(path).read()
    _fileio.get = lambda path: open(path, "rb").read()
    _mm.fileio = _fileio
    sys.modules["mmengine"] = _mm
    sys.modules["mmengine.fileio"] = _fileio

import torch

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import Point2D, StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks

from diffusion_planner.data_process.data_processor import DataProcessor
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
import diffusion_planner.model.module.decoder as decoder_mod
from diffusion_planner.model.diffusion_utils.sampling import dpm_sampler


# ---------------------------------------------------------------- config
def build_config():
    cfg = types.SimpleNamespace()
    # model defaults from train_predictor.py get_args()
    cfg.future_len = 80
    cfg.time_len = 21
    cfg.agent_state_dim = 11
    cfg.agent_num = 32
    cfg.static_objects_state_dim = 10
    cfg.static_objects_num = 5
    cfg.lane_len = 20
    cfg.lane_state_dim = 12
    cfg.lane_num = 70
    cfg.route_len = 20
    cfg.route_state_dim = 12
    cfg.route_num = 25
    cfg.encoder_drop_path_rate = 0.1
    cfg.decoder_drop_path_rate = 0.1
    cfg.encoder_depth = 3
    cfg.decoder_depth = 3
    cfg.num_heads = 6
    cfg.hidden_dim = 192
    cfg.diffusion_model_type = "x_start"
    cfg.predicted_neighbor_num = 10
    cfg.device = "cuda"
    cfg.normalization_file_path = os.path.join(REPO, "normalization.json")
    cfg.state_normalizer = StateNormalizer.from_json(cfg)
    cfg.observation_normalizer = ObservationNormalizer.from_json(cfg)
    cfg.guidance_fn = None
    return cfg


# ---------------------------------------------------------------- scene
def find_start_pose(map_api):
    """Pick a lane somewhere on the map and return (pose, roadblock)."""
    df = map_api._load_vector_map_layer("lanes_polygons")  # already in local UTM frame
    for idx in range(1000, len(df), 100):
        g = df.iloc[idx].geometry.centroid
        objs = map_api.get_proximal_map_objects(
            Point2D(g.x, g.y), 30.0, [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
        )
        roadblocks = objs[SemanticMapLayer.ROADBLOCK] + objs[SemanticMapLayer.ROADBLOCK_CONNECTOR]
        if not roadblocks:
            continue
        rb = roadblocks[0]
        lane = rb.interior_edges[0]
        path = lane.baseline_path.discrete_path
        pose = path[len(path) // 2]
        return pose, rb
    raise RuntimeError("no usable lane found")


def build_route(start_rb, depth=10):
    """Follow outgoing edges to build a route roadblock id chain."""
    ids = [start_rb.id]
    cur = start_rb
    for _ in range(depth):
        outs = cur.outgoing_edges
        if not outs:
            break
        cur = outs[0]
        ids.append(cur.id)
    return ids


def make_ego_state(pose, t_us):
    return EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(pose.x, pose.y, pose.heading),
        rear_axle_velocity_2d=StateVector2D(EGO_SPEED, 0.0),
        rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
        tire_steering_angle=0.0,
        time_point=TimePoint(int(t_us)),
        vehicle_parameters=get_pacifica_parameters(),
        is_in_auto_mode=True,
        angular_vel=0.0,
        angular_accel=0.0,
    )


def make_detections(pose, t_us, rng):
    """~30 vehicles + a few cones scattered around ego, consistent track tokens."""
    objs = []
    c, s = np.cos(pose.heading), np.sin(pose.heading)
    for i in range(NUM_NEIGHBOR_VEHICLES):
        # deterministic per-agent offset (same layout every frame, agents move with ego speed)
        lon = -40.0 + 80.0 * ((i * 37) % NUM_NEIGHBOR_VEHICLES) / NUM_NEIGHBOR_VEHICLES
        lat = -12.0 + 24.0 * ((i * 17) % NUM_NEIGHBOR_VEHICLES) / NUM_NEIGHBOR_VEHICLES
        x = pose.x + c * lon - s * lat
        y = pose.y + s * lon + c * lat
        box = OrientedBox(StateSE2(x, y, pose.heading), length=4.8, width=2.0, height=1.8)
        meta = SceneObjectMetadata(timestamp_us=int(t_us), token=f"a{i:03d}", track_id=i, track_token=f"tt{i:03d}")
        objs.append(
            Agent(
                tracked_object_type=TrackedObjectType.VEHICLE,
                oriented_box=box,
                velocity=StateVector2D(EGO_SPEED * c, EGO_SPEED * s),
                metadata=meta,
            )
        )
    for i in range(NUM_STATIC_OBJS):
        x = pose.x + c * (10 + 3 * i) - s * 4.0
        y = pose.y + s * (10 + 3 * i) + c * 4.0
        box = OrientedBox(StateSE2(x, y, pose.heading), length=0.5, width=0.5, height=0.8)
        meta = SceneObjectMetadata(timestamp_us=int(t_us), token=f"s{i:03d}", track_id=100 + i, track_token=f"st{i:03d}")
        objs.append(StaticObject(tracked_object_type=TrackedObjectType.TRAFFIC_CONE, oriented_box=box, metadata=meta))
    return DetectionsTracks(TrackedObjects(objs))


def make_history(pose, t0_us, rng, n_frames=21, dt=0.1):
    """History buffer mimicking SimulationHistoryBuffer for observation_adapter."""
    ego_states, observations = [], []
    for k in range(n_frames):
        # frames going forward in time, ego moving at constant speed, last frame at `pose`
        back = (n_frames - 1 - k) * dt * EGO_SPEED
        p = StateSE2(pose.x - np.cos(pose.heading) * back, pose.y - np.sin(pose.heading) * back, pose.heading)
        t_us = t0_us + k * dt * 1e6
        ego_states.append(make_ego_state(p, t_us))
        observations.append(make_detections(p, t_us, rng))
    return types.SimpleNamespace(
        current_state=(ego_states[-1], observations[-1]),
        observation_buffer=observations,
        ego_states=ego_states,
    )


# ---------------------------------------------------------------- timing
def time_cpu(fn, n=N_TIMED_CALLS, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return np.array(ts)


def time_gpu(fn, n=N_TIMED_CALLS, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return np.array(ts)


def stat(ts):
    return f"mean {ts.mean():7.2f} ms | median {np.median(ts):7.2f} ms | p90 {np.percentile(ts, 90):7.2f} ms"


def main():
    assert torch.cuda.is_available()
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    cfg = build_config()
    print(f"== GPU: {torch.cuda.get_device_name(0)}")

    # ---- map & scene
    print("== loading map ...")
    t0 = time.perf_counter()
    map_api = get_maps_api(MAP_ROOT, MAP_VERSION, MAP_NAME)
    pose, rb = find_start_pose(map_api)
    route_ids = build_route(rb)
    print(f"   map ready in {time.perf_counter() - t0:.1f}s, start=({pose.x:.1f},{pose.y:.1f}), route len={len(route_ids)}")

    processor = DataProcessor(cfg)

    # pre-build histories at slightly different ego positions (simulate replanning while moving)
    histories = []
    for k in range(N_TIMED_CALLS + 3):
        offset = k * EGO_SPEED * 0.1  # ego advances 0.1s per sim step
        p = StateSE2(pose.x + np.cos(pose.heading) * offset, pose.y + np.sin(pose.heading) * offset, pose.heading)
        histories.append(make_history(p, t0_us=1e6 + k * 1e5, rng=rng))

    # ---- stage 1: data processing (first call = cold map cache)
    t0 = time.perf_counter()
    inputs = processor.observation_adapter(histories[0], [], map_api, list(route_ids), device="cuda")
    cold_ms = (time.perf_counter() - t0) * 1e3
    print(f"\n[stage 1] observation_adapter cold call: {cold_ms:.1f} ms")

    it = iter(range(1, len(histories)))

    def run_data():
        k = next(it) % len(histories)
        processor.observation_adapter(histories[k], [], map_api, list(route_ids), device="cuda")

    # cycle through different ego positions
    counter = {"k": 0}

    def run_data():  # noqa: F811
        counter["k"] = (counter["k"] + 1) % len(histories)
        processor.observation_adapter(histories[counter["k"]], [], map_api, list(route_ids), device="cuda")

    ts_data = time_cpu(run_data)
    print(f"[stage 1] observation_adapter (data processing, CPU): {stat(ts_data)}")

    # ---- model
    model = Diffusion_Planner(cfg).eval().cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n== model params: {n_params / 1e6:.2f} M (random init, FLOPs identical to trained)")

    inputs = cfg.observation_normalizer(inputs)

    def run_norm():
        cfg.observation_normalizer(inputs)

    ts_norm = time_gpu(run_norm)

    with torch.no_grad():
        enc_out = model.encoder(inputs)

    def run_encoder():
        with torch.no_grad():
            model.encoder(inputs)

    ts_enc = time_gpu(run_encoder)
    print(f"[stage 2] encoder forward (GPU):  {stat(ts_enc)}")
    print(f"          obs normalizer:         {stat(ts_norm)}")

    # ---- stage 3: diffusion decoder, sweep sampling steps
    orig_sampler = dpm_sampler
    nfe = {"n": 0}
    orig_dit_forward = model.decoder.decoder.dit.forward

    def counting_forward(*a, **k):
        nfe["n"] += 1
        return orig_dit_forward(*a, **k)

    model.decoder.decoder.dit.forward = counting_forward

    results = {}
    for steps in [10, 5, 3, 2]:  # dpm_solver order=2 requires steps >= 2
        decoder_mod.dpm_sampler = lambda *a, **k: orig_sampler(*a, diffusion_steps=steps, **k)

        def run_decoder():
            with torch.no_grad():
                model.decoder(enc_out, inputs)

        nfe["n"] = 0
        run_decoder()
        nfe_per_call = nfe["n"]
        ts = time_gpu(run_decoder)
        results[steps] = (ts, nfe_per_call)
        tag = " (baseline)" if steps == 10 else ""
        print(f"[stage 3] decoder, steps={steps:2d} (NFE={nfe_per_call:2d}){tag}: {stat(ts)}")
    decoder_mod.dpm_sampler = orig_sampler

    # single NFE cost (one DiT forward)
    B = inputs["ego_current_state"].shape[0]
    P = 1 + cfg.predicted_neighbor_num
    x = torch.randn(B, P, (cfg.future_len + 1) * 4, device="cuda")
    t = torch.rand(B, device="cuda")
    nb_mask = inputs["neighbor_agents_past"][:, : cfg.predicted_neighbor_num, -1, :4]
    nb_mask = torch.sum(torch.ne(nb_mask, 0), dim=-1) == 0

    def run_dit_once():
        with torch.no_grad():
            orig_dit_forward(x, t, enc_out["encoding"], inputs["route_lanes"], nb_mask)

    ts_dit = time_gpu(run_dit_once)
    print(f"[stage 3] single DiT forward (1 NFE): {stat(ts_dit)}")

    # ---- summary
    ts_dec10 = results[10][0]
    total = ts_data.mean() + ts_norm.mean() + ts_enc.mean() + ts_dec10.mean()
    print("\n================ SUMMARY (per planner call, baseline steps=10) ================")
    for name, v in [
        ("data processing (CPU)", ts_data.mean()),
        ("obs normalizer", ts_norm.mean()),
        ("encoder (GPU)", ts_enc.mean()),
        (f"diffusion decoder (GPU, NFE={results[10][1]})", ts_dec10.mean()),
    ]:
        print(f"  {name:38s} {v:8.2f} ms   {100 * v / total:5.1f} %")
    print(f"  {'TOTAL':38s} {total:8.2f} ms")

    print("\n---- decoder-only scaling ----")
    base = results[10][0].mean()
    for steps, (ts, n) in results.items():
        dec_total = ts_data.mean() + ts_norm.mean() + ts_enc.mean() + ts.mean()
        print(
            f"  steps={steps:2d} NFE={n:2d}: decoder {ts.mean():7.2f} ms "
            f"({base / ts.mean():4.2f}x vs baseline) | end-to-end {dec_total:7.2f} ms ({total / dec_total:4.2f}x)"
        )

    out = {
        "gpu": torch.cuda.get_device_name(0),
        "data_processing_ms": ts_data.tolist(),
        "normalizer_ms": ts_norm.tolist(),
        "encoder_ms": ts_enc.tolist(),
        "decoder_ms": {str(k): v[0].tolist() for k, v in results.items()},
        "decoder_nfe": {str(k): v[1] for k, v in results.items()},
        "dit_single_nfe_ms": ts_dit.tolist(),
        "cold_data_processing_ms": cold_ms,
    }
    out_path = os.path.join(REPO, "profile", "profile_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nresults saved to {out_path}")


if __name__ == "__main__":
    main()
