
import json
import time
import warnings
import sys
from pathlib import Path

import torch
import numpy as np
from typing import Deque, Dict, List, Optional, Type

warnings.filterwarnings("ignore")

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.observation_type import Observation, DetectionsTracks
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, PlannerInitialization, PlannerInput
)

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.data_process.data_processor import DataProcessor
from diffusion_planner.utils.config import Config

GATE_V2_ROOT = Path(__file__).resolve().parents[2] / "0hzxcode" / "gate_v2"
if str(GATE_V2_ROOT) not in sys.path:
    sys.path.insert(0, str(GATE_V2_ROOT))

DIAG_ROOT = Path(__file__).resolve().parents[2] / "0hzxcode" / "gate_v2_output" / "val14_closedloop"
_scenario_counter = 0
_PLANNER_MODEL_CACHE: Dict[tuple, Diffusion_Planner] = {}


def identity(ego_state, predictions):
    return predictions


def _load_cached_planner_model(
    config: Config,
    ckpt_path: Optional[str],
    device: str,
    enable_ema: bool,
) -> Diffusion_Planner:
    key = (
        str(ckpt_path) if ckpt_path is not None else "__random__",
        device,
        bool(enable_ema),
        int(config.future_len),
        int(config.predicted_neighbor_num),
    )
    if key in _PLANNER_MODEL_CACHE:
        return _PLANNER_MODEL_CACHE[key]

    planner = Diffusion_Planner(config)
    if ckpt_path is not None:
        state_dict: Dict = torch.load(ckpt_path, map_location=device)

        if enable_ema:
            state_dict = state_dict["ema_state_dict"]
        elif "model" in state_dict.keys():
            state_dict = state_dict["model"]

        model_state_dict = {
            k[len("module.") :]: v
            for k, v in state_dict.items()
            if k.startswith("module.")
        }
        planner.load_state_dict(model_state_dict)
    else:
        print("load random model")

    planner.eval()
    planner = planner.to(device)
    _PLANNER_MODEL_CACHE[key] = planner
    return planner


class DiffusionPlanner(AbstractPlanner):
    def __init__(
            self,
            config: Config,
            ckpt_path: str,

            past_trajectory_sampling: TrajectorySampling, 
            future_trajectory_sampling: TrajectorySampling,

            enable_ema: bool = True,
            device: str = "cpu",
            gate_ckpt_path: Optional[str] = None,
            enable_warmstart: bool = False,
        ):

        assert device in ["cpu", "cuda"], f"device {device} not supported"
        if device == "cuda":
            assert torch.cuda.is_available(), "cuda is not available"
            
        self._future_horizon = future_trajectory_sampling.time_horizon # [s] 
        self._step_interval = future_trajectory_sampling.time_horizon / future_trajectory_sampling.num_poses # [s]
        
        self._config = config
        self._ckpt_path = ckpt_path

        self._past_trajectory_sampling = past_trajectory_sampling
        self._future_trajectory_sampling = future_trajectory_sampling

        self._ema_enabled = enable_ema
        self._device = device

        self._planner = _load_cached_planner_model(config, ckpt_path, device, enable_ema)

        self.data_processor = DataProcessor(config)
        
        self.observation_normalizer = config.observation_normalizer

        self._gate_controller = None
        self._enable_warmstart = enable_warmstart
        if enable_warmstart and gate_ckpt_path:
            from inference import GateWarmStartController

            self._gate_controller = GateWarmStartController(
                gate_ckpt_path,
                device=device,
                base_steps=10,
                enabled=True,
                future_len=config.future_len,
                predicted_neighbor_num=config.predicted_neighbor_num,
            )
            self._gate_controller.set_state_normalizer(config.state_normalizer)
        self._last_gate_meta: Dict = {}
        self._scenario_idx = 0

    def name(self) -> str:
        """
        Inherited.
        """
        return "diffusion_planner"
    
    def observation_type(self) -> Type[Observation]:
        """
        Inherited.
        """
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        """
        Inherited.
        """
        self._map_api = initialization.map_api
        self._route_roadblock_ids = initialization.route_roadblock_ids

        self._planner.eval()
        self._initialization = initialization

        if self._gate_controller is not None:
            self._gate_controller.reset()
        global _scenario_counter
        _scenario_counter += 1
        self._scenario_idx = _scenario_counter

    def planner_input_to_model_inputs(self, planner_input: PlannerInput) -> tuple[Dict[str, torch.Tensor], List[str]]:
        history = planner_input.history
        traffic_light_data = list(planner_input.traffic_light_data)
        model_inputs, selected_neighbor_tokens = self.data_processor.observation_adapter(
            history,
            traffic_light_data,
            self._map_api,
            self._route_roadblock_ids,
            self._device,
            return_neighbor_tokens=True,
        )
        return model_inputs, selected_neighbor_tokens

    def outputs_to_trajectory(self, outputs: Dict[str, torch.Tensor], ego_state_history: Deque[EgoState]) -> List[InterpolatableState]:    

        predictions = outputs['prediction'][0, 0].detach().cpu().numpy().astype(np.float64) # T, 4
        heading = np.arctan2(predictions[:, 3], predictions[:, 2])[..., None]
        predictions = np.concatenate([predictions[..., :2], heading], axis=-1) 

        states = transform_predictions_to_states(predictions, ego_state_history, self._future_horizon, self._step_interval)

        return states

    def _append_diag(self, rec: Dict) -> None:
        if not self._enable_warmstart:
            return
        DIAG_ROOT.mkdir(parents=True, exist_ok=True)
        out_path = DIAG_ROOT / "frames.jsonl"
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Inherited.
        """
        raw_inputs, selected_neighbor_tokens = self.planner_input_to_model_inputs(current_input)
        inputs = self.observation_normalizer(raw_inputs)

        warmstart = None
        gate_meta: Dict = {}
        if self._gate_controller is not None:
            from inference import anchor_from_ego_state, build_ego_history_from_ego_states

            ego_states = list(current_input.history.ego_states)
            ego_history = build_ego_history_from_ego_states(ego_states)
            anchor = anchor_from_ego_state(ego_states[-1])
            ops, gate_meta = self._gate_controller.predict_ops(raw_inputs, ego_history=ego_history)
            self._last_gate_meta = gate_meta

            ego_current = inputs["ego_current_state"][:, None, :4]
            neighbors_current = inputs["neighbor_agents_past"][:, : self._config.predicted_neighbor_num, -1, :4]
            current_states = torch.cat([ego_current, neighbors_current], dim=1)
            warmstart = self._gate_controller.prepare_warmstart(
                ops,
                current_states,
                anchor,
                neighbor_tokens=selected_neighbor_tokens,
                neighbor_agents_past=raw_inputs["neighbor_agents_past"],
                sde=self._planner.sde,
            )

        t0 = time.perf_counter()
        if torch.cuda.is_available() and self._device == "cuda":
            torch.cuda.synchronize()
        _, outputs = self._planner(inputs, warmstart=warmstart)
        if torch.cuda.is_available() and self._device == "cuda":
            torch.cuda.synchronize()
        decoder_ms = (time.perf_counter() - t0) * 1000.0

        if self._gate_controller is not None and "x0_norm" in outputs:
            from inference import anchor_from_ego_state

            anchor = anchor_from_ego_state(list(current_input.history.ego_states)[-1])
            self._gate_controller.on_decode(
                outputs["x0_norm"],
                anchor,
                neighbor_tokens=selected_neighbor_tokens,
            )

        ws_meta = outputs.get("warmstart_meta", {})
        if self._enable_warmstart:
            rec = {
                "scenario_idx": self._scenario_idx,
                "decoder_ms": round(decoder_ms, 3),
                "nfe": ws_meta.get("nfe"),
                "d_hat_m": gate_meta.get("d_hat_m"),
                "level": gate_meta.get("level", gate_meta.get("level_from_gate")),
                "t_start": gate_meta.get("t_start"),
                "steps": gate_meta.get("steps"),
                "forced_full": bool(gate_meta.get("forced_full", False) or ws_meta.get("forced_full", False)),
                "cache_miss": bool(warmstart.get("cache_miss", False)) if warmstart is not None else False,
                "passive_fallback": bool(ws_meta.get("passive_fallback", False)),
                "d_meas_m": ws_meta.get("d_meas_m"),
                "epsilon_m": ws_meta.get("epsilon_m"),
                "neighbor_score": gate_meta.get("neighbor_score"),
                "level_bump": gate_meta.get("level_bump"),
            }
            self._append_diag(rec)

        trajectory = InterpolatedTrajectory(
            trajectory=self.outputs_to_trajectory(outputs, current_input.history.ego_states)
        )

        return trajectory
