from typing import Dict, Tuple, Any, Union
import torch
import diffusion_planner.model.diffusion_utils.dpm_solver_pytorch as dpm


def build_dpm_solver_bundle(
    model: torch.nn.Module,
    other_model_params: Dict = {},
    noise_schedule_params: Dict = {},
    model_wrapper_params: Dict = {},
    dpm_solver_params: Dict = {},
):
    noise_schedule = dpm.NoiseScheduleVP(schedule="linear", **noise_schedule_params)
    model_fn = dpm.model_wrapper(
        model,
        noise_schedule,
        model_type=model.model_type,
        model_kwargs=other_model_params,
        **model_wrapper_params,
    )
    dpm_solver = dpm.DPM_Solver(
        model_fn, noise_schedule, algorithm_type="dpmsolver++", **dpm_solver_params
    )
    return model_fn, noise_schedule, dpm_solver


def dpm_sampler(
        model: torch.nn.Module, 
        x_T, 
        other_model_params: Dict={}, 
        diffusion_steps=10,

        noise_schedule_params: Dict = {},
        model_wrapper_params: Dict = {},
        dpm_solver_params: Dict = {},
        sample_params: Dict = {}
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    
    with torch.no_grad():
        sample_params = dict(sample_params)
        first_model_output = sample_params.pop("first_model_output", None)
        nfe_holder = sample_params.pop("nfe_holder", None)
        return_stats = sample_params.pop("return_stats", False)

        model_fn, noise_schedule, dpm_solver = build_dpm_solver_bundle(
            model,
            other_model_params=other_model_params,
            noise_schedule_params=noise_schedule_params,
            model_wrapper_params=model_wrapper_params,
            dpm_solver_params=dpm_solver_params,
        )

        nfe_counter = [0] if (nfe_holder is not None or return_stats) else None

        sample_dpm = dpm_solver.sample(
            x_T,
            steps=diffusion_steps,
            order=2,
            skip_type="logSNR",
            method="multistep",
            denoise_to_zero=True,
            first_model_output=first_model_output,
            nfe_counter=nfe_counter,
            **sample_params
        )

        stats = {"nfe": int(nfe_counter[0]) if nfe_counter is not None else None}
        if nfe_holder is not None:
            nfe_holder["nfe"] = stats["nfe"]

    if return_stats:
        return sample_dpm, stats
    return sample_dpm
