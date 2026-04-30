import os
import random
from typing import Optional, Dict, Any

import json
import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
from scipy.signal import find_peaks

import pytensor.tensor as tt

from tools.utils import return_efficiency_filename_path
from tools.interpolators import AttenuationInterpolator
from tools.tensor_interpolator import interpolate1d


DETECTOR = "NaIR"
NUM_SOURCE = 2
SCALE = 1e6
BRANCH_RATIO = 0.851
AIR_DENSITY = 1.243
CORRECTION_COEFFICIENT = 0.1
SOURCE_ENERGY = 661.657
DET_EFF = 0.021588808459214504

attenuation_df = pd.read_csv('./tools/attenuation_table.csv')
attenuation_interpolator = AttenuationInterpolator(attenuation_df)
MU_AIR = attenuation_interpolator.interpolate(SOURCE_ENERGY)
MU_AIR *= CORRECTION_COEFFICIENT * AIR_DENSITY


def set_parameters(raw_data, params, idxs=None, n_sources=NUM_SOURCE):
    cps = raw_data.ROI_P
    if idxs:
        peaks = idxs
    else:
        distance = len(cps) // 2
        peaks_found = False
        while distance > 0:
            peaks, _  = find_peaks(cps, distance=distance, prominence=float(np.std(cps)))
            if len(peaks) == n_sources:
                peaks_found = True
                break
            else:
                # print("[*] Peaks not found. Retrying...")
                distance -= max(1, distance // 5)

    if len(peaks) == 1:
        peaks = np.append(peaks, peaks[0])

    bkg_array = raw_data.ROI_BR
    params_ = {
        "MEAN_BKG_CPS": bkg_array.mean(),
        "BKG_STD": bkg_array.std(),
        "X_POS": raw_data["East"].values,
        "Y_POS": raw_data["North"].values,
        "MAX_X": [],
        "MAX_Y": [],
        "max_cps": [],
        "cps": cps.values,
        "LOWER_X": -params["size"][0],
        "UPPER_X": params["size"][0],
        "LOWER_Y": -params["size"][1],
        "UPPER_Y": params["size"][1],
        "indexes": peaks
    }
    for _, idx in enumerate(peaks):
        params_["MAX_X"].append(raw_data.East[idx])
        params_["MAX_Y"].append(raw_data.North[idx])
        params_["max_cps"].append(raw_data.ROI_P[idx])

    return params_, peaks_found


def mean_cps_tt(**kwargs):

    src_x = kwargs["x"]
    src_y = kwargs["y"]
    acts = kwargs["acts"]

    x_pos = kwargs["x_position"]
    y_pos = kwargs["y_position"]

    r_detector_sq = x_pos**2 + y_pos**2

    dist = tt.sqrt(
        r_detector_sq +
        src_x[:, None]**2 +
        src_y[:, None]**2 -
        2 * (x_pos * src_x[:, None] + y_pos * src_y[:, None])
    )
    dist = tt.clip(dist, 1e-3, 1e6)
    rate = (acts[:, None] * SCALE * BRANCH_RATIO * DET_EFF *
            tt.exp(-MU_AIR * dist)) / (4 * tt.pi * dist**2)

    return rate


def build_model(
    x_pos, y_pos,
    lower_x, upper_x,
    lower_y, upper_y,
    mean_bkg, bkg_std,
    cps,
    coordinates,
    prior_activity=1000,
    no_init_val=False
):
    pos_x_tt = tt.as_tensor_variable(x_pos, dtype="float64")
    pos_y_tt = tt.as_tensor_variable(y_pos, dtype="float64")

    coordinates = np.asarray(coordinates)
    x_init = coordinates[:, 0]
    y_init = coordinates[:, 1]

    y_init = [y+1e-3 if np.isclose(0, y) else y for y in y_init]
    sources = {"sources": range(NUM_SOURCE)}

    with pm.Model(coords=sources) as model:
        data = pm.Data("observed_cps", cps)
        if np.isclose(bkg_std, 0.0):
            bkg = mean_bkg
        else:
            sigma_bkg = pm.HalfNormal("sigma_bkg", bkg_std)
            bkg = pm.TruncatedNormal(
                "bkg",
                mu=mean_bkg,
                sigma=sigma_bkg,
                lower=0
            )

        if no_init_val:
            x = pm.Uniform(
                "x_src", lower=lower_x, upper=upper_x, dims="sources"
            )
            y = pm.Uniform(
                "y_src", lower=lower_y, upper=upper_y, dims="sources"
            )
        else:
            y = pm.Uniform(
                "y_src", lower=lower_y, upper=upper_y,
                dims="sources", initval=y_init
            )
            x = pm.Uniform(
                "x_src", lower=lower_x, upper=upper_x,
                dims="sources", initval=x_init
            )

        activity = pm.LogNormal(
            "act_src",
            mu=np.log(prior_activity),
            sigma=2,   # широкий 0.7 or 1.2 more wide
            dims="sources"
        )

        source_cps = mean_cps_tt(
            x_position=pos_x_tt,
            y_position=pos_y_tt,
            x=x,
            y=y,
            acts=activity
        )

        mu_total = source_cps.sum(axis=0) + bkg
        mu_total = tt.clip(mu_total, 1e-6, 1e9)  # protection from zeros
        pm.Poisson(
            "predicted_cps",
            mu=mu_total,
            observed=data
        )

    return model


def parse_filename(fname: str, number_of_sources: int = 1):
    """
    The function extracts real parameters from filename.
    It's sensitive for valid naming. Example of valid naming: 1_2_src_
    45_cps_bkg_1000MBq_1000MBq_100m_100m_1325154_6187254_1325890_6187934.csv
    Returns:
        real_params: dict
        indx: integer
        delta_dist: string
    """
    num_of_params = {1: 10, 2: 14, 3: 18}
    if isinstance(fname, str):
        fname = Path(fname)

    try:
        params_list = fname.stem.split("_")
        assert len(params_list) == num_of_params.get(number_of_sources)
    except AssertionError:
        raise
    else:
        params_list = params_list[1:]
        coords_parse = (
            params_list[-number_of_sources*2:] if number_of_sources > 0 else []
        )
        act_parse = params_list[5:5+number_of_sources]
        dist_parse = [int(x.strip('m')) for x in params_list if "m" in x]

        real_params = {}
        for i in range(number_of_sources):
            x_src = coords_parse[2*i]
            y_src = coords_parse[2*i+1]
            act = int(act_parse[i][:-3])
            distance = dist_parse[i]

            real_params[i+1] = {"x": int(x_src), "y": int(y_src),
                                "act": act, "dist": distance}

        return real_params


def build_coordinates(x_array, y_array, num_sources):
    return list(zip(x_array[:num_sources],
                    y_array[:num_sources]))


def run(
    df,
    params,
    csv_file,
    simnum: int = 2000,
    burnin: int = 500,
    n_chains: int = 2
):
    indexes = None

    real_params = parse_filename(csv_file, number_of_sources=NUM_SOURCE)
    init_params, peaks_found = set_parameters(df, params, idxs=indexes)

    if not peaks_found:
        print("[!] No 2 peaks found. Continuing...")
    
    MEAN_BKG_CPS = init_params["MEAN_BKG_CPS"]
    BKG_STD = init_params["BKG_STD"]
    max_cps = init_params["max_cps"]
    MAX_X = init_params["MAX_X"]
    MAX_Y = init_params["MAX_Y"]
    X_POS = init_params["X_POS"]
    Y_POS = init_params["Y_POS"]

    LOWER_X = init_params["LOWER_X"]
    UPPER_X = init_params["UPPER_X"]
    LOWER_Y = init_params["LOWER_Y"]
    UPPER_Y = init_params["UPPER_Y"]

    cps = init_params["cps"]
    max_idxs = init_params["indexes"]
    ref_coords_x = [real_params[j+1].get("x") for j in range(len(real_params))]
    ref_coords_y = [real_params[j+1].get("y") for j in range(len(real_params))]
    max_coordinates = build_coordinates(MAX_X, MAX_Y, NUM_SOURCE)
    true_acts = [real_params[i+1]["act"] for i in range(len(real_params))]

    model = build_model(
        X_POS, Y_POS,
        LOWER_X, UPPER_X, 
        LOWER_Y, UPPER_Y,
        MEAN_BKG_CPS, BKG_STD,
        cps,
        max_coordinates,
        no_init_val=False
    )

    seed = random.randint(10, 10000)
    with model:
        trace = pm.sample(
            simnum,
            tune=burnin,
            discard_tuned_samples=True,
            chains=n_chains,
            cores=min(n_chains, 2),
            target_accept=0.95,
            random_seed=[seed + i for i in range(n_chains)]
        )
        ppc = pm.sample_posterior_predictive(trace, random_seed=seed)
        trace.extend(ppc)
    return trace, real_params