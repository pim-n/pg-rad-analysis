import warnings
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy.typing as npt
from dataclasses import dataclass
from scipy.signal import find_peaks, peak_widths
from typing import List, Tuple, Dict
from datetime import datetime
from pathlib import Path
from PIL import Image
from io import BytesIO


warnings.filterwarnings("ignore")


def set_parameters_2_src(d, idxs=None):
    cps = d.ROI_P.values
    bkg_array = d.ROI_BR.values
    
    if idxs is not None:
        max_idxs = idxs
    else:
        cps_smooth = pd.Series(cps).rolling(5, center=True, min_periods=1).mean().values
        peaks, props = find_peaks(
            cps_smooth,
            prominence=np.std(cps_smooth),
            distance=len(cps_smooth)//10
        )
        if len(peaks) >= 2:
            strongest = np.argsort(cps_smooth[peaks])[-2:]
            max_idxs = peaks[strongest]
        else:
            max_idxs = np.argsort(cps_smooth)[-2:]
        max_idxs = sorted(max_idxs)
    
    params_2d = {
        "MEAN_BKG_CPS": bkg_array.mean(),
        "BKG_STD": bkg_array.std(),
        "X_POS": d["East"].values,
        "Y_POS": d["North"].values,
        "MAX_X": [],
        "MAX_Y": [],
        "max_cps": [],
        "cps": cps,
        "LOWER_X": d["East"].min(),
        "UPPER_X": d["East"].max(),
        "LOWER_Y": d["North"].min(),
        "UPPER_Y": d["North"].max(),
        "indices": max_idxs
    }
    for _, idx in enumerate(max_idxs):
        params_2d["MAX_X"].append(d.East[idx])
        params_2d["MAX_Y"].append(d.North[idx])
        params_2d["max_cps"].append(d.ROI_P[idx])
        
    return params_2d


def plot_base_distribution_5_by_2(x, pois_data, filenames, max_xs=None):
    fig, axes = plt.subplots(5, 2, figsize=(6, 4))
    axes = axes.flatten()

    for i, ax in enumerate(axes):    
        ax.plot(x, pois_data[i], color="b", linewidth=1)        
        if max_xs:
            ax.axvline(x[max_xs[i][0]], linestyle="--", color="r")
            ax.axvline(x[max_xs[i][1]], linestyle="--", color="r")
            
        ax.set_ylabel("CPS", fontsize=8)
        ax.set_title(filenames[i], fontsize=8)
    
    fig.tight_layout()     
    return fig


def plot_reconstructed_cps_5_by_2(parameters, pred_cps):
    fig, axes = plt.subplots(5, 2, figsize=(8, 6))
    axes = axes.flatten()
    x = np.arange(len(parameters[0].get("cps")))

    for i, ax in enumerate(axes):
        ax.scatter(x, parameters[i].get("cps"), c="w", ec="black", s=10)
        n_samples = pred_cps[i].shape[0]
        for j in range(0, n_samples, 100):
            ax.plot(pred_cps[i][j], c="r", alpha=.2, linewidth=.5)
        ax.set_ylabel("CPS")

    fig.tight_layout()
    plt.close()
    return fig


def plot_location_5_by_2(parameters, xy_src, post_xy_coords):

    fig, axes = plt.subplots(5, 2, figsize=(12, 16))
    axes = axes.flatten()

    for i, ax in enumerate(axes):

        x_coords = parameters[i].get("X_POS")
        y_coords = parameters[i].get("Y_POS")

        xy_src1, xy_src2 = xy_src[i]
        post_xy_coords1, post_xy_coords2 = post_xy_coords[i]

        x1, y1 = xy_src1
        x2, y2 = xy_src2

        post_x_coords1, post_y_coords1 = post_xy_coords1
        post_x_coords2, post_y_coords2 = post_xy_coords2

        ax.plot(x_coords, y_coords, color="b", linewidth=0.5, zorder=2)

        ax.scatter(x1, y1, color="black", marker="^", s=10, zorder=3)
        ax.scatter(x2, y2, color="black", marker="^", s=10, zorder=3)

        ax.scatter(post_x_coords1, post_y_coords1, color="r", marker="x", s=10, zorder=1)
        ax.scatter(post_x_coords2, post_y_coords2, color="r", marker="x", s=10, zorder=1)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=0.5)

    fig.tight_layout()
    plt.close()

    return fig


def save_results(data: pd.DataFrame | List[pd.DataFrame], path_to_save: str, files=None) -> None:
    if isinstance(data, list):
        for i, file in enumerate(files):
            no_format_filename = str(file).strip('.csv')
            SAMPLES_OUTPUT_FILE = f"{no_format_filename}_{datetime.now().date()}"
            SAMPLE_RESULTS_PATH = Path(path_to_save.joinpath(SAMPLES_OUTPUT_FILE + ".csv"))
            data[i].to_csv(SAMPLE_RESULTS_PATH)
    else:
        EVAL_RESULTS_PATH = Path(path_to_save.joinpath(f"evaluation_summary_{datetime.now().date()}" + ".csv"))
        data.to_csv(EVAL_RESULTS_PATH, index=False)


def parse_filename(fname: str, number_of_sources: int = 1):
    """
    The function extracts real parameters from filename. It's sensitive for valid naming
    Example of valid naming: 1_2_src_45_cps_bkg_1000MBq_1000MBq_100m_100m_1325154_6187254_1325890_6187934.csv
    
    Returns:
        real_params: dict
        indx: integer
        delta_dist: string
    """
    try:
        frags = fname.split("/")
        params_list = frags[-1].rstrip(".csv").split("_")
        delta_dist = frags[-3].split("_")[-1]
        assert len(params_list) == 14        
    except AssertionError:
        raise
    else:
        indx = int(params_list[0])
        params_list = params_list[1:]
        coords_parse = params_list[-number_of_sources*2:] if number_of_sources > 0 else []
        act_parse = params_list[5:5+number_of_sources]
        dist_parse = list(map(lambda x: int(x.strip('m')), params_list[-6:-4]))

        real_params = {indx: {}}
        for i in range(number_of_sources):
            x_src = coords_parse[2*i]
            y_src = coords_parse[2*i+1]
            act = int(act_parse[i][:-3])
            distance = dist_parse[i]
    
            real_params[indx][i+1] = {"x": int(x_src), "y": int(y_src), "act": act, "dist": distance}
    
        return real_params, indx, delta_dist

    
def signal_to_noise(gross_cps, is_smoothed=False):
    upper_bkg = gross_cps.quantile(0.975)
    lower_bkg = gross_cps.quantile(0.025)
    mean_cps_ = gross_cps.mean()
    amplitude_max = gross_cps[(gross_cps >= mean_cps_) & (gross_cps <= gross_cps.max())].max()
    bkg_avg = gross_cps[(gross_cps >= lower_bkg) & (gross_cps <= mean_cps_)].mean()

    if is_smoothed:
        bkg_avg_std = gross_cps[(gross_cps >= lower_bkg) & (gross_cps <= mean_cps_)].std()
    else:    
        bkg_avg_std = np.sqrt(bkg_avg)

    return amplitude_max, bkg_avg, (amplitude_max - bkg_avg)/bkg_avg_std


def get_fwhm(primary_photons, dist):
    distance = 40
    while distance:
        peaks, _ = find_peaks(primary_photons, distance=distance, height=primary_photons.max()/2)
        if len(peaks) != 2:
            print("[*] Peaks not found. Retrying...")
            distance -= 10
        else:
            break
          
    if len(peaks) != 2:
        print("[*] Manual peak search required")
        return None, None
    else:
        try:
            fwhm = peak_widths(primary_photons, peaks, rel_height=0.5)
            left_eval = np.interp(fwhm[2], np.arange(len(dist)), dist)
            right_eval = np.interp(fwhm[3], np.arange(len(dist)), dist)
            width_heights = fwhm[1]
            delta = dist[peaks[1]] - dist[peaks[0]]
            peak_sep = delta // (right_eval[0] - left_eval[0])
        except Exeption as err:
            print(f"[!] Error during peaks processing: {err}. Skipping...")
            return None, None
        else:
            return peak_sep, fwhm[1][0]

        
def combine_dataframes(files):
    print("[*] Building dataframes from files...")
    try:
        dataframes = [pd.read_csv(file) for file in files]
        col_to_rename = {"Dist(m)": "dist", "Dist": "dist"}
        print("[*] Renaming columns...")
        for df in dataframes:            
            df.rename(columns=col_to_rename, inplace=True)

        assert dataframes[random.randint(0, 9)].iloc[:, 4].name == "dist"
    except AssertionError:
        print(f"[!] Columns renaming failed. Skipping")
    else:
        print(f"[*] Columns have been renamed succesfully")
        
    return dataframes


def calculate_locliztion_prob(data: pd.DataFrame) -> pd.DataFrame:
    try:
        if data.columns.duplicated().any():
            print("[!] Duplicate columns detected, fixing...")
            data = data.loc[:, ~data.columns.duplicated()]
            
        columns = data.columns.to_list()
        total_lp1 = data["LP1"].sum()
        total_lp2 = data["LP2"].sum()
        
        total_lp_row = {col: np.nan for col in columns} 
        total_lp_row["LP1"] = total_lp1
        total_lp_row["LP2"] = total_lp2
        total_lp_row["Act1"] = "Total LP" 
        
        new_row_df = pd.DataFrame([total_lp_row])
        new_row_df = new_row_df.reindex(columns=data.columns)
        data = pd.concat([data, pd.DataFrame([total_lp_row])], ignore_index=True)  
        
        last_row = data.iloc[-1]
        total_lp1_value = last_row['LP1']
        total_lp2_value = last_row['LP2']
    except Exception:
        print("[!] Unexpected error during Localization probability calculations")
        raise 
    else:
        return data
    

@dataclass
class PositionResolver:
    road_start: Tuple[float, float] = (0, 0)
    road_end: Tuple[float, float] = (1169, 1169)

    def _line_params(self):
        x1, y1 = self.road_start
        x2, y2 = self.road_end

        A = y2 - y1
        B = x1 - x2
        C = x2*y1 - x1*y2
        return A, B, C

    def _distance_samples(self, x_samples, y_samples):
        A, B, C = self._line_params()
        numerator = np.abs(A*x_samples + B*y_samples + C)
        denominator = np.sqrt(A*A + B*B)
        return numerator / denominator

    def distance_stats(self, x_samples, y_samples) -> Dict[str, float]:
        d_samples = self._distance_samples(x_samples, y_samples)

        return {
            "mean": np.mean(d_samples),
            "std": np.std(d_samples),
            "median": np.median(d_samples),
            "hdi_3%": np.quantile(d_samples, 0.03),
            "hdi_97%": np.quantile(d_samples, 0.97),
        }

    def resolve_distance(self, x_samples, y_samples):
        d_samples = self._distance_samples(x_samples, y_samples)
        return np.median(d_samples)

    def resolve_all_sources(self, x_samples, y_samples):
        """
        x_samples: shape (n_sources, n_draws)
        y_samples: shape (n_sources, n_draws)
        """
        n_sources = x_samples.shape[0]
        results = []

        for i in range(n_sources):
            stats = self.distance_stats(
                x_samples[i],
                y_samples[i]
            )
            results.append(stats)
        return results


resolver = PositionResolver()
