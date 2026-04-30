import json

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from typing import List, Tuple, Callable, Dict

def extract_data(simulation_directory):
    df = pd.read_csv(glob(os.path.join(simulation_directory, "*.csv"))[0])
    with open(glob(os.path.join(simulation_directory, "*.json"))[0]) as f:
        params = json.load(f)

    readout_length = df['Dist'][1] - df['Dist'][0]

    src_1_pos = tuple(params['sources'][0]['position'][:2])
    src_2_pos = tuple(params['sources'][1]['position'][:2])

    # TODO: more rigorous direction vector solution
    coord_cols = ['East', 'North']
    start_pos = df.head(1)[coord_cols].to_numpy()[0]
    end_pos = df.tail(1)[coord_cols].to_numpy()[0]

    return src_1_pos, src_2_pos, start_pos, end_pos


def get_polar_params(coords_src1, coords_src2, road_start, road_end):
    road_vec = np.array([
        road_end[0] - road_start[0],
        road_end[1] - road_start[1]
    ])

    src_vec = np.array([
        coords_src2[0] - coords_src1[0],
        coords_src2[1] - coords_src1[1]
    ])

    theta_road = np.arctan2(road_vec[1], road_vec[0])
    theta_src  = np.arctan2(src_vec[1], src_vec[0])

    theta = theta_src - theta_road
    theta = np.arctan2(np.sin(theta), np.cos(theta))

    distance = np.hypot(src_vec[0], src_vec[1])
    return theta, distance

def plot_base_distribution(x: np.ndarray, cps: np.ndarray, max_x: List[int]) -> None:
    """
    The function create a cps base distribution
    params: x - coordinates or length of the detector's trajectory
            cps - count rate
            max_x - x coordinate where the maximum count rate appeared
    return: None
    """
    spines = ["right", "top"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, cps, color="b", linewidth=1)
    for spine in spines:
        ax.spines[spine].set_visible(False)

    if max_x:
        for index in max_x:
            ax.axvline(index, linestyle="--", color="r")
    ax.set_ylabel("CPS", fontsize=14)
    plt.show()


def plot_act_density(samples: dict, true_act=None):
    n_rows = len(samples)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 6), sharex=True, squeeze=False)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.spines.top.set_visible(False)
        ax.spines.right.set_visible(False)
        ax.spines.bottom.set_color("black")
        ax.spines.left.set_color("black")
        ax.set_ylabel("Density", fontsize=14)
        ax.grid(False)
        ax.axvline(true_act[i], color="red", linestyle="--")
        sns.kdeplot(samples[i+1], fill=True, color="b", ax=axes[i])
        
    axes[-1].set_xlabel("Activity, MBq", fontsize=14)      
    plt.show()


def plot_location(
    data,
    src_x_real,
    src_y_real,
    merged_x,
    merged_y
):
    x_coords = data.East
    y_coords = data.North

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_coords, y_coords, color="b", linewidth=1)
    ax.scatter(src_x_real, src_y_real, color="black", marker="^")

    n_sources = merged_x.shape[0]
    colors = ["r", "orange", "green", "purple"]

    for s in range(n_sources):
        ax.scatter(
            merged_x[s][::10],
            merged_y[s][::10],
            color=colors[s % len(colors)],
            marker="x",
            alpha=0.6
        )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.grid(True, alpha=0.5)
    plt.show()
    