import json

import numpy as np
import pandas as pd


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