import numpy as np
import pandas as pd
import pymc as pm
from scipy.stats import multivariate_normal
from scipy.stats import poisson
from scipy.special import factorial
import matplotlib.pyplot as plt
import time

# Assuming previous parts' code (global parameters, functions) are defined.
# I am including global parameters, calc_lg, and f_calc_infinite_response and the geom_eff calculation
# to make this snippet runnable and self-contained.

sim = 30000
infinite_extent = 10
mu_soil = 0.00813 * 1520
mu_air = 0.00949484
deposition_depth = 0.05
scale_factor = 10
detector_height = 15

area_matrix_bounds = [-100, 100]
measurement_matrix_bounds = [-100, 100]
area_matrix_size = [scale_factor, scale_factor]
measurement_matrix_size = [scale_factor, scale_factor]

area_matrix_x = np.tile(np.linspace(area_matrix_bounds[0], area_matrix_bounds[1], area_matrix_size[1]), (area_matrix_size[0], 1))
area_matrix_y = np.tile(np.linspace(area_matrix_bounds[0], area_matrix_bounds[1], area_matrix_size[0]), (area_matrix_size[1], 1)).T
area_matrix_z = np.zeros((area_matrix_size[0], area_matrix_size[1]))
area_matrix = [area_matrix_x, area_matrix_y, area_matrix_z]

measurement_matrix_x = np.tile(np.linspace(measurement_matrix_bounds[0], measurement_matrix_bounds[1], measurement_matrix_size[1]), (measurement_matrix_size[0], 1))
measurement_matrix_y = np.tile(np.linspace(measurement_matrix_bounds[0], measurement_matrix_bounds[1], measurement_matrix_size[0]), (measurement_matrix_size[1], 1)).T
measurement_matrix_z = np.full((measurement_matrix_size[0], measurement_matrix_size[1]), detector_height)
measurement_matrix = [measurement_matrix_x, measurement_matrix_y, measurement_matrix_z]

area_list = area_matrix
measurement_list = measurement_matrix
list_size = area_matrix_size[0] * area_matrix_size[1]
area_size = area_matrix_size[0] * area_matrix_size[1]
measurement_size = measurement_matrix_size[0] * measurement_matrix_size[1]

geom_eff = np.empty((measurement_size, measurement_size))

# calc_lg function from Part 1
def calc_lg(approx_det_h=5, middle=False, points_to_calculate=500, x_sc=1, y_sc=1):
    lg_sizes = np.concatenate([np.arange(1, 5, 0.5), np.arange(5, 101)])
    tot_fluence_vector = []

    for lg_size in lg_sizes:
        extent_dst = points_to_calculate * lg_size
        dist_points = np.arange(-extent_dst, extent_dst + lg_size, lg_size)
        single_square = (dist_points[1] - dist_points[0])**2

        if not middle:
            mat_x_app = np.tile(dist_points, (len(dist_points), 1))
            mat_y_app = np.tile(dist_points, (len(dist_points), 1)).T
        else:
            mat_x_app = np.tile(dist_points, (len(dist_points), 1)) + (lg_size / 2) * x_sc
            mat_y_app = np.tile(dist_points, (len(dist_points), 1)).T + (lg_size / 2) * y_sc

        approx_det_dist = np.sqrt(mat_x_app**2 + mat_y_app**2 + approx_det_h**2)
        approx_total_fluence = np.sum(((single_square) * np.exp(-0.0103 * approx_det_dist)) / (4 * np.pi * approx_det_dist**2))
        tot_fluence_vector.append(approx_total_fluence)
    return pd.DataFrame({'lg_sizes': lg_sizes, 'tot_fluence_vector': tot_fluence_vector})

fluence_grid = calc_lg(detector_height)
fluence_grid.loc[:, 'tot_fluence_vector'] = fluence_grid['tot_fluence_vector'] / fluence_grid['tot_fluence_vector'][0]
lg_id = abs(round(measurement_matrix_y[0, 1] - measurement_matrix_y[0, 0]))
fluence_id = fluence_grid['lg_sizes'][abs(fluence_grid['lg_sizes'] - lg_id).idxmin()]
fluence_correction = fluence_grid[fluence_grid['lg_sizes'] == fluence_id]['tot_fluence_vector'].values[0]

print("\nEvaluating detector response function for geom_eff...")
for i in range(measurement_size):

    meas_x_flat = measurement_matrix_x.flatten()
    meas_y_flat = measurement_matrix_y.flatten()
    meas_z_flat = measurement_matrix_z.flatten()

    current_meas_x = meas_x_flat[i]
    current_meas_y = meas_y_flat[i]
    current_meas_z = meas_z_flat[i] # This is just detector_height

    dst = np.sqrt(
        (area_list[0] - current_meas_x)**2 +
        (area_list[1] - current_meas_y)**2 +
        (area_list[2] - current_meas_z)**2 # area_list[2] is always 0
    )
    dst_2d = np.sqrt(
        (area_list[0] - current_meas_x)**2 +
        (area_list[1] - current_meas_y)**2
    )
    l = np.sqrt(deposition_depth**2 + ((deposition_depth * dst_2d) / (detector_height + deposition_depth))**2)

    # The result of the element-wise division and multiplication needs to be flattened
    # before assigning to a row of geom_eff
    geom_eff[i, :] = (fluence_correction * 1e3 * (
        (np.exp(-mu_air * (dst - l)) * np.exp(-mu_soil * l)) /
        (4 * np.pi * dst**2)
    )).flatten() # Flatten the result to match the 1D row of geom_eff


def f_calc_infinite_response(extent, area_matrix_size, area_matrix_bounds, height, activity=1, efficiency=1):
    if extent < 1:
        return -np.inf # Use -np.inf for negative infinity

    matrix_step_size = abs((area_matrix_bounds[0] - area_matrix_bounds[1])) / (area_matrix_size[0] - 1)
    print(f"\nNumber of points for infinite plane extent: {extent}")
    print(f"Infinite plane extent to be calculated: {round(extent * matrix_step_size)} m")

    coordinate_vector = np.arange(area_matrix_bounds[0] - extent * matrix_step_size,
                                  area_matrix_bounds[1] + extent * matrix_step_size + matrix_step_size, # +step_size to include end
                                  matrix_step_size)

    meas_coordinate_vector = np.arange(area_matrix_bounds[0],
                                       area_matrix_bounds[1] + matrix_step_size, # +step_size to include end
                                       matrix_step_size)

    activity_x = np.tile(coordinate_vector, (len(coordinate_vector), 1))
    activity_y = np.tile(coordinate_vector, (len(coordinate_vector), 1)).T
    activity_z = np.full((len(coordinate_vector), len(coordinate_vector)), height)
    activity_matrix = np.full((len(coordinate_vector), len(coordinate_vector)), 1) # All ones initially

    # Set activity to 0 within the main area_matrix_bounds
    mask_x = (activity_x >= area_matrix_bounds[0]) & (activity_x <= area_matrix_bounds[1])
    mask_y = (activity_y >= area_matrix_bounds[0]) & (activity_y <= area_matrix_bounds[1])
    activity_matrix[mask_x & mask_y] = 0

    to_return = np.empty((area_matrix_size[0], area_matrix_size[1]))

    for i in range(area_matrix_size[0]): # Iterate over measurement x-coordinates
        for j in range(area_matrix_size[1]): # Iterate over measurement y-coordinates
            # For each measurement point (meas_coordinate_vector[i], meas_coordinate_vector[j]),
            # calculate distances to all points in the extended activity_x, activity_y grid.

            distances = np.sqrt(
                (activity_x - meas_coordinate_vector[i])**2 +
                (activity_y - meas_coordinate_vector[j])**2 +
                (activity_z)**2
            )

            distances_2d = np.sqrt(
                (activity_x - meas_coordinate_vector[i])**2 +
                (activity_y - meas_coordinate_vector[j])**2
            )

            l = np.sqrt(deposition_depth**2 + ((deposition_depth * distances_2d) / (detector_height + deposition_depth))**2)

            term = activity_matrix * (np.exp(-mu_air * (distances - l)) * np.exp(-mu_soil * l)) / (4 * np.pi * distances**2)
            to_return[j, i] = fluence_correction * 1e3 * np.sum(term)

    return to_return

################## MCMC PART - with PyMC ##############################

number_of_parameters = area_matrix_size[0] * area_matrix_size[1]
activity_matrix = np.full((area_matrix_size[0], area_matrix_size[1]), 100)
activity_matrix[35] = 350
activity_matrix = activity_matrix * activity_normalization

#infinite response
infinite_response = f_calc_infinite_response(infinite_extent, area_matrix_size, area_matrix_bounds, height=detector_height)

# Generate data using the known activity matrix
#The function uses infinite response, which needs to be calculated before.
expected_counts = np.dot(geom_eff, activity_matrix.flatten()) + np.mean(activity_matrix) * infinite_response.flatten()
mcmc_data = np.random.poisson(expected_counts).reshape(scale_factor, scale_factor)

# PyMC Model Definition
with pm.Model() as model:
    lambda_param = pm.Gamma("lambda", alpha=1, beta=0.01, shape=number_of_parameters)
    mu = pm.Deterministic("mu", pm.math.dot(geom_eff, lambda_param) + np.mean(lambda_param) * infinite_response.flatten())
    y = pm.Poisson("y", mu=mu, observed=mcmc_data.flatten())
    trace = pm.sample(sim, tune=1000, cores=1)

lambda_samples = trace.posterior["lambda"].values.reshape(-1, number_of_parameters)

# Summary Statistics
print(pm.summary(trace))

# MAP estimate (you can calculate it from the samples)
map_matrix = np.zeros(number_of_parameters)
for i in range(number_of_parameters):
    density = np.histogram(lambda_samples[:, i], bins=50, density=True)
    MAP_value = density[1][np.argmax(density[0])]  # Mode of the histogram
    map_matrix[i] = MAP_value

map_matrix = map_matrix.reshape(scale_factor, scale_factor)

print("\nMAP Matrix:")
print(map_matrix)

# Visualization (Example)
plt.figure(figsize=(10, 6))
plt.plot(lambda_samples[:, 0], label="Lambda[0]")
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.legend()
plt.title("MCMC Trace Plot - Lambda[0]")
plt.show()
