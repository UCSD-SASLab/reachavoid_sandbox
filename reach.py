import hj_reachability as hj
import jax.numpy as jnp
from dynamics import Dubins

import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.figsize": (10, 10),
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
    }
)

# Define target such that x \in T \iff target(x) >= 0
target = lambda x: 0.5 - jnp.linalg.norm(x[:2] - jnp.array([4., 0.])) 


# Define dynamics
dynamics = Dubins(speed=1.0, max_steering_angle=1.0, max_vel_dist=0.2)


# Define grid
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
    hj.sets.Box(jnp.array([-5.0, -5.0, 0.0]), jnp.array([5.0, 5.0, 2 * jnp.pi])),
    (51, 51, 51),
    periodic_dims=2,
)

target_sdf = hj.utils.multivmap(target, jnp.arange(grid.ndim))(
    grid.states
).squeeze()


# Define boundary value
boundary_value = target_sdf

# Define value postprocessor (basically DP update step)
brat = lambda t, V: jnp.maximum(V, target_sdf)
solver_settings = hj.SolverSettings.with_accuracy(
    "very_high", value_postprocessor=brat)

# Define lookback time
t_lookback = -10.0
times = jnp.linspace(0.0, t_lookback, 10)

# Solve hj reachavoid problem
V_alltimes = hj.solve(solver_settings, dynamics, grid, times, boundary_value, progress_bar=True)

theta_idx = 0

plt.figure(figsize=(10, 10))

plt.contourf(grid.coordinate_vectors[0], grid.coordinate_vectors[1], target_sdf[:, :, theta_idx].T, levels=[0, 10], colors="green")

for i, V in enumerate(V_alltimes):
    alpha = 0.1 + 0.1 * i
    plt.contour(grid.coordinate_vectors[0], grid.coordinate_vectors[1], V[:, :, theta_idx].T, levels=[0], colors="blue", alpha=alpha)
plt.contourf(grid.coordinate_vectors[0], grid.coordinate_vectors[1], V_alltimes[-1][:, :, theta_idx].T, levels=[0, 5], colors="blue", alpha=0.1)

plt.title(f"Reachable tubes for theta = {grid.coordinate_vectors[2][theta_idx]}")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
