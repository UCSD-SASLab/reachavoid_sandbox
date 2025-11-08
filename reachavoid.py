import hj_reachability as hj
import jax.numpy as jnp
from dynamics import Dubins
from utils import reachable_tube_plot, reachable_tube_video

# Define obstacle such that x \in F \iff obstacle(x) <= 0
obstacle = lambda x: jnp.linalg.norm(x[:2]) - 2.0

# Define target such that x \in T \iff target(x) >= 0
target = lambda x: 0.5 - jnp.linalg.norm(x[:2] - jnp.array([4.0, 0.0]))


# Define dynamics
dynamics = Dubins(speed=1.0, max_steering_angle=1.0, max_vel_dist=0.2)


# Define grid
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
    hj.sets.Box(jnp.array([-5.0, -5.0, 0.0]), jnp.array([5.0, 5.0, 2 * jnp.pi])),
    (51, 51, 51),
    periodic_dims=2,
)

# Define obstacle and avoid sdf functions
obstacle_sdf = hj.utils.multivmap(obstacle, jnp.arange(grid.ndim))(
    grid.states
).squeeze()

target_sdf = hj.utils.multivmap(target, jnp.arange(grid.ndim))(grid.states).squeeze()


# Define boundary value
boundary_value = jnp.minimum(obstacle_sdf, target_sdf)

# Define value postprocessor (basically DP update step)
brat = lambda t, V: jnp.maximum(jnp.minimum(V, obstacle_sdf), target_sdf)
solver_settings = hj.SolverSettings.with_accuracy("very_high", value_postprocessor=brat)

# Define lookback time
t_lookback = -10.0
times = jnp.linspace(0.0, t_lookback, 101)

# Solve hj reachavoid problem
V_alltimes = hj.solve(
    solver_settings, dynamics, grid, times, boundary_value, progress_bar=True
)

reachable_tube_plot(
    grid,
    V_alltimes[::10],
    obstacle_sdf=obstacle_sdf,
    target_sdf=target_sdf,
    theta_idx=0,
)

reachable_tube_video(
    grid,
    V_alltimes,
    obstacle_sdf=obstacle_sdf,
    target_sdf=target_sdf,
    theta_idx=0,
    ts=times,
)
