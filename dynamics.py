import jax.numpy as jnp
import hj_reachability as hj

class Dubins(hj.dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(self, speed: float, max_steering_angle: float, max_vel_dist: float):
        self.speed = speed
        self.max_steering_angle = max_steering_angle
        self.max_vel_dist = max_vel_dist
        self.control_space = hj.sets.Box(jnp.array([-1.0]), jnp.array([1.0]))
        self.disturbance_space = hj.sets.Box(jnp.array([-1.0, -1.0]), jnp.array([1.0, 1.0]))
        super().__init__(control_mode="max", disturbance_mode="min", control_space=self.control_space, disturbance_space=self.disturbance_space)

    def open_loop_dynamics(self, state: jnp.ndarray, time: float) -> jnp.ndarray:
        return jnp.array([self.speed * jnp.cos(state[2]), self.speed * jnp.sin(state[2]), 0.])

    def control_jacobian(self, state: jnp.ndarray, time: float) -> jnp.ndarray:
        return jnp.array([[0], [0], [self.max_steering_angle]])

    def disturbance_jacobian(self, state: jnp.ndarray, time: float) -> jnp.ndarray:
        return jnp.array([[self.max_vel_dist, 0], [0, self.max_vel_dist], [0, 0]])
