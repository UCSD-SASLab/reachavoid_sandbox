import matplotlib.pyplot as plt
import matplotlib.animation as animation

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


def reachable_tube_plot(grid, Vs, obstacle_sdf=None, target_sdf=None, theta_idx=0):
    plt.figure(figsize=(10, 10))

    if obstacle_sdf is not None:
        plt.contourf(
            grid.coordinate_vectors[0],
            grid.coordinate_vectors[1],
            obstacle_sdf[:, :, theta_idx].T,
            levels=[-10, 0],
            colors="red",
        )

    if target_sdf is not None:
        plt.contourf(
            grid.coordinate_vectors[0],
            grid.coordinate_vectors[1],
            target_sdf[:, :, theta_idx].T,
            levels=[0, 10],
            colors="green",
        )
    num_vs = len(Vs)
    if Vs.ndim == grid.ndim:
        Vs = Vs[None]

    for i, V in enumerate(Vs):
        alpha = 0.1 + (1.0 - 0.1) * i / num_vs
        plt.contour(
            grid.coordinate_vectors[0],
            grid.coordinate_vectors[1],
            V[:, :, theta_idx].T,
            levels=[0],
            colors="blue",
            alpha=alpha,
        )

    plt.contourf(
        grid.coordinate_vectors[0],
        grid.coordinate_vectors[1],
        Vs[-1][:, :, theta_idx].T,
        levels=[0, 5],
        colors="blue",
        alpha=0.1,
    )
    plt.title(f"Reachable tubes for theta = {grid.coordinate_vectors[2][theta_idx]}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def reachable_tube_video(grid, Vs, obstacle_sdf=None, target_sdf=None, theta_idx=0, ts=None):
    """
    Create an animated video showing the evolution of the reachable tube over time.
    
    Args:
        grid: Grid object from hj_reachability
        Vs: Value function array over time (shape: (n_times, nx, ny, nz) or (nx, ny, nz))
        obstacle_sdf: Signed distance function for obstacles (optional)
        target_sdf: Signed distance function for targets (optional)
        theta_idx: Index for theta dimension to visualize
    """
    # Ensure Vs has time dimension
    if Vs.ndim == grid.ndim:
        Vs = Vs[None, ...]
    if ts is None:
        ts = range(len(Vs))
        time_name = "step"
        time_unit = ""
    else:
        time_name = "t"
        time_unit = "s"
    num_times = len(Vs)
    
    # Create figure and axis
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    
    def animate(frame):
        # Clear the axis for this frame
        ax.clear()
        
        # Redraw static elements (obstacles and targets)
        if obstacle_sdf is not None:
            ax.contourf(
                grid.coordinate_vectors[0],
                grid.coordinate_vectors[1],
                obstacle_sdf[:, :, theta_idx].T,
                levels=[-10, 0],
                colors="red",
                alpha=0.3,
            )
        
        if target_sdf is not None:
            ax.contourf(
                grid.coordinate_vectors[0],
                grid.coordinate_vectors[1],
                target_sdf[:, :, theta_idx].T,
                levels=[0, 10],
                colors="green",
                alpha=0.3,
            )
        
        # Get current value function
        V_current = Vs[frame]
        
        # Fill the current reachable set
        ax.contourf(
            grid.coordinate_vectors[0],
            grid.coordinate_vectors[1],
            V_current[:, :, theta_idx].T,
            levels=[0, 5],
            colors="blue",
            alpha=0.1,
        )
        
        # Set up axis properties
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        
        # Update title with current time step
        theta_val = grid.coordinate_vectors[2][theta_idx]
        ax.set_title(
            f"Reachable tube evolution ({time_name} = {ts[frame]:.2f} {time_unit}, theta = {theta_val:.2f})",
            fontsize=16,
            fontweight="bold",
        )
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=num_times, interval=4000 // num_times, blit=False, repeat=True
    )
    
    plt.tight_layout()
    plt.show()
