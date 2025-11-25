import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

from dem.src.fvm.grid import StaggeredGrid
from dem.src.fvm.fluid_solver import FVMSolver
import dem.src.fvm.coupling as coupling
from dem.src.core.polygons import Polygons
from dem.src.core.polygon_solver import PolygonSolver

def run_simulation():
    # 1. Setup Domain
    nx = 32
    ny = 64
    dx = 0.002 # 2mm resolution
    dy = 0.002
    
    width = nx * dx
    height = ny * dy
    
    print(f"Domain Size: {width:.3f} x {height:.3f} m")
    
    # 2. Initialize Fluid
    grid = StaggeredGrid(nx, ny, dx, dy)
    fluid_solver = FVMSolver(grid)
    
    # Set fluid properties (Water)
    fluid_solver.rho_water = 1000.0
    fluid_solver.mu_water = 0.001
    
    # Switch to k-epsilon
    fluid_solver.turbulence_model = 'k-epsilon'
    
    # 3. Initialize Solid (Particle)
    polygons = Polygons(material='steel') # Steel density ~7800
    polygon_solver = PolygonSolver()
    
    # Add a square particle
    # Center it horizontally, place it near top
    cx = width / 2.0
    cy = height * 0.8
    size = 0.01 # 1cm cube
    
    # Vertices (Counter-Clockwise)
    h = size / 2.0
    vertices = [
        [-h, -h],
        [h, -h],
        [h, h],
        [-h, h]
    ]
    
    polygons.add(vertices, x=[cx, cy], v=[0.0, 0.0], theta=0.0, omega=0.0, color='r')
    
    # 4. Simulation Loop
    dt = 1e-4
    t_max = 2.0
    t = 0.0
    step = 0
    
    # Data Logging
    data_log = [] # List of [t, y, vy, ay]
    
    # Visualization Setup
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 2)
    
    ax_vel = fig.add_subplot(gs[0, 0])
    ax_pres = fig.add_subplot(gs[0, 1])
    ax_k = fig.add_subplot(gs[1, 0])
    ax_eps = fig.add_subplot(gs[1, 1])
    ax_pos = fig.add_subplot(gs[2, 0])
    ax_v_plot = fig.add_subplot(gs[2, 1])
    
    # Fluid Velocity Magnitude
    im_vel = ax_vel.imshow(np.zeros((ny, nx)), origin='lower', extent=[0, width, 0, height], 
                   cmap='viridis', vmin=0, vmax=0.1)
    ax_vel.set_title("Velocity Magnitude")
    patch_vel = plt.Polygon(polygons.vertices_world[0], closed=True, fc='r', ec='k')
    ax_vel.add_patch(patch_vel)
    fig.colorbar(im_vel, ax=ax_vel, fraction=0.046, pad=0.04)

    # Pressure
    # Hydrostatic P ~ rho*g*h = 1000*9.8*0.128 ~ 1250 Pa
    im_pres = ax_pres.imshow(np.zeros((ny, nx)), origin='lower', extent=[0, width, 0, height], 
                   cmap='coolwarm', vmin=-100, vmax=1500)
    ax_pres.set_title("Pressure")
    patch_pres = plt.Polygon(polygons.vertices_world[0], closed=True, fc='r', ec='k')
    ax_pres.add_patch(patch_pres)
    fig.colorbar(im_pres, ax=ax_pres, fraction=0.046, pad=0.04)

    # TKE
    im_k = ax_k.imshow(np.zeros((ny, nx)), origin='lower', extent=[0, width, 0, height], 
                   cmap='plasma', vmin=0, vmax=1e-4)
    ax_k.set_title("Turbulent Kinetic Energy (k)")
    patch_k = plt.Polygon(polygons.vertices_world[0], closed=True, fc='r', ec='k')
    ax_k.add_patch(patch_k)
    fig.colorbar(im_k, ax=ax_k, fraction=0.046, pad=0.04)

    # Epsilon/Omega
    # Omega ~ 100 initially, Epsilon ~ small
    if fluid_solver.turbulence_model == 'k-epsilon':
        eps_vmax = 1e-3 # Guessing range for epsilon
    else:
        eps_vmax = 200.0 # For Omega
        
    im_eps = ax_eps.imshow(np.zeros((ny, nx)), origin='lower', extent=[0, width, 0, height], 
                   cmap='inferno', vmin=0, vmax=eps_vmax)
    ax_eps.set_title("Dissipation Rate (epsilon)") # Default title
    patch_eps = plt.Polygon(polygons.vertices_world[0], closed=True, fc='r', ec='k')
    ax_eps.add_patch(patch_eps)
    fig.colorbar(im_eps, ax=ax_eps, fraction=0.046, pad=0.04)
    
    # Plots
    line_pos, = ax_pos.plot([], [], 'b-')
    ax_pos.set_title("Vertical Position (y)")
    ax_pos.set_xlabel("Time (s)")
    ax_pos.set_ylabel("y (m)")
    ax_pos.set_xlim(0, t_max)
    # ax_pos.set_ylim(0, height) # Let autoscale handle it or set initial
    
    line_v, = ax_v_plot.plot([], [], 'r-')
    ax_v_plot.set_title("Vertical Velocity (vy)")
    ax_v_plot.set_xlabel("Time (s)")
    ax_v_plot.set_ylabel("vy (m/s)")
    ax_v_plot.set_xlim(0, t_max)
    # ax_v_plot.set_ylim(-0.2, 0.1) # Let autoscale handle it
    
    title = fig.suptitle(f"t = {t:.3f} s")
    
    print("Starting Simulation...")
    
    def update(frame):
        nonlocal t, step
        
        # Run multiple physics steps per frame
        steps_per_frame = 20
        
        for _ in range(steps_per_frame):
            # A. Fluid Step
            fluid_solver.solve(dt)
            
            # B. Coupling: Solid -> Fluid (IBM Forcing)
            grid.rasterize_polygons(polygons)
            coupling.apply_ibm_forcing(grid)
            
            # C. Coupling: Fluid -> Solid (Hydrodynamic Forces)
            polygons.reset_forces()
            polygons.gravity([0.0, -9.81])
            coupling.compute_hydrodynamic_forces(grid, polygons)
            
            # D. Solid Step
            polygons.integrate_pos(dt)
            polygon_solver.solve(polygons, dt)
            polygons.integrate_vel(dt)
            
            # Log Data
            # t, y, vy, ay
            # ay is stored in polygons.a
            data_log.append([t, polygons.x[0, 1], polygons.v[0, 1], polygons.a[0, 1]])
            
            t += dt
            step += 1
            
        # Update Visualization
        # 1. Velocity
        u_c = 0.5 * (grid.u[:-1, :] + grid.u[1:, :])
        v_c = 0.5 * (grid.v[:, :-1] + grid.v[:, 1:])
        vel_mag = np.sqrt(u_c**2 + v_c**2)
        im_vel.set_data(vel_mag.T)
        patch_vel.set_xy(polygons.vertices_world[0])
        
        # 2. Pressure
        im_pres.set_data(grid.p.T)
        # im_pres.set_clim(vmin=np.min(grid.p), vmax=np.max(grid.p)) # Fixed range now
        patch_pres.set_xy(polygons.vertices_world[0])

        # 3. TKE
        if fluid_solver.k is not None:
            im_k.set_data(fluid_solver.k.T)
            patch_k.set_xy(polygons.vertices_world[0])

        # 4. Epsilon/Omega
        if fluid_solver.omega_t is not None:
            im_eps.set_data(fluid_solver.omega_t.T)
            patch_eps.set_xy(polygons.vertices_world[0])
            
            if fluid_solver.turbulence_model == 'k-epsilon':
                 ax_eps.set_title("Dissipation Rate (epsilon)")
            else:
                 ax_eps.set_title("Specific Dissipation (omega)")
            
        # 5. Plots
        data_arr = np.array(data_log)
        if len(data_arr) > 0:
            # Downsample for plotting speed if needed
            plot_data = data_arr[::10] 
            line_pos.set_data(plot_data[:, 0], plot_data[:, 1])
            line_v.set_data(plot_data[:, 0], plot_data[:, 2])
            
            # Rescale axes
            ax_pos.relim()
            ax_pos.autoscale_view()
            
            ax_v_plot.relim()
            ax_v_plot.autoscale_view()
        
        title.set_text(f"t = {t:.3f} s, y = {polygons.x[0,1]:.3f}")
        
        return im_vel, im_pres, im_k, im_eps, patch_vel, patch_pres, patch_k, patch_eps, line_pos, line_v, title
        
    ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=False)
    
    ani.save('sedimentation_k_epsilon.gif', writer='pillow', fps=20)
    print("Simulation Complete. Saved to sedimentation_k_epsilon.gif")
    
    # Save Data to CSV
    header = "time,y,vy,ay"
    np.savetxt("particle_data.csv", np.array(data_log), delimiter=",", header=header, comments='')
    print("Data saved to particle_data.csv")

if __name__ == "__main__":
    run_simulation()
