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
    ny = 100 # Higher domain to allow splashing/movement
    dx = 0.01 # 1cm resolution (coarse but fast for demo)
    dy = 0.01
    
    width = nx * dx
    height = ny * dy
    
    print(f"Domain Size: {width:.3f} x {height:.3f} m")
    
    # 2. Initialize Fluid
    grid = StaggeredGrid(nx, ny, dx, dy)
    fluid_solver = FVMSolver(grid)
    
    # Enable Variable Density (VOF)
    fluid_solver.variable_density = True
    fluid_solver.turbulence_model = 'k-epsilon' # Use standard k-epsilon
    
    # Initialize VOF (Water Level at 0.5m)
    water_level = 0.5
    for j in range(ny):
        y = (j + 0.5) * dy
        if y < water_level:
            grid.alpha[:, j] = 1.0
        else:
            grid.alpha[:, j] = 0.0
            
    # 3. Initialize Solid (Particle)
    # Density 500 kg/m3 (Wood-like, half of water)
    polygons = Polygons(material='steel') 
    polygons.mtr.density = 500.0 
    polygon_solver = PolygonSolver()
    
    # Add a cube
    L = 0.1 # 10cm
    cx = width / 2.0
    cy = water_level + L/2.0 + 0.05 # Drop from 5cm above water
    
    h = L / 2.0
    vertices = [
        [-h, -h],
        [h, -h],
        [h, h],
        [-h, h]
    ]
    
    polygons.add(vertices, x=[cx, cy], v=[0.0, 0.0], theta=0.0, omega=0.0, color='brown')
    
    # 4. Simulation Loop
    dt = 1e-3 # Larger dt for stability with VOF? Or smaller? 
    # CFL: u*dt/dx < 1. If u ~ 1m/s, dt < 0.01. 1e-3 is safe.
    t_max = 5.0
    t = 0.0
    step = 0
    
    # Data Logging
    data_log = [] # [t, y, vy]
    
    # Visualization Setup
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 2)
    
    ax_alpha = fig.add_subplot(gs[0, 0])
    ax_pres = fig.add_subplot(gs[0, 1])
    ax_pos = fig.add_subplot(gs[1, :])
    
    # Alpha (Water)
    im_alpha = ax_alpha.imshow(grid.alpha.T, origin='lower', extent=[0, width, 0, height], 
                               cmap='Blues', vmin=0, vmax=1)
    ax_alpha.set_title("Water Fraction (Alpha)")
    patch_alpha = plt.Polygon(polygons.vertices_world[0], closed=True, fc='brown', ec='k')
    ax_alpha.add_patch(patch_alpha)
    
    # Pressure
    im_pres = ax_pres.imshow(grid.p.T, origin='lower', extent=[0, width, 0, height], 
                             cmap='coolwarm')
    ax_pres.set_title("Pressure")
    patch_pres = plt.Polygon(polygons.vertices_world[0], closed=True, fc='brown', ec='k')
    ax_pres.add_patch(patch_pres)
    fig.colorbar(im_pres, ax=ax_pres)
    
    # Position Plot
    line_pos, = ax_pos.plot([], [], 'b-', label='Vertical Position')
    ax_pos.axhline(water_level, color='c', linestyle='--', label='Water Level')
    # Equilibrium: Center at water_level + L/2 - h_sub
    # h_sub = L * 500/1000 = 0.05
    # y_eq = 0.5 + 0.05 - 0.05 = 0.5?
    # Wait. Center y.
    # Bottom of cube at y_b. Submerged depth h_sub = water_level - y_b.
    # y_c = y_b + L/2.
    # So y_c = water_level - h_sub + L/2.
    # h_sub = 0.05. L/2 = 0.05.
    # y_c = 0.5 - 0.05 + 0.05 = 0.5.
    # So center should settle exactly at water level!
    ax_pos.axhline(water_level, color='g', linestyle=':', label='Equilibrium')
    
    ax_pos.set_title("Vertical Position")
    ax_pos.set_xlabel("Time (s)")
    ax_pos.set_ylabel("y (m)")
    ax_pos.set_xlim(0, t_max)
    ax_pos.set_ylim(0, height)
    ax_pos.legend()
    
    title = fig.suptitle(f"t = {t:.3f} s")
    
    print("Starting Simulation...")
    
    def update(frame):
        nonlocal t, step
        
        steps_per_frame = 10
        
        for _ in range(steps_per_frame):
            # A. Fluid Step
            fluid_solver.solve(dt)
            
            # B. Coupling: Solid -> Fluid
            grid.rasterize_polygons(polygons)
            coupling.apply_ibm_forcing(grid)
            
            # C. Coupling: Fluid -> Solid
            polygons.reset_forces()
            polygons.gravity([0.0, -9.81])
            coupling.compute_hydrodynamic_forces(grid, polygons)
            
            # D. Solid Step
            polygons.integrate_pos(dt)
            polygon_solver.solve(polygons, dt)
            polygons.integrate_vel(dt)
            
            data_log.append([t, polygons.x[0, 1], polygons.v[0, 1]])
            
            t += dt
            step += 1
            
        # Update Viz
        im_alpha.set_data(grid.alpha.T)
        patch_alpha.set_xy(polygons.vertices_world[0])
        
        im_pres.set_data(grid.p.T)
        im_pres.set_clim(vmin=np.min(grid.p), vmax=np.max(grid.p))
        patch_pres.set_xy(polygons.vertices_world[0])
        
        data_arr = np.array(data_log)
        if len(data_arr) > 0:
            plot_data = data_arr[::10]
            line_pos.set_data(plot_data[:, 0], plot_data[:, 1])
            ax_pos.set_ylim(0.4, 0.7) # Zoom in around water level
            
        title.set_text(f"t = {t:.3f} s, y = {polygons.x[0,1]:.3f}")
        
        return im_alpha, im_pres, patch_alpha, patch_pres, line_pos, title
        
    ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=False)
    
    ani.save('floating_cube.gif', writer='pillow', fps=20)
    print("Simulation Complete. Saved to floating_cube.gif")
    
    # Save Data
    np.savetxt("floating_data.csv", np.array(data_log), delimiter=",", header="time,y,vy")

if __name__ == "__main__":
    run_simulation()
