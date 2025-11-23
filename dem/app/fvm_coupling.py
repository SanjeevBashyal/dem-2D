# Generic imports
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolyPatch

# Custom imports
from dem.app.base_app          import *
from dem.src.core.polygons     import *
from dem.src.core.polygon_solver import *
from dem.src.fvm.grid          import *
from dem.src.fvm.fluid_solver  import *
from dem.src.fvm.coupling      import *

### ************************************************
### FVM Coupling App
class fvm_coupling(base_app):
    ### ************************************************
    ### Constructor
    def __init__(self):
        super().__init__()

        # Parameters
        self.dt        = 1.0e-4
        self.t_max     = 2.0
        self.plot_freq = 50
        self.nt        = int(self.t_max/self.dt)
        self.it        = 0
        self.t         = 0.0
        self.plot_it   = 0
        self.plot_show = True
        self.plot_png  = False
        self.plot_trajectory = False
        self.d_lst     = [] 
        
        # Domain
        self.L = 1.0
        self.H = 1.0
        self.nx = 50
        self.ny = 50
        self.dx = self.L / self.nx
        self.dy = self.H / self.ny
        
        # FVM Grid & Solver
        self.grid = StaggeredGrid(self.nx, self.ny, self.dx, self.dy)
        self.fluid_solver = FVMSolver(self.grid)
        
        # Polygons
        self.polygons = Polygons(material="steel")
        self.dem_solver = PolygonSolver()
        
        # Initial Conditions
        # Fill bottom half with water
        for j in range(self.ny // 2):
            self.grid.alpha[:, j] = 1.0
            
        # Add a falling block
        verts = [[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05]]
        self.polygons.add(verts, [0.5, 0.7], [0,0], 0.0, 0.0, color='brown')
        
        # Add walls (fixed polygons)
        t = 0.1
        # Bottom
        self.add_wall([-0.1, -t], [self.L+0.1, -t], [self.L+0.1, 0], [-0.1, 0])
        # Left
        self.add_wall([-t, 0], [0, 0], [0, self.H], [-t, self.H])
        # Right
        self.add_wall([self.L, 0], [self.L+t, 0], [self.L+t, self.H], [self.L, self.H])

    def add_wall(self, p1, p2, p3, p4):
        verts = [p1, p2, p3, p4]
        self.polygons.add(verts, [0,0], [0,0], 0.0, 0.0, color='k', fixed=True)

    ### ************************************************
    ### Compute forces & Update
    def forces(self):
        pass

    def update(self):
        # 1. Rasterize Polygons (Update eps_s and u_solid)
        self.grid.rasterize_polygons(self.polygons)
        
        # 2. Fluid Predictor (Advection + Gravity)
        # Note: fluid_solver.solve does full step. 
        # We need to inject IBM forcing in between.
        # Let's break down fluid_solver.solve or modify it.
        # For now, let's assume we can call steps individually or use a modified solve.
        
        # Step A: Advect
        advect_velocity(self.grid.u, self.grid.v, self.grid.u, self.grid.v, 
                        self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy, self.dt)
        
        # Step B: Gravity
        self.grid.v[:, 1:-1] -= 9.81 * self.dt
        
        # 3. Coupling (IBM Forcing)
        # Force fluid velocity to match solid velocity
        apply_ibm_forcing(self.grid)
        
        # 4. Pressure Solve (Projection)
        compute_divergence(self.grid.u, self.grid.v, self.grid.div, 
                           self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy)
        
        rho = 1000.0 # Simplified
        rhs = self.grid.div * rho / self.dt
        
        solve_pressure_sor(self.grid.p, rhs, self.grid.nx, self.grid.ny, 
                           self.grid.dx, self.grid.dy, 50, 1e-4, 1.7)
        
        # 5. Correct Velocity
        correct_velocity(self.grid.u, self.grid.v, self.grid.p, 
                         self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy, self.dt, rho)
        
        # 6. VOF Advection
        advect_scalar(self.grid.alpha, self.grid.u, self.grid.v, 
                      self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy, self.dt)
        
        # 7. DEM Physics
        self.polygons.reset_forces()
        
        # Gravity
        self.polygons.gravity([0, -9.81])
        
        # Hydrodynamic Forces
        compute_hydrodynamic_forces(self.grid, self.polygons)
        
        # Integration (Verlet)
        self.polygons.integrate_pos(self.dt)
        self.dem_solver.solve(self.polygons, self.dt)
        self.polygons.integrate_vel(self.dt)
        
        # Time step
        self.it += 1
        self.t  += self.dt

    ### ************************************************
    ### Plot
    def plot(self):
        if (self.it % self.plot_freq != 0):
            return

        plt.clf()
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_xlim(-0.1, self.L + 0.1)
        ax.set_ylim(-0.1, self.H + 0.1)
        
        # Plot Fluid (VOF)
        # Contour plot of alpha
        X, Y = np.meshgrid(np.arange(self.nx)*self.dx + 0.5*self.dx, 
                           np.arange(self.ny)*self.dy + 0.5*self.dy, indexing='ij')
        
        # plt.contourf(X, Y, self.grid.alpha, levels=[0.5, 1.0], colors=['c'], alpha=0.5)
        # Or imshow
        plt.imshow(self.grid.alpha.T, origin='lower', extent=[0, self.L, 0, self.H], 
                   cmap='Blues', alpha=0.5, vmin=0, vmax=1)
        
        # Plot Velocity Quiver (subsampled)
        # plt.quiver(X[::2,::2], Y[::2,::2], self.grid.u[::2,::2].T, self.grid.v[::2,::2].T, scale=10)
        
        # Plot Polygons
        for i in range(self.polygons.np):
            verts = self.polygons.vertices_world[i]
            c = self.polygons.color[i]
            poly = PolyPatch(verts, closed=True, facecolor=c, edgecolor='k', alpha=0.8)
            ax.add_patch(poly)
            
        plt.pause(0.001)
