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
### FVM Open Channel App
class fvm_open_channel(base_app):
    ### ************************************************
    ### Constructor
    def __init__(self):
        super().__init__()

        # Parameters
        self.dt        = 5.0e-5 # Small dt for stability
        self.t_max     = 2.0
        self.plot_freq = 100
        self.nt        = int(self.t_max/self.dt)
        self.it        = 0
        self.t         = 0.0
        self.plot_it   = 0
        self.plot_show = True
        self.plot_png  = False
        self.plot_trajectory = False
        self.d_lst     = [] 
        
        # Domain
        self.L = 2.0
        self.H = 0.5 # Back to smaller height
        self.nx = 100
        self.ny = 25 # Back to standard resolution
        self.dx = self.L / self.nx
        self.dy = self.H / self.ny
        
        # Channel Parameters
        self.slope_deg = 5.0
        self.g_mag     = 9.81
        # Gravity tilted (Grid aligned with slope)
        self.gx        = self.g_mag * math.sin(math.radians(self.slope_deg))
        self.gy        = -self.g_mag * math.cos(math.radians(self.slope_deg))
        
        # Bed Geometry (Flat in grid)
        self.bed_h = 0.0
        
        # FVM Grid & Solver
        self.grid = StaggeredGrid(self.nx, self.ny, self.dx, self.dy)
        self.fluid_solver = FVMSolver(self.grid)
        # Override gravity in solver
        self.fluid_solver.g = 0.0 # We handle gravity manually in update
        
        # Polygons
        self.polygons = Polygons(material="sand")
        self.dem_solver = PolygonSolver()
        self.dem_solver.k1 = 1.0e7
        self.dem_solver.k2 = 2.0e7
        
        # Initial Conditions
        # Fill domain with water up to height h_water above bed
        h_water = 0.2
        j_water = int(h_water / self.dy)
        
        self.grid.alpha[:, :j_water] = 1.0
        self.grid.u[:, :j_water] = 3.0
        
        # Initialize turbulence
        if self.fluid_solver.k is not None:
            k_init = 1.5 * (0.05 * 3.0)**2
            L_scale = 0.07 * 0.4
            omega_init = k_init**0.5 / (0.09**0.25 * L_scale)
            
            self.fluid_solver.k[:, :j_water] = k_init
            self.fluid_solver.omega_t[:, :j_water] = omega_init
        
        # Add Bed (Fixed Polygons)
        # Flat bed at bottom
        t = 0.1
        self.add_wall([-0.1, -t], [self.L+0.1, -t], [self.L+0.1, 0], [-0.1, 0])
        
        # Create Bed of Particles (Mobile)
        # Place particles along the bottom with no overlap
        self.create_particle_bed()
        
        # Particle Emitter State (Disabled)
        self.rocks_emitted = 0
        self.max_rocks = 0

    def add_wall(self, p1, p2, p3, p4):
        verts = [p1, p2, p3, p4]
        self.polygons.add(verts, [0,0], [0,0], 0.0, 0.0, color='k', fixed=True)

    def create_random_poly(self, radius):
        n_sides = np.random.randint(4, 7)
        verts = []
        for k in range(n_sides):
            angle = 2 * math.pi * k / n_sides
            vx = radius * math.cos(angle)
            vy = radius * math.sin(angle)
            verts.append([vx, vy])
        return verts

    ### ************************************************
    ### Compute forces & Update
    def forces(self):
        pass

    def update(self):
        # 1. Emitter (Disabled)
        # if self.t > 0.5 and self.it % 200 == 0 and self.rocks_emitted < self.max_rocks:
        #     self.emit_rocks()
            
        # 2. FVM Boundary Conditions (Inlet)
        self.apply_inlet_bc()
        
        # 3. Rasterize Polygons
        self.grid.rasterize_polygons(self.polygons)
        
        # 4. Fluid Predictor (Advection)
        advect_velocity(self.grid.u, self.grid.v, self.grid.u, self.grid.v, 
                        self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy, self.dt)
        
        # 5. Gravity (Rotated)
        # u (vertical faces) gets gx
        self.grid.u[1:-1, :] += self.gx * self.dt
        # v (horizontal faces) gets gy
        self.grid.v[:, 1:-1] += self.gy * self.dt
        
        # 6. Coupling (IBM Forcing)
        apply_ibm_forcing(self.grid)
        
        # 7. Pressure Solve
        compute_divergence(self.grid.u, self.grid.v, self.grid.div, 
                           self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy)
        
        rho = 1000.0
        rhs = self.grid.div * rho / self.dt
        
        solve_pressure_sor(self.grid.p, rhs, self.grid.nx, self.grid.ny, 
                           self.grid.dx, self.grid.dy, 20, 1e-3, 1.7)
        
        # 8. Correct Velocity
        correct_velocity(self.grid.u, self.grid.v, self.grid.p, 
                         self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy, self.dt, rho)
        
        # 9. VOF Advection
        advect_scalar(self.grid.alpha, self.grid.u, self.grid.v, 
                      self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy, self.dt)
        
        # 10. DEM Physics
        self.polygons.reset_forces()
        self.polygons.gravity([self.gx, self.gy])
        compute_hydrodynamic_forces(self.grid, self.polygons)
        
        self.polygons.integrate_pos(self.dt)
        self.dem_solver.solve(self.polygons, self.dt)
        self.polygons.integrate_vel(self.dt)
        
        # 11. Outlet BC (Simple outflow)
        # Already handled by advection pushing things out?
        # We need to ensure zero gradient at outlet for stability
        self.apply_outlet_bc()
        
        # Time step
        self.it += 1
        self.t  += self.dt

    def apply_inlet_bc(self):
        # Fixed velocity at x=0
        h_inlet = 0.2
        j_max = int(h_inlet / self.dy)
        
        # u = 3.0 m/s (High velocity)
        self.grid.u[0, :j_max] = 3.0
        self.grid.u[0, j_max:] = 0.0
        
        # v = 0
        self.grid.v[0, :] = 0.0
        
        # alpha = 1
        self.grid.alpha[0, :j_max] = 1.0
        self.grid.alpha[0, j_max:] = 0.0
        
        # Turbulence BCs
        k_inlet = 1.5 * (0.05 * 3.0)**2
        L_scale = 0.07 * 0.4
        omega_inlet = k_inlet**0.5 / (0.09**0.25 * L_scale)
        
        if self.fluid_solver.k is not None:
            self.fluid_solver.k[0, :j_max] = k_inlet
            self.fluid_solver.omega_t[0, :j_max] = omega_inlet

    def apply_outlet_bc(self):
        # Zero gradient at x=L (i=nx)
        # u[nx, :] = u[nx-1, :]
        self.grid.u[self.nx, :] = self.grid.u[self.nx-1, :]
        self.grid.v[self.nx-1, :] = self.grid.v[self.nx-2, :] # v is nx-1 wide? v is (nx, ny+1)
        # v index i goes 0 to nx-1.
        
        # alpha
        self.grid.alpha[self.nx-1, :] = self.grid.alpha[self.nx-2, :]

    def create_particle_bed(self):
        # Create 3 particles with proper spacing
        
        radius = 0.02
        spacing = 0.3 
        
        x_start = 0.7
        
        for i in range(3):
            x = x_start + i * spacing
            y = radius + 0.01 # Just above bottom (flat bed)
            
            verts = self.create_random_poly(radius)
            # Initial velocity 0
            self.polygons.add(verts, [x, y], [0.0, 0.0], 0.0, 0.0, color='brown')

    def emit_rocks(self):
        radius = 0.025
        margin = 0.01
        
        for attempt in range(10):
            y = 0.3 + np.random.rand() * 0.1
            x = 0.1
            
            # Check overlap
            overlap = False
            for i in range(self.polygons.np):
                dist_sq = (x - self.polygons.x[i,0])**2 + (y - self.polygons.x[i,1])**2
                min_dist = radius + 0.03 + margin
                if dist_sq < min_dist**2:
                    overlap = True
                    break
            
            if not overlap:
                verts = self.create_random_poly(radius)
                self.polygons.add(verts, [x, y], [0.5, 0.0], 0.0, 0.0, color='brown')
                self.rocks_emitted += 1
                print(f"Emitted rock {self.rocks_emitted}")
                break

    ### ************************************************
    ### Plot
    def plot(self):
        if (self.it % self.plot_freq != 0):
            return

        # Calculate Speed at Cell Centers
        u_c = 0.5 * (self.grid.u[0:self.nx, :] + self.grid.u[1:self.nx+1, :])
        v_c = 0.5 * (self.grid.v[:, 0:self.ny] + self.grid.v[:, 1:self.ny+1])
        speed = np.sqrt(u_c**2 + v_c**2)

        plt.clf()
        # Create 3 subplots
        fig = plt.gcf()
        if len(fig.axes) != 3:
            fig.set_size_inches(10, 12)
            
        ax1 = plt.subplot(3, 1, 1)
        ax2 = plt.subplot(3, 1, 2)
        ax3 = plt.subplot(3, 1, 3)
        
        axes = [ax1, ax2, ax3]
        titles = ["VOF (Water Fraction)", "Pressure (Pa)", "Velocity Magnitude (m/s)"]
        data_list = [self.grid.alpha, self.grid.p, speed]
        cmaps = ['Blues', 'viridis', 'plasma']
        # Fixed ranges
        vmins = [0.0, -500.0, 0.0]
        vmaxs = [1.0, 3000.0, 4.0]
        
        # Rotation for visualization
        theta = -math.radians(self.slope_deg)
        c, s = math.cos(theta), math.sin(theta)
        R = np.array([[c, -s], [s, c]])
        
        # Create grid coordinates
        x = np.linspace(0, self.L, self.nx+1)
        y = np.linspace(0, self.H, self.ny+1)
        X, Y = np.meshgrid(x, y)
        
        # Rotate coordinates
        pts = np.column_stack([X.ravel(), Y.ravel()])
        pts_rot = np.dot(pts, R.T)
        X_rot = pts_rot[:, 0].reshape(X.shape)
        Y_rot = pts_rot[:, 1].reshape(Y.shape)
        
        # Corners for limits
        corners = np.array([[0, 0], [self.L, 0], [self.L, self.H], [0, self.H]])
        corners_rot = np.dot(corners, R.T)
        min_x = np.min(corners_rot[:, 0]) - 0.1
        max_x = np.max(corners_rot[:, 0]) + 0.1
        min_y = np.min(corners_rot[:, 1]) - 0.1
        max_y = np.max(corners_rot[:, 1]) + 0.1

        for i, ax in enumerate(axes):
            ax.set_aspect('equal')
            ax.set_title(titles[i])
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            
            # Plot Data
            # pcolormesh shading='flat' needs X,Y one larger than C
            # X_rot is (ny+1, nx+1), data is (nx, ny). Transpose data to (ny, nx)
            im = ax.pcolormesh(X_rot, Y_rot, data_list[i].T, cmap=cmaps[i], shading='flat', 
                               vmin=vmins[i], vmax=vmaxs[i])
            plt.colorbar(im, ax=ax)
            
            # Plot Polygons (Rotated)
            for k in range(self.polygons.np):
                verts = self.polygons.vertices_world[k]
                verts_rot = np.dot(verts, R.T)
                col = self.polygons.color[k]
                poly = PolyPatch(verts_rot, closed=True, facecolor=col, edgecolor='k', alpha=0.8)
                ax.add_patch(poly)
                
            # Draw bed line
            ax.plot([corners_rot[0,0], corners_rot[1,0]], [corners_rot[0,1], corners_rot[1,1]], 'k-', linewidth=2)

        plt.tight_layout()
        plt.pause(0.001)
