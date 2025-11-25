
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import math

# Custom imports
from dem.app.base_app          import *
from dem.src.core.polygons     import *
from dem.src.core.polygon_solver import *
from dem.src.core.polygon_collision import sat_check
from dem.src.fvm.grid          import *
from dem.src.fvm.fluid_solver  import *
from dem.src.fvm.coupling      import *

class sediment_transport_k_omega_app(base_app):
    def __init__(self, run_name="sediment_transport", slope=0.01, u_inlet=1.0):
        super().__init__()
        self.run_name = run_name
        
        # Parameters
        self.dt        = 1.0e-4 # Small dt for stability with IBM
        self.t_max     = 2.0
        self.plot_freq = 100
        self.nt        = int(self.t_max/self.dt)
        self.it        = 0
        self.t         = 0.0
        
        # Domain
        self.L = 2.0
        self.H = 0.4 # Increased to 0.4m
        self.nx = 200
        self.ny = 40 # Increased to 40 to keep dy=0.01
        self.dx = self.L / self.nx
        self.dy = self.H / self.ny
        
        # Physics
        self.slope = slope
        self.u_inlet = u_inlet
        self.g_mag = 9.81
        theta = np.arctan(slope)
        self.gx = self.g_mag * np.sin(theta)
        self.gy = -self.g_mag * np.cos(theta)
        
        # FVM Grid & Solver
        self.grid = StaggeredGrid(self.nx, self.ny, self.dx, self.dy)
        self.fluid_solver = FVMSolver(self.grid)
        self.fluid_solver.variable_density = True # Enable VOF
        self.fluid_solver.turbulence_model = 'k-omega-sst' # RANS k-omega SST
        self.fluid_solver.g = np.array([self.gx, self.gy])
        
        # Polygons
        self.polygons = Polygons(material="sand")
        self.polygons.mtr.density = 2500.0 # Gravel
        self.dem_solver = PolygonSolver()
        self.dem_solver.k1 = 1.0e7
        
        # Initial Conditions (Fluid)
        h_water = 0.2 # Increased to 0.2m
        j_water = int(h_water / self.dy)
        
        self.grid.alpha[:, :j_water] = 1.0
        self.grid.u[:, :j_water] = self.u_inlet
        
        # Turbulence Init
        # Explicitly allocate arrays
        self.fluid_solver.k = np.zeros((self.nx, self.ny))
        self.fluid_solver.omega_t = np.zeros((self.nx, self.ny)) # Specific Dissipation Rate
        
        # k = 1.5 * (I * u)^2
        # eps = C_mu^0.75 * k^1.5 / L
        # omega = eps / (beta_star * k)
        k_init = 1.5 * (0.05 * self.u_inlet)**2
        L_scale = 0.07 * h_water
        eps_init = 0.09**0.75 * k_init**1.5 / L_scale
        omega_init = eps_init / (0.09 * k_init)
        
        self.fluid_solver.k[:, :j_water] = k_init
        self.fluid_solver.omega_t[:, :j_water] = omega_init
        # Set background small values
        self.fluid_solver.k[self.fluid_solver.k == 0] = 1e-8
        self.fluid_solver.omega_t[self.fluid_solver.omega_t == 0] = 1e-8
            
        # Boundary Conditions (Fixed Pressure at Outlet)
        self.setup_pressure_bc(h_water)
        
        # Bed Initialization
        self.create_packed_bed()
        
        # Flux Tracking
        self.flux_plane_x = 1.6 # Moved further downstream
        self.flux_history = []
        self.time_history = []
        self.max_v_history = []
        self.max_k_history = []
        self.max_eps_history = []
        
        # Visualization
        self.fig = plt.figure(figsize=(10, 15), constrained_layout=True)
        self.gs = GridSpec(5, 2, width_ratios=[20, 1], figure=self.fig)
        
        self.ax_fluid = self.fig.add_subplot(self.gs[0, 0])
        self.ax_p     = self.fig.add_subplot(self.gs[1, 0])
        self.ax_u     = self.fig.add_subplot(self.gs[2, 0])
        self.ax_k     = self.fig.add_subplot(self.gs[3, 0])
        self.ax_omega = self.fig.add_subplot(self.gs[4, 0])
        
        # Colorbars axes
        self.cax_p     = self.fig.add_subplot(self.gs[1, 1])
        self.cax_u     = self.fig.add_subplot(self.gs[2, 1])
        self.cax_k     = self.fig.add_subplot(self.gs[3, 1])
        self.cax_omega = self.fig.add_subplot(self.gs[4, 1])

    def setup_pressure_bc(self, h_water):
        self.fluid_solver.p_fixed_mask = np.zeros((self.nx, self.ny), dtype=bool)
        self.fluid_solver.p_fixed_val = np.zeros((self.nx, self.ny))
        
        # Outlet at i = nx-1
        for j in range(self.ny):
            y = (j + 0.5) * self.dy
            if y <= h_water:
                self.fluid_solver.p_fixed_mask[self.nx-1, j] = True
                self.fluid_solver.p_fixed_val[self.nx-1, j] = self.fluid_solver.rho_water * abs(self.gy) * (h_water - y)
            else:
                self.fluid_solver.p_fixed_mask[self.nx-1, j] = True
                self.fluid_solver.p_fixed_val[self.nx-1, j] = 0.0

    def get_world_vertices(self, local_verts, pos, theta):
        c = math.cos(theta)
        s = math.sin(theta)
        R = np.array([[c, -s], [s, c]])
        rotated = np.dot(local_verts, R.T)
        return rotated + pos

    def create_packed_bed(self):
        print("Initializing packed bed...")
        # Larger length: 0.5 to 1.5 (1.0m length)
        bed_start_x = 0.5
        bed_end_x = 1.5
        particle_size = 0.02
        
        target_particles = 60 # Keep count roughly same as before
        
        # Floor Wall
        self.polygons.add([[-0.1, -0.05], [self.L+0.1, -0.05], [self.L+0.1, 0.0], [-0.1, 0.0]], 
                          [0,0], [0,0], 0.0, 0.0, color='k', fixed=True)
        
        # Random Packing
        n_attempts = 2000
        count = 0
        for _ in range(n_attempts):
            if count >= target_particles:
                break
                
            x = bed_start_x + np.random.rand() * (bed_end_x - bed_start_x)
            y = particle_size/2 + np.random.rand() * 0.05 # Start low
            
            # Random Shape
            n_sides = np.random.randint(4, 7)
            radius = particle_size / 2 * (0.8 + 0.4 * np.random.rand())
            verts = []
            for k in range(n_sides):
                angle = 2 * math.pi * k / n_sides
                vx = radius * math.cos(angle)
                vy = radius * math.sin(angle)
                verts.append([vx, vy])
            verts = np.array(verts)
            
            theta = np.random.rand() * 2 * math.pi
            
            # Check Overlap
            cand_verts = self.get_world_vertices(verts, [x, y], theta)
            overlap = False
            for i in range(self.polygons.np):
                existing_verts = self.polygons.vertices_world[i]
                colliding, _, _ = sat_check(cand_verts, existing_verts)
                if colliding:
                    overlap = True
                    break
            
            if not overlap:
                self.polygons.add(verts, [x, y], [0.0, 0.0], theta, 0.0, color='brown')
                count += 1
                
        print(f"Initialized {self.polygons.np} particles (including wall). Target was {target_particles}.")

    def update(self):
        # 1. Inlet BC
        h_inlet = 0.2 # Increased to 0.2m
        j_max = int(h_inlet / self.dy)
        self.grid.u[0, :j_max] = self.u_inlet
        self.grid.u[0, j_max:] = 0.0
        self.grid.v[0, :] = 0.0
        self.grid.alpha[0, :j_max] = 1.0
        self.grid.alpha[0, j_max:] = 0.0
        
        # Turbulence BCs
        k_inlet = 1.5 * (0.05 * self.u_inlet)**2
        L_scale = 0.07 * h_inlet
        eps_inlet = 0.09**0.75 * k_inlet**1.5 / L_scale
        omega_inlet = eps_inlet / (0.09 * k_inlet)
        
        if self.fluid_solver.k is not None:
            self.fluid_solver.k[0, :j_max] = k_inlet
            self.fluid_solver.k[0, j_max:] = 1e-8
            self.fluid_solver.omega_t[0, :j_max] = omega_inlet
            self.fluid_solver.omega_t[0, j_max:] = 1e-8
        
        # 2. Rasterize Solid for IBM
        self.grid.rasterize_polygons(self.polygons)
        
        # 3. Fluid Advection (Predictor)
        advect_velocity(self.grid.u, self.grid.v, self.grid.u, self.grid.v, 
                        self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy, self.dt)
        
        # 3.5 Turbulence Solve & Diffusion
        if self.fluid_solver.turbulence_model == 'k-omega-sst' and self.fluid_solver.k is not None:
             solve_k_omega_sst(self.fluid_solver.k, self.fluid_solver.omega_t, self.grid.nu_t, 
                             self.grid.u, self.grid.v, 
                             self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy, self.dt,
                             self.fluid_solver.mu_water/self.fluid_solver.rho_water, 
                             self.fluid_solver.a1, self.fluid_solver.beta_star, 
                             self.fluid_solver.sigma_k1, self.fluid_solver.sigma_w1, self.fluid_solver.beta1, self.fluid_solver.alpha1,
                             self.fluid_solver.sigma_k2, self.fluid_solver.sigma_w2, self.fluid_solver.beta2, self.fluid_solver.alpha2)
                           
             # Diffuse Velocity (Viscous + Turbulent)
             diffuse_velocity(self.grid.u, self.grid.v, self.grid.nu_t, 
                              self.fluid_solver.mu_water/self.fluid_solver.rho_water, 
                              self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy, self.dt)
        
        # 4. VOF Advection
        advect_scalar_bfecc(self.grid.alpha, self.grid.u, self.grid.v, 
                            self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy, self.dt)
        
        # 5. Gravity
        self.grid.u[1:-1, :] += self.gx * self.dt
        self.grid.v[:, 1:-1] += self.gy * self.dt
        
        # 6. IBM Forcing (Solid -> Fluid)
        apply_ibm_forcing(self.grid)
        
        # 7. Pressure Solve
        compute_divergence(self.grid.u, self.grid.v, self.grid.div, 
                           self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy)
        
        rho_field = self.grid.alpha * self.fluid_solver.rho_water + (1.0 - self.grid.alpha) * self.fluid_solver.rho_air
        rhs = self.grid.div / self.dt
        
        solve_pressure_sor_variable_rho(self.grid.p, rhs, rho_field, 
                                        self.fluid_solver.p_fixed_mask, self.fluid_solver.p_fixed_val,
                                        self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy, 
                                        50, 1e-3, 1.7)
                                        
        # 8. Correct Velocity
        correct_velocity_variable_rho(self.grid.u, self.grid.v, self.grid.p, rho_field,
                                      self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy, self.dt)
                                      
        # 9. DEM Integration
        self.polygons.integrate_pos(self.dt)
        self.polygons.reset_forces()
        
        # Gravity
        self.polygons.gravity([self.gx, self.gy])
        
        # Hydro Forces (Fluid -> Solid)
        compute_hydrodynamic_forces(self.grid, self.polygons)
        
        # Collisions
        self.dem_solver.solve(self.polygons, self.dt)
        
        self.polygons.integrate_vel(self.dt)
        
        # 10. Outlet BC (Zero Gradient)
        self.grid.u[self.nx, :] = self.grid.u[self.nx-1, :]
        self.grid.v[self.nx-1, :] = self.grid.v[self.nx-2, :]
        self.grid.alpha[self.nx-1, :] = self.grid.alpha[self.nx-2, :]
        
        # Stats
        self.it += 1
        self.t += self.dt
        
        if self.it % 100 == 0:
            flux = np.sum(self.polygons.x[:, 0] > self.flux_plane_x)
            self.flux_history.append(flux)
            self.time_history.append(self.t)
            
            # Calculate stats
            max_v = np.max(np.linalg.norm(self.polygons.v, axis=1)) if self.polygons.np > 0 else 0.0
            max_k = np.max(self.fluid_solver.k) if self.fluid_solver.k is not None else 0.0
            max_omega = np.max(self.fluid_solver.omega_t) if self.fluid_solver.omega_t is not None else 0.0
            
            self.max_v_history.append(max_v)
            self.max_k_history.append(max_k)
            self.max_eps_history.append(max_omega)
            
            print(f"t={self.t:.3f}, Flux={flux}, Max V_p={max_v:.3f}, Max k={max_k:.4f}, Max omega={max_omega:.4f}", flush=True)

    def plot(self, frame):
        # Helper to plot particles
        def plot_particles(ax):
            for i in range(self.polygons.np):
                poly = plt.Polygon(self.polygons.vertices_world[i], closed=True, facecolor=self.polygons.color[i], edgecolor='black')
                ax.add_patch(poly)
        
        # 1. Main View (VOF + Particles)
        self.ax_fluid.clear()
        self.ax_fluid.imshow(self.grid.alpha.T, origin='lower', extent=[0, self.L, 0, self.H], cmap='Blues', alpha=0.5, vmin=0, vmax=1)
        plot_particles(self.ax_fluid)
        self.ax_fluid.set_xlim(0, self.L)
        self.ax_fluid.set_ylim(0, self.H)
        self.ax_fluid.set_title(f"Sediment Transport ({self.run_name}) - t={self.t:.3f}s")
        
        # 2. Pressure (Fixed Scale: -500 to 2500 Pa)
        self.ax_p.clear()
        self.cax_p.clear()
        im_p = self.ax_p.imshow(self.grid.p.T, origin='lower', extent=[0, self.L, 0, self.H], cmap='viridis', vmin=-500, vmax=2500)
        plot_particles(self.ax_p)
        self.ax_p.set_title("Pressure (Pa)")
        self.fig.colorbar(im_p, cax=self.cax_p)
        
        # 3. Velocity Magnitude (Fixed Scale: 0 to 2.0 m/s)
        self.ax_u.clear()
        self.cax_u.clear()
        u_cc = 0.5 * (self.grid.u[:-1, :] + self.grid.u[1:, :])
        v_cc = 0.5 * (self.grid.v[:, :-1] + self.grid.v[:, 1:])
        vel_mag = np.sqrt(u_cc**2 + v_cc**2)
        im_u = self.ax_u.imshow(vel_mag.T, origin='lower', extent=[0, self.L, 0, self.H], cmap='plasma', vmin=0, vmax=2.0)
        plot_particles(self.ax_u)
        self.ax_u.set_title("Velocity Magnitude (m/s)")
        self.fig.colorbar(im_u, cax=self.cax_u)
        
        # 4. TKE (Adjusted Scale: 0 to 0.02 m2/s2)
        self.ax_k.clear()
        self.cax_k.clear()
        if self.fluid_solver.k is not None:
            im_k = self.ax_k.imshow(self.fluid_solver.k.T, origin='lower', extent=[0, self.L, 0, self.H], cmap='inferno', vmin=0, vmax=0.02)
            plot_particles(self.ax_k)
            self.ax_k.set_title("Turbulent Kinetic Energy (m2/s2)")
            self.fig.colorbar(im_k, cax=self.cax_k)
            
        # 5. Dissipation (Omega)
        self.ax_omega.clear()
        self.cax_omega.clear()
        if self.fluid_solver.omega_t is not None:
            # Use log scale for omega
            omega_plot = np.log10(self.fluid_solver.omega_t + 1e-10)
            im_omega = self.ax_omega.imshow(omega_plot.T, origin='lower', extent=[0, self.L, 0, self.H], cmap='magma', vmin=0, vmax=4)
            plot_particles(self.ax_omega)
            self.ax_omega.set_title("Specific Dissipation Rate (log10 Omega) (1/s)")
            self.fig.colorbar(im_omega, cax=self.cax_omega)

def run_simulation(run_name, slope, u_inlet):
    app = sediment_transport_k_omega_app(run_name, slope, u_inlet)
    
    def update_anim(frame):
        for _ in range(200): # Substeps (Increased to 200)
            app.update()
        app.plot(frame)
        
    ani = animation.FuncAnimation(app.fig, update_anim, frames=list(range(50)), interval=200, cache_frame_data=False)
    ani.save(f"sediment_{run_name}_refined.gif", writer='pillow', fps=10)
    
    # Save extended data to CSV
    data = np.column_stack((app.time_history, app.flux_history, app.max_v_history, app.max_k_history, app.max_eps_history))
    header = "Time,Flux,Max_Particle_Vel,Max_K,Max_Omega"
    np.savetxt(f"flux_{run_name}_refined.csv", data, delimiter=",", header=header, comments='')
    print(f"Simulation {run_name} Complete.")

if __name__ == "__main__":
    # Run Sub-Critical
    print("Running Sub-Critical Refined...")
    run_simulation("sub_critical", 0.01, 1.0)
    
    # Run Super-Critical
    print("Running Super-Critical Refined...")
    run_simulation("super_critical", 0.02, 1.5)
