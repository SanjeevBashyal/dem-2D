# Generic imports
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolyPatch
from matplotlib.patches import Circle

# Custom imports
from dem.app.base_app          import *
from dem.src.core.polygons     import *
from dem.src.core.polygon_solver import *
from dem.src.core.fluid        import *
from dem.src.core.sph_solver   import *
from dem.src.core.sph_coupling import *

### ************************************************
### Open Channel Flow App
class open_channel(base_app):
    ### ************************************************
    ### Constructor
    def __init__(self):
        super().__init__()

        # Parameters
        self.dt        = 5.0e-5 # Smaller dt for SPH
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
        
        # Channel Parameters
        self.slope_deg = 10.0 # Steeper slope
        self.g_mag     = 9.81
        self.gx        = self.g_mag * math.sin(math.radians(self.slope_deg))
        self.gy        = -self.g_mag * math.cos(math.radians(self.slope_deg))
        
        # Geometry
        self.L = 2.0 # Length
        self.H = 0.5 # Height
        
        self.rocks_emitted = 0
        self.max_rocks = 3
        
        # Polygons (Bed + Particles)
        self.polygons = Polygons(material="steel")
        self.dem_solver = PolygonSolver()
        # Stiffer collisions for stability
        self.dem_solver.k1 = 1.0e7
        self.dem_solver.k2 = 2.0e7
        
        # Fluid (SPH)
        self.h = 0.04 # Smoothing length
        self.fluid = FluidParticles(rho0=1000.0, h=self.h, mass=0.01) # Mass depends on spacing
        self.sph_solver = SPHSolver(h=self.h)
        
        # Setup Bed (Fixed Polygons)
        # Bottom
        t = 0.1
        self.add_wall([-0.1, -t], [self.L+0.1, -t], [self.L+0.1, 0], [-0.1, 0])
        
        # Initial Fluid Block
        # Spacing
        dx = self.h * 0.6 # Closer spacing
        nx = int(1.5 / dx) # Fill more length (was 0.5)
        ny = int(0.3 / dx) # Fill more height (was 0.2)
        
        for i in range(nx):
            for j in range(ny):
                x = 0.1 + i * dx
                y = 0.05 + j * dx
                # Add jitter
                x += (np.random.rand() - 0.5) * 0.1 * dx
                y += (np.random.rand() - 0.5) * 0.1 * dx
                
                self.fluid.add([x, y], [0.5, 0.0], color='c')
                
        # Initial Rocks removed to establish flow first

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
        # Not used directly, we override update to handle sub-cycling if needed
        pass

    def update(self):
        # 1. Emitter (Inlet)
        if self.it % 50 == 0:
            self.emit_particles()
            
        # Emit rocks after flow is established (e.g., t > 0.5s)
        if self.t > 0.5 and self.it % 100 == 0 and self.rocks_emitted < self.max_rocks:
            self.emit_rocks()
            
        # 2. Integration Step
        
        # A. Reset Forces
        self.fluid.reset_forces()
        self.polygons.reset_forces()
        
        # B. Gravity
        # Fluid
        self.fluid.f[:, 0] += self.fluid.mass * self.gx
        self.fluid.f[:, 1] += self.fluid.mass * self.gy
        
        # Polygons
        self.polygons.gravity([self.gx, self.gy])
        
        # C. SPH Physics
        self.sph_solver.solve(self.fluid, self.dt)
        
        # D. Coupling
        compute_coupling_forces(self.fluid, self.polygons)
        
        # E. DEM Physics (Collisions)
        # First half update for DEM (Verlet)
        self.polygons.integrate_pos(self.dt)
        
        # Solve collisions (updates forces)
        self.dem_solver.solve(self.polygons, self.dt)
        
        # F. Integration
        # Fluid
        self.fluid.integrate(self.dt)
        
        # Polygons (Second half update)
        self.polygons.integrate_vel(self.dt)
        
        # 3. Outlet (Removal)
        self.remove_particles()
        
        # Time step
        self.it += 1
        self.t  += self.dt

    def emit_particles(self):
        # Emit fluid at x=0
        # Column of particles
        dx = self.h * 0.6 # Closer spacing
        ny = 8 # More particles in column
        for j in range(ny):
            y = 0.05 + j * dx
            # Check if space is free?
            # Simple emit
            self.fluid.add([0.0, y], [1.0, 0.0], color='c') # Initial velocity

    def emit_rocks(self):
        # Emit a rock at inlet
        # Try to find a non-overlapping position
        radius = 0.025
        margin = 0.01
        
        for attempt in range(10):
            y = 0.3 + np.random.rand() * 0.1
            x = 0.1
            
            # Check overlap with existing polygons
            overlap = False
            for i in range(self.polygons.np):
                # Simple bounding circle check
                dist_sq = (x - self.polygons.x[i,0])**2 + (y - self.polygons.x[i,1])**2
                # Assuming max radius of existing is similar
                min_dist = radius + 0.03 + margin # 0.03 is approx max radius of others
                if dist_sq < min_dist**2:
                    overlap = True
                    break
            
            if not overlap:
                verts = self.create_random_poly(radius)
                # Initial velocity matching flow roughly
                self.polygons.add(verts, [x, y], [0.5, 0.0], 0.0, 0.0, color='brown')
                self.rocks_emitted += 1
                print(f"Emitted rock {self.rocks_emitted}")
                break

    def remove_particles(self):
        # Remove fluid > L
        if self.fluid.np > 0:
            mask = self.fluid.x[:, 0] < self.L
            if np.sum(mask) < self.fluid.np:
                # Rebuild arrays
                self.fluid.x = self.fluid.x[mask]
                self.fluid.v = self.fluid.v[mask]
                self.fluid.rho = self.fluid.rho[mask]
                self.fluid.p = self.fluid.p[mask]
                self.fluid.f = self.fluid.f[mask]
                # Color list
                new_color = []
                for i, keep in enumerate(mask):
                    if keep: new_color.append(self.fluid.color[i])
                self.fluid.color = new_color
                self.fluid.np = len(self.fluid.x)
                
        # Remove polygons > L
        # Need delete method in Polygons or manual
        # Polygons class doesn't have delete yet? 
        # It has 'delete' in particles.py but not polygons.py.
        # Let's implement delete in Polygons later if needed.
        # For now, let them fall off.

    ### ************************************************
    ### Plot
    def plot(self):
        if (self.it % self.plot_freq != 0):
            return

        plt.clf()
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_xlim(-0.1, self.L + 0.1)
        ax.set_ylim(-0.1, self.H)
        
        # Plot Polygons
        for i in range(self.polygons.np):
            verts = self.polygons.vertices_world[i]
            c = self.polygons.color[i]
            poly = PolyPatch(verts, closed=True, facecolor=c, edgecolor='k', alpha=0.8)
            ax.add_patch(poly)
            
        # Plot Fluid
        if self.fluid.np > 0:
            # Scatter plot is faster
            plt.scatter(self.fluid.x[:,0], self.fluid.x[:,1], s=10, c='c', alpha=0.5)
            
        plt.pause(0.001)
