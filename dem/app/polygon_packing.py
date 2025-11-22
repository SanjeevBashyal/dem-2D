# Generic imports
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolyPatch

# Custom imports
from dem.app.base_app          import *
from dem.src.core.polygons     import *
from dem.src.core.polygon_solver import *

### ************************************************
### Polygon packing app
class polygon_packing(base_app):
    ### ************************************************
    ### Constructor
    def __init__(self):
        super().__init__()

        # Parameters
        self.dt        = 1.0e-4
        self.t_max     = 2.0
        self.plot_freq = 100
        self.nt        = int(self.t_max/self.dt)
        self.it        = 0
        self.t         = 0.0
        self.plot_it   = 0
        self.plot_show = True
        self.plot_png  = False
        self.plot_trajectory = False
        self.d_lst     = [] # Needed for plot
        
        # Polygons
        self.polygons = Polygons(material="steel")
        self.solver   = PolygonSolver()
        
        # Walls (Static polygons)
        # Container: [-0.5, 0.5] x [0.0, 1.0]
        w = 1.0
        h = 1.0
        t = 0.1 # thickness
        
        # Floor
        self.add_wall([-w/2 - t, -t], [w/2 + t, -t], [w/2 + t, 0], [-w/2 - t, 0])
        # Left Wall
        self.add_wall([-w/2 - t, 0], [-w/2, 0], [-w/2, h], [-w/2 - t, h])
        # Right Wall
        self.add_wall([w/2, 0], [w/2 + t, 0], [w/2 + t, h], [w/2, h])
        
        # Particles
        # Add random polygons
        np_particles = 50
        for i in range(np_particles):
            x = (np.random.rand() - 0.5) * 0.8 * w
            y = 0.2 + np.random.rand() * 0.8 * h
            
            # Random shape: Triangle, Square, Pentagon
            n_sides = np.random.randint(3, 6)
            radius = 0.03 + np.random.rand() * 0.02
            
            verts = []
            for k in range(n_sides):
                angle = 2 * math.pi * k / n_sides
                vx = radius * math.cos(angle)
                vy = radius * math.sin(angle)
                verts.append([vx, vy])
            
            # Random rotation
            theta = np.random.rand() * 2 * math.pi
            omega = 0.0
            v = [0.0, 0.0]
            
            self.polygons.add(verts, [x, y], v, theta, omega, color='b')

        # Mark walls as fixed (first 3)
        # self.fixed_indices = [0, 1, 2]
        # Handled by fixed=True in add_wall

        # Initial solve to set up forces
        self.solver.solve(self.polygons, self.dt)

    def add_wall(self, p1, p2, p3, p4):
        verts = [p1, p2, p3, p4]
        self.polygons.add(verts, [0,0], [0,0], 0.0, 0.0, color='k', fixed=True)

    ### ************************************************
    ### Compute forces
    def forces(self):
        # 1. Integrate Position (First half of Verlet)
        # We need to skip fixed particles if we want them to be absolutely static.
        # But infinite mass handles dynamics.
        # However, integrate_pos updates x += v*dt. If v is 0, x doesn't change.
        # Forces will produce a = f/m. If m is huge, a is tiny.
        # So it should be fine.
        
        self.polygons.integrate_pos(self.dt)
        
        # 2. Reset Forces
        self.polygons.reset_forces()
        
        # 3. Gravity
        self.polygons.gravity(9.81)
        
        # 4. Solve Contacts (Calculate new forces)
        self.solver.solve(self.polygons, self.dt)

    ### ************************************************
    ### Update positions (Second half of Verlet)
    def update(self):
        # 5. Integrate Velocity
        self.polygons.integrate_vel(self.dt)
        
        # Base update (time stepping)
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
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.1, 1.2)
        
        for i in range(self.polygons.np):
            verts = self.polygons.vertices_world[i]
            c = self.polygons.color[i]
            poly = PolyPatch(verts, closed=True, facecolor=c, edgecolor='k', alpha=0.8)
            ax.add_patch(poly)
            
        plt.pause(0.001)
