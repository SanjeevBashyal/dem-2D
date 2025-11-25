# Generic imports
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolyPatch

# Custom imports
from dem.app.base_app          import *
from dem.src.core.polygons     import *
from dem.src.core.polygon_solver import *
from dem.src.core.polygon_collision import sat_check

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
        self.particles_to_add = 50
        self.added_particles = 0
        self.emit_timer = 0.0
        self.emit_interval = 0.05 # Emit every 0.05 seconds
        
        # Initial solve to set up forces (for walls)
        self.solver.solve(self.polygons, self.dt)

    def get_world_vertices(self, local_verts, pos, theta):
        c = math.cos(theta)
        s = math.sin(theta)
        R = np.array([[c, -s], [s, c]])
        rotated = np.dot(local_verts, R.T)
        return rotated + pos

    def emit_particle(self):
        w = 1.0
        h = 1.0
        
        # Try to find a valid spot
        for attempt in range(20):
            x = (np.random.rand() - 0.5) * 0.8 * w
            y = 0.8 + np.random.rand() * 0.2 * h # Emit from top area
            
            # Random shape: Triangle, Square, Pentagon
            n_sides = np.random.randint(3, 6)
            radius = 0.03 + np.random.rand() * 0.02
            
            verts = []
            for k in range(n_sides):
                angle = 2 * math.pi * k / n_sides
                vx = radius * math.cos(angle)
                vy = radius * math.sin(angle)
                verts.append([vx, vy])
            
            verts = np.array(verts)
            
            # Random rotation
            theta = np.random.rand() * 2 * math.pi
            omega = 0.0
            v = [0.0, -1.0] # Initial downward velocity
            
            # Check overlap
            cand_verts = self.get_world_vertices(verts, [x, y], theta)
            overlap = False
            
            for i in range(self.polygons.np):
                existing_verts = self.polygons.vertices_world[i]
                colliding, _, _ = sat_check(cand_verts, existing_verts)
                if colliding:
                    overlap = True
                    break
            
            if not overlap:
                self.polygons.add(verts, [x, y], v, theta, omega, color='b')
                self.added_particles += 1
                return True
                
        return False

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
        # Emission
        self.emit_timer += self.dt
        if self.emit_timer >= self.emit_interval and self.added_particles < self.particles_to_add:
            if self.emit_particle():
                self.emit_timer = 0.0
        
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
