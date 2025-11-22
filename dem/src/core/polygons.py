# Generic imports
import math
import numpy as np

# Custom imports
from dem.src.material.material import *

### ************************************************
### Class defining array of polygonal particles
class Polygons:
    ### ************************************************
    ### Constructor
    def __init__(self,
                 material    = "steel",
                 store       = False):

        self.mtr         = material_factory.create(material)
        self.store       = store
        self.np          = 0
        
        # Lists to store vertex data (since polygon vertex counts can vary)
        self.vertices_local = [] # List of numpy arrays
        self.vertices_world = [] # List of numpy arrays

        self.reset()

    ### ************************************************
    ### Reset arrays
    def reset(self):
        self.np = 0
        self.vertices_local = []
        self.vertices_world = []

        # State arrays
        self.m     = np.zeros(0)          # mass
        self.I     = np.zeros(0)          # moment of inertia
        self.x     = np.zeros((0, 2))     # position (center of mass)
        self.v     = np.zeros((0, 2))     # velocity
        self.theta = np.zeros(0)          # rotation angle (radians)
        self.omega = np.zeros(0)          # angular velocity
        
        self.a     = np.zeros((0, 2))     # linear acceleration
        self.alpha = np.zeros(0)          # angular acceleration
        
        self.f     = np.zeros((0, 2))     # force accumulator
        self.tau   = np.zeros(0)          # torque accumulator
        
        self.fixed = np.zeros(0, dtype=bool) # Fixed particles mask

        # Material properties (per particle)
        self.e_wall  = np.zeros(0)
        self.mu_wall = np.zeros(0)
        self.e_part  = np.zeros(0)
        self.mu_part = np.zeros(0)
        self.Y       = np.zeros(0)
        self.G       = np.zeros(0)
        
        self.color   = []

    ### ************************************************
    ### Add a polygon
    ### vertices: list or array of [x, y] local coordinates
    def add(self, vertices, x, v, theta, omega, color='b', fixed=False):
        self.np += 1
        
        # Geometry
        verts = np.array(vertices, dtype=np.float64)
        self.vertices_local.append(verts)
        self.vertices_world.append(np.zeros_like(verts))
        
        # Mass properties calculation
        area, inertia = self.compute_mass_properties(verts)
        mass = area * self.mtr.density
        
        self.m = np.append(self.m, mass)
        self.I = np.append(self.I, inertia * self.mtr.density) # Inertia depends on density too? 
        # Wait, compute_mass_properties usually gives geometric properties. 
        # If it returns geometric moment of inertia (second moment of area), we multiply by density.
        # Let's verify the helper function later.
        
        # State
        self.x = np.vstack([self.x, x]) if self.x.size else np.array([x])
        self.v = np.vstack([self.v, v]) if self.v.size else np.array([v])
        self.theta = np.append(self.theta, theta)
        self.omega = np.append(self.omega, omega)
        
        # Force/Acc placeholders
        self.a = np.vstack([self.a, [0,0]]) if self.a.size else np.zeros((1,2))
        self.alpha = np.append(self.alpha, 0.0)
        self.f = np.vstack([self.f, [0,0]]) if self.f.size else np.zeros((1,2))
        self.tau = np.append(self.tau, 0.0)
        
        self.fixed = np.append(self.fixed, fixed)

        # Material
        self.e_wall  = np.append(self.e_wall,  self.mtr.e_wall)
        self.mu_wall = np.append(self.mu_wall, self.mtr.mu_wall)
        self.e_part  = np.append(self.e_part,  self.mtr.e_part)
        self.mu_part = np.append(self.mu_part, self.mtr.mu_part)
        self.Y       = np.append(self.Y,       self.mtr.Y)
        self.G       = np.append(self.G,       self.mtr.G)
        
        self.color.append(color)
        
        # Initial update of world vertices
        self.update_vertices_single(self.np - 1)

    ### ************************************************
    ### Compute Area and Moment of Inertia for a polygon
    ### Assumes uniform density = 1 for the return values (Geometric properties)
    def compute_mass_properties(self, vertices):
        # Shoelace formula for Area
        # I = sum ... complex for arbitrary polygon
        # For now, let's implement a simple version or assume centered.
        # If vertices are relative to COM, sum(x) = 0, sum(y) = 0.
        
        # https://en.wikipedia.org/wiki/Polygon#Area
        x = vertices[:, 0]
        y = vertices[:, 1]
        
        # Area
        # A = 0.5 * |sum(x_i y_{i+1} - x_{i+1} y_i)|
        # We need to wrap around
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        
        area = 0.5 * np.abs(np.sum(x * y_next - x_next * y))
        
        # Moment of Inertia about the origin (which should be COM)
        # I_z = sum( (x_i^2 + x_i*x_{i+1} + x_{i+1}^2 + y_i^2 + y_i*y_{i+1} + y_{i+1}^2) * (x_i*y_{i+1} - x_{i+1}*y_i) ) / 12
        # This is the formula for I about the origin.
        
        cross_prod = x * y_next - x_next * y
        term_x = x**2 + x * x_next + x_next**2
        term_y = y**2 + y * y_next + y_next**2
        
        I_z = np.sum((term_x + term_y) * cross_prod) / 12.0
        
        return area, abs(I_z)

    ### ************************************************
    ### Update world vertices based on current state
    def update_vertices(self):
        for i in range(self.np):
            self.update_vertices_single(i)

    def update_vertices_single(self, i):
        # Rotation matrix
        c = math.cos(self.theta[i])
        s = math.sin(self.theta[i])
        R = np.array([[c, -s], [s, c]])
        
        # Local vertices
        local = self.vertices_local[i]
        
        # Rotate and translate
        # World = R * Local + Pos
        # local is (N, 2). R is (2, 2).
        # We want (N, 2) output.
        # (R @ local.T).T + pos
        
        rotated = np.dot(local, R.T)
        self.vertices_world[i] = rotated + self.x[i]

    ### ************************************************
    ### Reset forces and torques
    def reset_forces(self):
        self.f[:] = 0.0
        self.tau[:] = 0.0

    ### ************************************************
    ### Add gravity
    def gravity(self, g):
        # g is a vector [gx, gy] or scalar (assumed y)
        # If scalar, assume vertical
        if isinstance(g, (float, int)):
            self.f[:, 1] -= self.m * g
        else:
            self.f[:, 0] += self.m * g[0]
            self.f[:, 1] += self.m * g[1]

    ### ************************************************
    ### Velocity Verlet: Position Update
    ### x(t+dt) = x(t) + v(t)dt + 0.5*a(t)dt^2
    def integrate_pos(self, dt):
        # Calculate current acceleration from forces
        # (Assuming f includes all forces from previous step or gravity)
        # Note: For the very first step, f might be just gravity.
        
        # Store old acceleration for velocity update
        # We need to handle the shape carefully
        self.a[:, 0] = self.f[:, 0] / self.m
        self.a[:, 1] = self.f[:, 1] / self.m
        self.alpha   = self.tau / self.I
        
        # Zero out acceleration for fixed particles
        self.a[self.fixed] = 0.0
        self.alpha[self.fixed] = 0.0
        
        # Update position
        self.x += self.v * dt + 0.5 * self.a * dt**2
        
        # Update rotation
        self.theta += self.omega * dt + 0.5 * self.alpha * dt**2
        
        # Update world vertices
        self.update_vertices()

    ### ************************************************
    ### Velocity Verlet: Velocity Update
    ### v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))dt
    def integrate_vel(self, dt):
        # Calculate new acceleration from new forces
        a_new = np.zeros_like(self.a)
        a_new[:, 0] = self.f[:, 0] / self.m
        a_new[:, 1] = self.f[:, 1] / self.m
        
        alpha_new = self.tau / self.I
        
        # Zero out acceleration for fixed particles
        a_new[self.fixed] = 0.0
        alpha_new[self.fixed] = 0.0
        
        # Update velocity
        self.v += 0.5 * (self.a + a_new) * dt
        
        # Update angular velocity
        self.omega += 0.5 * (self.alpha + alpha_new) * dt
        
        # Zero out velocity for fixed particles (safety)
        self.v[self.fixed] = 0.0
        self.omega[self.fixed] = 0.0
        
        # Update stored acceleration (optional, but good for consistency)
        self.a[:] = a_new[:]
        self.alpha[:] = alpha_new[:]

