import numpy as np

class FluidParticles:
    def __init__(self, rho0=1000.0, h=0.1, mass=1.0):
        self.rho0 = rho0
        self.h    = h
        self.mass = mass # Assumed constant for all particles for now
        
        self.np = 0
        
        # Arrays
        self.x   = np.zeros((0, 2)) # Position
        self.v   = np.zeros((0, 2)) # Velocity
        self.rho = np.zeros(0)      # Density
        self.p   = np.zeros(0)      # Pressure
        self.f   = np.zeros((0, 2)) # Force accumulator
        
        self.color = [] # Visualization color

    def reset(self):
        self.np = 0
        self.x   = np.zeros((0, 2))
        self.v   = np.zeros((0, 2))
        self.rho = np.zeros(0)
        self.p   = np.zeros(0)
        self.f   = np.zeros((0, 2))
        self.color = []

    def add(self, x, v, color='c'):
        self.np += 1
        self.x = np.vstack([self.x, x]) if self.x.size else np.array([x], dtype=np.float64)
        self.v = np.vstack([self.v, v]) if self.v.size else np.array([v], dtype=np.float64)
        self.rho = np.append(self.rho, self.rho0)
        self.p   = np.append(self.p, 0.0)
        self.f   = np.vstack([self.f, [0.0, 0.0]]) if self.f.size else np.zeros((1,2))
        self.color.append(color)

    def reset_forces(self):
        self.f[:] = 0.0

    def integrate(self, dt):
        # Symplectic Euler or Velocity Verlet
        # v += a * dt
        # x += v * dt
        
        # Acceleration
        a = self.f / self.mass # Assuming uniform mass
        
        self.v += a * dt
        self.x += self.v * dt
