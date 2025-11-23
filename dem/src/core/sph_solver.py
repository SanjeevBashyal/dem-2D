import numpy as np
import math
import numba as nb

# Constants
GAMMA = 7.0
RHO0  = 1000.0
C_S   = 50.0 # Speed of sound (should be > 10 * max_velocity)
B     = RHO0 * C_S**2 / GAMMA

### ************************************************
### Cubic Spline Kernel
### q = r / h
### W(r, h) = alpha_d * ...
### In 2D, alpha_d = 10 / (7 * pi * h^2)
@nb.njit(cache=True)
def kernel(r, h):
    q = r / h
    val = 0.0
    alpha = 10.0 / (7.0 * math.pi * h**2)
    
    if 0 <= q < 1:
        val = alpha * (1 - 1.5 * q**2 + 0.75 * q**3)
    elif 1 <= q < 2:
        val = alpha * 0.25 * (2 - q)**3
        
    return val

### ************************************************
### Gradient of Cubic Spline Kernel
### Returns scalar part of gradient (needs to be multiplied by vector r_ij / r)
### grad W = dW/dq * dq/dr * (r/r)
### dq/dr = 1/h
@nb.njit(cache=True)
def kernel_grad_scalar(r, h):
    q = r / h
    val = 0.0
    alpha = 10.0 / (7.0 * math.pi * h**2)
    
    if 0 <= q < 1:
        # d/dq (1 - 1.5q^2 + 0.75q^3) = -3q + 2.25q^2
        val = alpha * (-3.0 * q + 2.25 * q**2)
    elif 1 <= q < 2:
        # d/dq (0.25(2-q)^3) = 0.75(2-q)^2 * (-1)
        val = -alpha * 0.75 * (2.0 - q)**2
        
    return val / h # Chain rule dq/dr

### ************************************************
### SPH Solver Class
class SPHSolver:
    def __init__(self, h):
        self.h = h
        # Grid for neighbor search
        self.grid_size = 2.0 * h # Support radius is 2h for cubic spline
        
    def solve(self, fluid, dt):
        n = fluid.np
        if n == 0: return

        x = fluid.x
        v = fluid.v
        rho = fluid.rho
        p = fluid.p
        f = fluid.f
        m = fluid.mass
        
        # 1. Neighbor Search (Naive O(N^2) for now, optimize later)
        # Or use a simple grid if N is large.
        # Let's stick to N^2 with numba for simplicity and correctness first.
        
        # 2. Compute Density and Pressure
        compute_density_pressure(x, rho, p, m, self.h, n)
        
        # 3. Compute Internal Forces (Pressure + Viscosity)
        compute_forces(x, v, rho, p, f, m, self.h, n)

### ************************************************
### Compute Density and Pressure
@nb.njit(cache=True)
def compute_density_pressure(x, rho, p, m, h, n):
    for i in range(n):
        rho[i] = 0.0
        for j in range(n):
            dist_sq = (x[i,0]-x[j,0])**2 + (x[i,1]-x[j,1])**2
            dist = math.sqrt(dist_sq)
            
            if dist < 2.0 * h:
                rho[i] += m * kernel(dist, h)
                
        # Tait's Equation of State
        # P = B * ((rho/rho0)^gamma - 1)
        ratio = rho[i] / RHO0
        # Clamp ratio to avoid instability if rho drops too low (though it shouldn't)
        if ratio < 1.0: ratio = 1.0 
        
        p[i] = B * (ratio**GAMMA - 1.0)

### ************************************************
### Compute Internal Forces
@nb.njit(cache=True)
def compute_forces(x, v, rho, p, f, m, h, n):
    # Viscosity parameters
    alpha_visc = 0.1
    c_s = C_S
    
    for i in range(n):
        for j in range(i+1, n):
            dist_sq = (x[i,0]-x[j,0])**2 + (x[i,1]-x[j,1])**2
            dist = math.sqrt(dist_sq)
            
            if dist < 2.0 * h and dist > 1e-6:
                # Kernel Gradient
                grad_w_scalar = kernel_grad_scalar(dist, h)
                
                # Vector r_ij = r_i - r_j
                dx = x[i,0] - x[j,0]
                dy = x[i,1] - x[j,1]
                
                # Normalized vector
                nx = dx / dist
                ny = dy / dist
                
                grad_w_x = grad_w_scalar * nx
                grad_w_y = grad_w_scalar * ny
                
                # Pressure Force
                # Fp_i = - sum m_j * (pi/rhoi^2 + pj/rhoj^2) * grad_W
                term_p = (p[i]/(rho[i]**2) + p[j]/(rho[j]**2))
                
                fp_x = -m * term_p * grad_w_x
                fp_y = -m * term_p * grad_w_y
                
                # Viscosity Force (Monaghan Artificial Viscosity)
                # Pi_ij = ...
                v_ij_x = v[i,0] - v[j,0]
                v_ij_y = v[i,1] - v[j,1]
                
                v_dot_r = v_ij_x * dx + v_ij_y * dy
                
                visc_term = 0.0
                if v_dot_r < 0:
                    mu_v = h * v_dot_r / (dist_sq + 0.01 * h**2)
                    rho_bar = 0.5 * (rho[i] + rho[j])
                    visc_term = -alpha_visc * c_s * mu_v / rho_bar
                    # Add beta term if needed
                
                # Total force contribution (Pressure + Viscosity)
                # F = -m * (PressureTerm + ViscosityTerm) * grad_W
                # Note: My PressureTerm above already includes the negative sign and mass?
                # Wait, standard formula:
                # dv/dt = - sum m (p/rho^2 + p/rho^2 + Pi_ij) grad_W
                # So Force = m_i * dv/dt
                
                # Let's combine:
                # term = p_i/rho_i^2 + p_j/rho_j^2 + Pi_ij
                
                term_total = term_p + visc_term
                
                fx = -m * m * term_total * grad_w_x
                fy = -m * m * term_total * grad_w_y
                
                # Apply to i
                f[i,0] += fx
                f[i,1] += fy
                
                # Apply opposite to j
                f[j,0] -= fx
                f[j,1] -= fy
