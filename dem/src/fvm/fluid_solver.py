import numpy as np
import numba as nb

class FVMSolver:
    def __init__(self, grid):
        self.grid = grid
        self.rho_water = 1000.0
        self.rho_air   = 1.0
        self.mu_water  = 0.001
        self.mu_air    = 0.00001
        self.g         = 9.81
        
        # Turbulence Parameters
        # Turbulence Parameters (k-omega)
        self.k = None     # Turbulent Kinetic Energy
        self.omega_t = None # Specific Dissipation Rate
        # self.nu_t is now in grid
        
        # k-omega Constants (Standard Wilcox 2006 or similar)
        self.beta_star = 0.09
        self.alpha_kw  = 0.52
        self.beta_kw   = 0.072
        self.sigma_k   = 0.5
        self.sigma_w   = 0.5
        
        # SOR Parameters
        self.max_iter = 100
        self.tol      = 1e-4
        self.omega    = 1.7

    def solve(self, dt):
        # 1. Advect Velocity (Semi-Lagrangian)
        advect_velocity(self.grid.u, self.grid.v, self.grid.u, self.grid.v, 
                        self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy, dt)
        
        # 1.5 Diffusion (Turbulence)
        # 1.5 Turbulence (k-omega)
        if self.k is None:
            self.k = np.zeros((self.grid.nx, self.grid.ny)) + 1e-6
            self.omega_t = np.zeros((self.grid.nx, self.grid.ny)) + 100.0 # High initial dissipation
            
        solve_k_omega(self.k, self.omega_t, self.grid.nu_t, self.grid.u, self.grid.v, 
                      self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy, dt,
                      self.mu_water/self.rho_water, self.beta_star, self.alpha_kw, self.beta_kw, self.sigma_k, self.sigma_w)
        
        # Diffuse Velocity
        diffuse_velocity(self.grid.u, self.grid.v, self.grid.nu_t, self.mu_water/self.rho_water, 
                         self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy, dt)
        
        # 2. Apply Gravity
        # Simple explicit
        self.grid.v[:, 1:-1] -= self.g * dt
        
        # 3. Pressure Solve (Projection)
        # Compute Divergence
        compute_divergence(self.grid.u, self.grid.v, self.grid.div, 
                           self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy)
        
        # Solve Poisson: laplacian(p) = rho/dt * div
        # Note: rho varies, but for standard projection we often use constant rho 
        # or solve div(1/rho grad p) = div/dt.
        # For simplicity, let's assume water density for pressure solve or use average.
        rho = self.rho_water 
        rhs = self.grid.div * rho / dt
        
        solve_pressure_sor(self.grid.p, rhs, self.grid.nx, self.grid.ny, 
                           self.grid.dx, self.grid.dy, self.max_iter, self.tol, self.omega)
        
        # 4. Correct Velocity
        correct_velocity(self.grid.u, self.grid.v, self.grid.p, 
                         self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy, dt, rho)
        
        # 5. Advect VOF (Alpha) using BFECC for sharpness
        advect_scalar_bfecc(self.grid.alpha, self.grid.u, self.grid.v, 
                            self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy, dt)

### ************************************************
### Semi-Lagrangian Advection (Velocity)
@nb.njit(cache=True)
def advect_velocity(u, v, u_old, v_old, nx, ny, dx, dy, dt):
    # Advect u (defined at i+0.5, j)
    for i in range(1, nx):
        for j in range(ny):
            x = i * dx
            y = (j + 0.5) * dy
            
            # Velocity at (x,y)
            # u is at (i,j), v is at (i-0.5, j+0.5) -> need interpolation
            # Simple 1st order for now
            uc = u_old[i, j]
            # vc needs averaging
            vc = 0.25 * (v_old[i-1, j] + v_old[i, j] + v_old[i-1, j+1] + v_old[i, j+1])
            
            # Backtrace
            x_prev = x - uc * dt
            y_prev = y - vc * dt
            
            # Interpolate u_old at (x_prev, y_prev)
            u[i, j] = sample_field(u_old, x_prev, y_prev, dx, dy, 0.0, 0.5)

    # Advect v (defined at i, j+0.5)
    for i in range(nx):
        for j in range(1, ny):
            x = (i + 0.5) * dx
            y = j * dy
            
            # Velocity at (x,y)
            uc = 0.25 * (u_old[i, j-1] + u_old[i+1, j-1] + u_old[i, j] + u_old[i+1, j])
            vc = v_old[i, j]
            
            x_prev = x - uc * dt
            y_prev = y - vc * dt
            
            v[i, j] = sample_field(v_old, x_prev, y_prev, dx, dy, 0.5, 0.0)

### ************************************************
### Advect Scalar (Standard Semi-Lagrangian)
@nb.njit(cache=True)
def advect_scalar(field, u, v, nx, ny, dx, dy, dt):
    field_new = np.zeros_like(field)
    _advect_field(field, field_new, u, v, nx, ny, dx, dy, dt)
    
    for i in range(nx):
        for j in range(ny):
            field[i, j] = field_new[i, j]

### ************************************************
### Advect Scalar (BFECC)
@nb.njit(cache=True)
def advect_scalar_bfecc(field, u, v, nx, ny, dx, dy, dt):
    # 1. Forward: phi* = Advect(phi, dt)
    phi_star = np.zeros_like(field)
    _advect_field(field, phi_star, u, v, nx, ny, dx, dy, dt)
    
    # 2. Backward: phi** = Advect(phi*, -dt)
    phi_star_star = np.zeros_like(field)
    _advect_field(phi_star, phi_star_star, u, v, nx, ny, dx, dy, -dt)
    
    # 3. Error correction: phi_mod = phi + 0.5 * (phi - phi**)
    phi_mod = np.zeros_like(field)
    for i in range(nx):
        for j in range(ny):
            # Limiter to prevent overshoots (Min-Max of neighbors?)
            # For VOF, just clamping at end is usually okay, but BFECC can overshoot.
            # Let's just apply the correction.
            phi_mod[i, j] = field[i, j] + 0.5 * (field[i, j] - phi_star_star[i, j])
            
            # Clamp phi_mod to be safe?
            # if phi_mod[i,j] < 0: phi_mod[i,j] = 0
            # if phi_mod[i,j] > 1: phi_mod[i,j] = 1
            
    # 4. Final Forward: phi_new = Advect(phi_mod, dt)
    phi_new = np.zeros_like(field)
    _advect_field(phi_mod, phi_new, u, v, nx, ny, dx, dy, dt)
    
    # Copy back and Clamp
    for i in range(nx):
        for j in range(ny):
            val = phi_new[i, j]
            if val < 0.0: val = 0.0
            if val > 1.0: val = 1.0
            field[i, j] = val

@nb.njit(cache=True)
def _advect_field(src, dst, u, v, nx, ny, dx, dy, dt):
    for i in range(nx):
        for j in range(ny):
            x = (i + 0.5) * dx
            y = (j + 0.5) * dy
            
            # Velocity at center
            uc = 0.5 * (u[i, j] + u[i+1, j])
            vc = 0.5 * (v[i, j] + v[i, j+1])
            
            x_prev = x - uc * dt
            y_prev = y - vc * dt
            
            dst[i, j] = sample_field(src, x_prev, y_prev, dx, dy, 0.5, 0.5)

### ************************************************
### Bilinear Interpolation
@nb.njit(cache=True)
def sample_field(field, x, y, dx, dy, off_x, off_y):
    # Convert to grid coords
    # Grid definition: val[i,j] is at ((i+off_x)*dx, (j+off_y)*dy)
    
    i_f = x / dx - off_x
    j_f = y / dy - off_y
    
    i = int(i_f)
    j = int(j_f)
    
    # Clamp
    nx, ny = field.shape
    if i < 0: i = 0
    if i >= nx - 1: i = nx - 2
    if j < 0: j = 0
    if j >= ny - 1: j = ny - 2
    
    # Weights
    wx = i_f - i
    wy = j_f - j
    
    # Clamp weights (in case x was far out)
    if wx < 0: wx = 0
    if wx > 1: wx = 1
    if wy < 0: wy = 0
    if wy > 1: wy = 1
    
    val = (1 - wx) * (1 - wy) * field[i, j] + \
          wx * (1 - wy) * field[i+1, j] + \
          (1 - wx) * wy * field[i, j+1] + \
          wx * wy * field[i+1, j+1]
          
    return val

### ************************************************
### Compute Divergence
@nb.njit(cache=True)
def compute_divergence(u, v, div, nx, ny, dx, dy):
    for i in range(nx):
        for j in range(ny):
            # div = du/dx + dv/dy
            du = (u[i+1, j] - u[i, j]) / dx
            dv = (v[i, j+1] - v[i, j]) / dy
            div[i, j] = du + dv

### ************************************************
### Solve Pressure (SOR)
@nb.njit(cache=True)
def solve_pressure_sor(p, rhs, nx, ny, dx, dy, max_iter, tol, omega):
    dx2 = dx*dx
    dy2 = dy*dy
    denom = 2.0 * (1.0/dx2 + 1.0/dy2)
    
    for it in range(max_iter):
        max_diff = 0.0
        for i in range(nx):
            for j in range(ny):
                # Neighbors (Neumann BCs: p[-1] = p[0])
                p_w = p[i-1, j] if i > 0 else p[i, j]
                p_e = p[i+1, j] if i < nx-1 else p[i, j]
                p_s = p[i, j-1] if j > 0 else p[i, j]
                p_n = p[i, j+1] if j < ny-1 else p[i, j]
                
                # Laplace(p) = rhs
                # (p_w - 2p + p_e)/dx2 + ... = rhs
                
                residual = rhs[i, j] - ((p_w + p_e - 2*p[i,j])/dx2 + (p_s + p_n - 2*p[i,j])/dy2)
                
                # Gauss-Seidel step
                # p_new = (p_w + p_e)/dx2 + (p_s + p_n)/dy2 - rhs
                # p_new /= denom
                
                val_gs = ((p_w + p_e)/dx2 + (p_s + p_n)/dy2 - rhs[i, j]) / denom
                
                # SOR
                p_new_val = (1 - omega) * p[i, j] + omega * val_gs
                
                diff = abs(p_new_val - p[i, j])
                if diff > max_diff:
                    max_diff = diff
                    
                p[i, j] = p_new_val
                
        if max_diff < tol:
            break

### ************************************************
### Correct Velocity
@nb.njit(cache=True)
def correct_velocity(u, v, p, nx, ny, dx, dy, dt, rho):
    scale = dt / rho
    
    # u = u - scale * dp/dx
    for i in range(1, nx):
        for j in range(ny):
            dp_dx = (p[i, j] - p[i-1, j]) / dx
            u[i, j] -= scale * dp_dx
            
    # v = v - scale * dp/dy
    for i in range(nx):
        for j in range(1, ny):
            dp_dy = (p[i, j] - p[i, j-1]) / dy
            dp_dy = (p[i, j] - p[i, j-1]) / dy
            v[i, j] -= scale * dp_dy

### ************************************************
### ************************************************
### Solve k-omega Transport
@nb.njit(cache=True)
def solve_k_omega(k, w, nu_t, u, v, nx, ny, dx, dy, dt, nu_mol, beta_star, alpha, beta, sigma_k, sigma_w):
    # 1. Advect k and omega
    # Use same advection as scalar (Standard, or BFECC if affordable)
    # Standard is more stable for turbulence quantities which must be positive
    advect_scalar(k, u, v, nx, ny, dx, dy, dt)
    advect_scalar(w, u, v, nx, ny, dx, dy, dt)
    
    # 2. Production and Dissipation
    k_new = np.zeros_like(k)
    w_new = np.zeros_like(w)
    
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            # Gradients of Velocity
            # du/dy approx at center
            du_dy = (u[i, j+1] - u[i, j-1]) / (2.0 * dy)
            dv_dx = (v[i+1, j] - v[i-1, j]) / (2.0 * dx)
            
            # Strain Rate S_ij (Simplified 2D)
            # S^2 = 2 S_ij S_ij
            # S_12 = 0.5 * (du/dy + dv/dx)
            # S^2 approx (du/dy + dv/dx)^2 for shear flow
            S2 = (du_dy + dv_dx)**2
            
            # Production Pk = nu_t * S^2
            Pk = nu_t[i, j] * S2
            
            # Dissipation = beta_star * k * w
            
            # Update k
            # dk/dt = Pk - beta_star * k * w + diff
            # Diffusion: div( (nu + sigma_k*nu_t) grad k )
            
            nu_eff_k = nu_mol + sigma_k * nu_t[i, j]
            
            d2k_dx2 = (k[i+1, j] - 2*k[i, j] + k[i-1, j]) / dx**2
            d2k_dy2 = (k[i, j+1] - 2*k[i, j] + k[i, j-1]) / dy**2
            
            k_new[i, j] = k[i, j] + dt * (Pk - beta_star * k[i, j] * w[i, j] + nu_eff_k * (d2k_dx2 + d2k_dy2))
            
            # Update omega
            # dw/dt = alpha * w/k * Pk - beta * w^2 + diff
            # Production term often written as alpha * S^2
            
            Pw = alpha * S2
            
            nu_eff_w = nu_mol + sigma_w * nu_t[i, j]
            
            d2w_dx2 = (w[i+1, j] - 2*w[i, j] + w[i-1, j]) / dx**2
            d2w_dy2 = (w[i, j+1] - 2*w[i, j] + w[i, j-1]) / dy**2
            
            w_new[i, j] = w[i, j] + dt * (Pw - beta * w[i, j]**2 + nu_eff_w * (d2w_dx2 + d2w_dy2))
            
            # Clip
            if k_new[i, j] < 1e-8: k_new[i, j] = 1e-8
            if w_new[i, j] < 1e-8: w_new[i, j] = 1e-8
            
    # Update arrays and compute nu_t
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            k[i, j] = k_new[i, j]
            w[i, j] = w_new[i, j]
            nu_t[i, j] = k[i, j] / w[i, j]

### ************************************************
### Diffuse Velocity (Explicit)
@nb.njit(cache=True)
def diffuse_velocity(u, v, nu_t, nu_mol, nx, ny, dx, dy, dt):
    # Explicit diffusion: u += dt * div(nu * grad u)
    # Simplified: u += dt * (nu_eff * d2u/dx2 + nu_eff * d2u/dy2)
    # Note: nu_eff varies spatially.
    
    # For u (i+0.5, j)
    u_new = np.zeros_like(u)
    for i in range(1, nx):
        for j in range(1, ny-1):
            # Viscosity at u location (average from centers)
            # nu_t is at centers i,j. u is at i,j (staggered index i means i+0.5)
            # u[i,j] is between center i and i+1? No.
            # Grid: u[i,j] is face between cell i and i+1? No, standard MAC:
            # u[i,j] is right face of cell i? Or left face?
            # Usually u[i,j] is face i-1/2.
            # Let's assume u[i,j] is face between i-1 and i.
            
            # nu at face i,j
            nu_c = 0.5 * (nu_t[i-1, j] + nu_t[i, j]) + nu_mol
            
            # Laplacian
            d2u_dx2 = (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2
            d2u_dy2 = (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2
            
            u_new[i, j] = u[i, j] + dt * nu_c * (d2u_dx2 + d2u_dy2)
            
    # For v (i, j+0.5)
    v_new = np.zeros_like(v)
    for i in range(1, nx-1):
        for j in range(1, ny):
            # nu at face i,j (horizontal)
            nu_c = 0.5 * (nu_t[i, j-1] + nu_t[i, j]) + nu_mol
            
            d2v_dx2 = (v[i+1, j] - 2*v[i, j] + v[i-1, j]) / dx**2
            d2v_dy2 = (v[i, j+1] - 2*v[i, j] + v[i, j-1]) / dy**2
            
            v_new[i, j] = v[i, j] + dt * nu_c * (d2v_dx2 + d2v_dy2)
            
    # Update (excluding boundaries for now)
    for i in range(1, nx):
        for j in range(1, ny-1):
            u[i, j] = u_new[i, j]
            
    for i in range(1, nx-1):
        for j in range(1, ny):
            v[i, j] = v_new[i, j]
