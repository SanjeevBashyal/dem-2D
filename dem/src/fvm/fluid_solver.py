import numpy as np
import numba as nb

class FVMSolver:
    def __init__(self, grid):
        self.grid = grid
        self.rho_water = 1000.0
        self.rho_air   = 1.0
        self.mu_water  = 0.001
        self.mu_air    = 0.00001
        self.g         = np.array([0.0, 9.81]) # Vector Gravity [gx, gy]
        
        self.variable_density = False # Default to False, enable for VOF
        
        # Fixed Pressure BCs
        self.p_fixed_mask = None # Boolean mask (True where pressure is fixed)
        self.p_fixed_val  = None # Values for fixed pressure
        
        # Turbulence Parameters
        # Turbulence Parameters (k-omega)
        self.k = None     # Turbulent Kinetic Energy
        self.omega_t = None # Specific Dissipation Rate
        # self.nu_t is now in grid
        
        # k-omega Constants (Standard Wilcox 2006 or similar)
        self.beta_star_kw = 0.09
        self.alpha_kw  = 0.52
        self.beta_kw   = 0.072
        self.sigma_k   = 0.5
        self.sigma_w   = 0.5
        
        # k-epsilon Constants (Standard)
        self.C_mu = 0.09
        self.sigma_k_eps = 1.0
        self.sigma_eps = 1.3
        self.C1_eps = 1.44
        self.C2_eps = 1.92
        
        # k-omega SST Constants (Menter 2003)
        self.a1 = 0.31
        self.beta_star = 0.09
        
        # Set 1 (Inner)
        self.sigma_k1 = 0.85
        self.sigma_w1 = 0.5
        self.beta1    = 0.075
        self.alpha1   = 5.0/9.0 
        
        # Set 2 (Outer)
        self.sigma_k2 = 1.0
        self.sigma_w2 = 0.856
        self.beta2    = 0.0828
        self.alpha2   = 0.44
        
        self.turbulence_model = 'k-omega-sst' # 'k-omega', 'k-epsilon', 'k-omega-sst'
        
        # SOR Parameters
        self.max_iter = 100
        self.tol      = 1e-4
        self.omega    = 1.7

    def solve(self, dt):
        # 1. Advect Velocity (Semi-Lagrangian)
        advect_velocity(self.grid.u, self.grid.v, self.grid.u, self.grid.v, 
                        self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy, dt)
        
        # 1.5 Diffusion (Turbulence)
        if self.turbulence_model == 'k-omega':
            if self.k is None:
                self.k = np.zeros((self.grid.nx, self.grid.ny)) + 1e-6
                self.omega_t = np.zeros((self.grid.nx, self.grid.ny)) + 100.0 # High initial dissipation
                
            solve_k_omega(self.k, self.omega_t, self.grid.nu_t, self.grid.u, self.grid.v, 
                          self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy, dt,
                          self.mu_water/self.rho_water, self.beta_star_kw, self.alpha_kw, self.beta_kw, self.sigma_k, self.sigma_w)
                          
        elif self.turbulence_model == 'k-epsilon':
            if self.k is None:
                self.k = np.zeros((self.grid.nx, self.grid.ny)) + 1e-6
                self.omega_t = np.zeros((self.grid.nx, self.grid.ny)) + 1e-6 # Actually epsilon here
                
            solve_k_epsilon(self.k, self.omega_t, self.grid.nu_t, self.grid.u, self.grid.v,
                            self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy, dt,
                            self.mu_water/self.rho_water, self.C_mu, self.sigma_k_eps, self.sigma_eps, self.C1_eps, self.C2_eps)
                            
        elif self.turbulence_model == 'k-omega-sst':
            if self.k is None:
                self.k = np.zeros((self.grid.nx, self.grid.ny)) + 1e-6
                self.omega_t = np.zeros((self.grid.nx, self.grid.ny)) + 100.0
                
            solve_k_omega_sst(self.k, self.omega_t, self.grid.nu_t, self.grid.u, self.grid.v,
                              self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy, dt,
                              self.mu_water/self.rho_water, 
                              self.a1, self.beta_star, 
                              self.sigma_k1, self.sigma_w1, self.beta1, self.alpha1,
                              self.sigma_k2, self.sigma_w2, self.beta2, self.alpha2)
        
        # Diffuse Velocity
        diffuse_velocity(self.grid.u, self.grid.v, self.grid.nu_t, self.mu_water/self.rho_water, 
                         self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy, dt)
        
        # 2. Apply Gravity
        # Simple explicit, support vector gravity
        if isinstance(self.g, (float, int)):
             # Scalar, assume vertical down
             self.grid.v[:, 1:-1] -= self.g * dt
        else:
             # Vector
             self.grid.u[1:-1, :] += self.g[0] * dt
             self.grid.v[:, 1:-1] += self.g[1] * dt
        
        # 5. Advect VOF (Alpha) using BFECC for sharpness
        # Done before pressure solve to get correct density
        advect_scalar_bfecc(self.grid.alpha, self.grid.u, self.grid.v, 
                            self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy, dt)
                            
        # Compute Density Field
        # rho = alpha * rho_water + (1-alpha) * rho_air
        rho_field = self.grid.alpha * self.rho_water + (1.0 - self.grid.alpha) * self.rho_air
        
        # 3. Pressure Solve (Projection)
        # Compute Divergence
        compute_divergence(self.grid.u, self.grid.v, self.grid.div, 
                           self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy)
        
        # Solve Poisson
        
        # Prepare Fixed Pressure Mask if not set
        if self.p_fixed_mask is None:
            self.p_fixed_mask = np.zeros((self.grid.nx, self.grid.ny), dtype=bool)
            self.p_fixed_val = np.zeros((self.grid.nx, self.grid.ny))
            
        if self.variable_density:
            # Variable Density: div(1/rho grad p) = div/dt
            rhs = self.grid.div / dt
            solve_pressure_sor_variable_rho(self.grid.p, rhs, rho_field, 
                                            self.p_fixed_mask, self.p_fixed_val,
                                            self.grid.nx, self.grid.ny, 
                                            self.grid.dx, self.grid.dy, self.max_iter, self.tol, self.omega)
            
            # 4. Correct Velocity
            correct_velocity_variable_rho(self.grid.u, self.grid.v, self.grid.p, rho_field,
                                          self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy, dt)
        else:
            # Constant Density (Water)
            rho = self.rho_water 
            rhs = self.grid.div * rho / dt
            
            solve_pressure_sor(self.grid.p, rhs, 
                               self.p_fixed_mask, self.p_fixed_val,
                               self.grid.nx, self.grid.ny, 
                               self.grid.dx, self.grid.dy, self.max_iter, self.tol, self.omega)
            
            # 4. Correct Velocity
            correct_velocity(self.grid.u, self.grid.v, self.grid.p, 
                             self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy, dt, rho)

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
def solve_pressure_sor(p, rhs, p_fixed_mask, p_fixed_val, nx, ny, dx, dy, max_iter, tol, omega):
    dx2 = dx*dx
    dy2 = dy*dy
    denom = 2.0 * (1.0/dx2 + 1.0/dy2)
    
    for it in range(max_iter):
        max_diff = 0.0
        for i in range(nx):
            for j in range(ny):
                # Check fixed pressure
                if p_fixed_mask[i, j]:
                    p[i, j] = p_fixed_val[i, j]
                    continue
                    
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
### Solve Pressure (SOR) - Variable Density
@nb.njit(cache=True)
def solve_pressure_sor_variable_rho(p, rhs, rho, p_fixed_mask, p_fixed_val, nx, ny, dx, dy, max_iter, tol, omega):
    dx2 = dx*dx
    dy2 = dy*dy
    
    for it in range(max_iter):
        max_diff = 0.0
        for i in range(nx):
            for j in range(ny):
                # Check fixed pressure
                if p_fixed_mask[i, j]:
                    p[i, j] = p_fixed_val[i, j]
                    continue
                    
                # Neighbors (Neumann BCs)
                p_w = p[i-1, j] if i > 0 else p[i, j]
                p_e = p[i+1, j] if i < nx-1 else p[i, j]
                p_s = p[i, j-1] if j > 0 else p[i, j]
                p_n = p[i, j+1] if j < ny-1 else p[i, j]
                
                # Density at faces (Harmonic mean is often better for transport, arithmetic for diffusion?)
                # Arithmetic mean for now:
                rho_c = rho[i, j]
                rho_w = 0.5 * (rho[i-1, j] + rho_c) if i > 0 else rho_c
                rho_e = 0.5 * (rho[i+1, j] + rho_c) if i < nx-1 else rho_c
                rho_s = 0.5 * (rho[i, j-1] + rho_c) if j > 0 else rho_c
                rho_n = 0.5 * (rho[i, j+1] + rho_c) if j < ny-1 else rho_c
                
                # Coeffs: 1/rho
                kw = 1.0 / rho_w
                ke = 1.0 / rho_e
                ks = 1.0 / rho_s
                kn = 1.0 / rho_n
                
                # Discretization:
                # (ke*(pe-p) - kw*(p-pw))/dx2 + ... = rhs
                # p * (-(ke+kw)/dx2 - (kn+ks)/dy2) + ke*pe/dx2 + ... = rhs
                
                coeff_p = (ke + kw)/dx2 + (kn + ks)/dy2
                sigma = (ke*p_e + kw*p_w)/dx2 + (kn*p_n + ks*p_s)/dy2
                
                # p_new = (sigma - rhs) / coeff_p
                val_gs = (sigma - rhs[i, j]) / coeff_p
                
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
### Correct Velocity - Variable Density
@nb.njit(cache=True)
def correct_velocity_variable_rho(u, v, p, rho, nx, ny, dx, dy, dt):
    # u = u - dt/rho * dp/dx
    for i in range(1, nx):
        for j in range(ny):
            dp_dx = (p[i, j] - p[i-1, j]) / dx
            
            # rho at u face
            rho_face = 0.5 * (rho[i, j] + rho[i-1, j])
            
            u[i, j] -= (dt / rho_face) * dp_dx
            
    # v = v - dt/rho * dp/dy
    for i in range(nx):
        for j in range(1, ny):
            dp_dy = (p[i, j] - p[i, j-1]) / dy
            
            # rho at v face
            rho_face = 0.5 * (rho[i, j] + rho[i, j-1])
            
            v[i, j] -= (dt / rho_face) * dp_dy

### ************************************************
### Solve k-omega Transport
@nb.njit(cache=True)
def solve_k_omega(k, w, nu_t, u, v, nx, ny, dx, dy, dt, nu_mol, beta_star, alpha, beta, sigma_k, sigma_w):
    # 1. Advect k and omega
    # Use same advection as scalar (Standard, or BFECC if affordable)
    # Standard is more stable for turbulence quantities which must be positive
    advect_scalar(k, u, v, nx, ny, dx, dy, dt)
    advect_scalar(w, u, v, nx, ny, dx, dy, dt)
    
    # Enforce positivity after advection
    for i in range(nx):
        for j in range(ny):
            if k[i, j] < 1e-8: k[i, j] = 1e-8
            if w[i, j] < 1e-8: w[i, j] = 1e-8
    
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
            # Limiter
            Pk = min(nu_t[i, j] * S2, 10.0 * beta_star * k[i, j] * w[i, j])
            
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
            
            # Clip
            if k_new[i, j] < 1e-8: k_new[i, j] = 1e-8
            if w_new[i, j] < 1e-8: w_new[i, j] = 1e-8
            
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

### ************************************************
### Solve k-epsilon Transport
@nb.njit(cache=True)
def solve_k_epsilon(k, eps, nu_t, u, v, nx, ny, dx, dy, dt, nu_mol, C_mu, sigma_k, sigma_eps, C1_eps, C2_eps):
    # 1. Advect k and epsilon
    advect_scalar(k, u, v, nx, ny, dx, dy, dt)
    advect_scalar(eps, u, v, nx, ny, dx, dy, dt)
    
    # 2. Production and Dissipation
    k_new = np.zeros_like(k)
    eps_new = np.zeros_like(eps)
    
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            # Gradients of Velocity
            du_dy = (u[i, j+1] - u[i, j-1]) / (2.0 * dy)
            dv_dx = (v[i+1, j] - v[i-1, j]) / (2.0 * dx)
            
            S2 = (du_dy + dv_dx)**2
            
            # Production Pk = nu_t * S^2
            Pk = nu_t[i, j] * S2
            
            # Update k
            # dk/dt = Pk - epsilon + diff
            nu_eff_k = nu_mol + nu_t[i, j] / sigma_k
            
            d2k_dx2 = (k[i+1, j] - 2*k[i, j] + k[i-1, j]) / dx**2
            d2k_dy2 = (k[i, j+1] - 2*k[i, j] + k[i, j-1]) / dy**2
            
            k_new[i, j] = k[i, j] + dt * (Pk - eps[i, j] + nu_eff_k * (d2k_dx2 + d2k_dy2))
            
            # Update epsilon
            # deps/dt = C1 * eps/k * Pk - C2 * eps^2/k + diff
            # Avoid division by zero
            k_val = max(k[i, j], 1e-8)
            
            nu_eff_eps = nu_mol + nu_t[i, j] / sigma_eps
            
            d2eps_dx2 = (eps[i+1, j] - 2*eps[i, j] + eps[i-1, j]) / dx**2
            d2eps_dy2 = (eps[i, j+1] - 2*eps[i, j] + eps[i, j-1]) / dy**2
            
            source_eps = (C1_eps * Pk - C2_eps * eps[i, j]) * eps[i, j] / k_val
            
            eps_new[i, j] = eps[i, j] + dt * (source_eps + nu_eff_eps * (d2eps_dx2 + d2eps_dy2))
            
            # Clip
            if k_new[i, j] < 1e-8: k_new[i, j] = 1e-8
            if eps_new[i, j] < 1e-8: eps_new[i, j] = 1e-8
            
    # Update arrays and compute nu_t
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            k[i, j] = k_new[i, j]
            eps[i, j] = eps_new[i, j]
            nu_t[i, j] = C_mu * k[i, j]**2 / eps[i, j]

### ************************************************
### Solve k-omega SST Transport
@nb.njit(cache=True)
def solve_k_omega_sst(k, w, nu_t, u, v, nx, ny, dx, dy, dt, nu_mol, a1, beta_star, sigma_k1, sigma_w1, beta1, alpha1, sigma_k2, sigma_w2, beta2, alpha2):
    # 1. Advect k and omega
    advect_scalar(k, u, v, nx, ny, dx, dy, dt)
    advect_scalar(w, u, v, nx, ny, dx, dy, dt)
    
    # 2. Production and Dissipation
    k_new = np.zeros_like(k)
    w_new = np.zeros_like(w)
    
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            # Blending Function F1
            # Need distance to wall d. For now assume simple bottom wall d = y
            d = (j + 0.5) * dy
            
            k_val = max(k[i, j], 1e-8)
            w_val = max(w[i, j], 1e-8)
            
            # arg1 = min(max(sqrt(k)/(beta_star*w*d), 500*nu/(d^2*w)), 4*sigma_w2*k/(CDkw*d^2))
            # Simplified for now:
            arg1 = max(np.sqrt(k_val)/(beta_star*w_val*d), 500.0*nu_mol/(d**2*w_val))
            F1 = np.tanh(arg1**4)
            
            # Blend constants
            sigma_k = F1 * sigma_k1 + (1-F1) * sigma_k2
            sigma_w = F1 * sigma_w1 + (1-F1) * sigma_w2
            beta    = F1 * beta1    + (1-F1) * beta2
            alpha   = F1 * alpha1   + (1-F1) * alpha2
            
            # Gradients
            du_dy = (u[i, j+1] - u[i, j-1]) / (2.0 * dy)
            dv_dx = (v[i+1, j] - v[i-1, j]) / (2.0 * dx)
            S2 = (du_dy + dv_dx)**2
            
            # Production Pk
            # Limiter: Pk = min(nu_t * S2, 10 * beta_star * k * w)
            Pk = min(nu_t[i, j] * S2, 10.0 * beta_star * k_val * w_val)
            
            # Update k
            nu_eff_k = nu_mol + sigma_k * nu_t[i, j]
            d2k_dx2 = (k[i+1, j] - 2*k[i, j] + k[i-1, j]) / dx**2
            d2k_dy2 = (k[i, j+1] - 2*k[i, j] + k[i, j-1]) / dy**2
            
            k_new[i, j] = k[i, j] + dt * (Pk - beta_star * k_val * w_val + nu_eff_k * (d2k_dx2 + d2k_dy2))
            
            # Update omega
            # Cross diffusion CDkw
            # grad k dot grad w
            dk_dx = (k[i+1, j] - k[i-1, j]) / (2*dx)
            dk_dy = (k[i, j+1] - k[i, j-1]) / (2*dy)
            dw_dx = (w[i+1, j] - w[i-1, j]) / (2*dx)
            dw_dy = (w[i, j+1] - w[i, j-1]) / (2*dy)
            
            CDkw = 2.0 * sigma_w2 * (dk_dx*dw_dx + dk_dy*dw_dy) / w_val
            
            # F1 blending for source
            # Pw = alpha * S2 (approx)
            # dw/dt = alpha * S2 - beta * w^2 + diff + 2(1-F1)sigma_w2/w * grad k * grad w
            
            nu_eff_w = nu_mol + sigma_w * nu_t[i, j]
            d2w_dx2 = (w[i+1, j] - 2*w[i, j] + w[i-1, j]) / dx**2
            d2w_dy2 = (w[i, j+1] - 2*w[i, j] + w[i, j-1]) / dy**2
            
            cross_diff = 2.0 * (1.0 - F1) * sigma_w2 * (dk_dx*dw_dx + dk_dy*dw_dy) / w_val
            
            w_new[i, j] = w[i, j] + dt * (alpha * S2 - beta * w_val**2 + nu_eff_w * (d2w_dx2 + d2w_dy2) + cross_diff)
            
            # Clip
            if k_new[i, j] < 1e-8: k_new[i, j] = 1e-8
            if w_new[i, j] < 1e-8: w_new[i, j] = 1e-8
            
    # Compute nu_t (SST Limiter)
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            k[i, j] = k_new[i, j]
            w[i, j] = w_new[i, j]
            
            # F2 blending
            d = (j + 0.5) * dy
            arg2 = max(2.0*np.sqrt(k[i, j])/(beta_star*w[i, j]*d), 500.0*nu_mol/(d**2*w[i, j]))
            F2 = np.tanh(arg2**2)
            
            # Gradients for S
            du_dy = (u[i, j+1] - u[i, j-1]) / (2.0 * dy)
            dv_dx = (v[i+1, j] - v[i-1, j]) / (2.0 * dx)
            S = np.sqrt((du_dy + dv_dx)**2)
            
            # nu_t = a1 * k / max(a1 * w, S * F2)
            nu_t[i, j] = a1 * k[i, j] / max(a1 * w[i, j], S * F2 + 1e-10)