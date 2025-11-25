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
            # nu_t = C_mu * k^2 / epsilon
            nu_t[i, j] = C_mu * k[i, j]**2 / eps[i, j]
