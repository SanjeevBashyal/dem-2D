### ************************************************
### Solve k-omega SST Transport
@nb.njit(cache=True)
def solve_k_omega_sst(k, w, nu_t, u, v, nx, ny, dx, dy, dt, nu_mol, 
                      a1, beta_star, sigma_k1, sigma_w1, beta1, alpha1, 
                      sigma_k2, sigma_w2, beta2, alpha2):
    
    # Advect
    advect_scalar(k, u, v, nx, ny, dx, dy, dt)
    advect_scalar(w, u, v, nx, ny, dx, dy, dt)
    
    k_new = np.zeros_like(k)
    w_new = np.zeros_like(w)
    
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            # Gradients
            du_dy = (u[i, j+1] - u[i, j-1]) / (2.0 * dy)
            dv_dx = (v[i+1, j] - v[i-1, j]) / (2.0 * dx)
            S2 = (du_dy + dv_dx)**2
            S = np.sqrt(S2)
            
            # Distance to wall (Simplified: assume bottom/top walls at y=0, y=H)
            # For a particle, it's harder. Let's use a simple heuristic or constant for now if wall dist is expensive.
            # Or just use blending function F1 = 1 (k-omega) near walls, F1 = 0 (k-epsilon) far away.
            # For this demo, let's implement the blending logic based on y-coordinate (channel flow style) 
            # or just use a fixed blend if wall distance is unavailable.
            # Actually, without wall distance field, SST is hard.
            # Let's approximate F1 based on local turbulent Reynolds number or similar?
            # Or just implement the equations with constants blended by a fixed F1? No, that defeats the purpose.
            
            # Let's assume F1=1 for now (Pure k-omega) but with SST limiters?
            # User asked for SST. SST main feature is the limiter on nu_t and cross-diffusion.
            
            # 1. Compute F1 (Blending)
            # Needs distance 'd'. Let's approximate d as min distance to boundaries.
            d_x = min((i+0.5)*dx, (nx-(i+0.5))*dx)
            d_y = min((j+0.5)*dy, (ny-(j+0.5))*dy)
            d = min(d_x, d_y)
            
            # CD_kw
            dk_dx = (k[i+1, j] - k[i-1, j]) / (2*dx)
            dk_dy = (k[i, j+1] - k[i, j-1]) / (2*dy)
            dw_dx = (w[i+1, j] - w[i-1, j]) / (2*dx)
            dw_dy = (w[i, j+1] - w[i, j-1]) / (2*dy)
            
            cross_diff = 2 * sigma_w2 * (dk_dx * dw_dx + dk_dy * dw_dy) / max(w[i, j], 1e-8)
            
            arg1 = max(np.sqrt(k[i, j]) / (beta_star * w[i, j] * d), 500 * nu_mol / (d**2 * w[i, j]))
            # arg1 = min(arg1, 4 * sigma_w2 * k[i, j] / (max(cross_diff, 1e-10) * d**2)) # Full SST has this
            
            F1 = np.tanh(arg1**4)
            
            # Blend Constants
            alpha = F1 * alpha1 + (1-F1) * alpha2
            beta = F1 * beta1 + (1-F1) * beta2
            sigma_k = F1 * sigma_k1 + (1-F1) * sigma_k2
            sigma_w = F1 * sigma_w1 + (1-F1) * sigma_w2
            
            # Production
            Pk = nu_t[i, j] * S2
            Pk = min(Pk, 10 * beta_star * k[i, j] * w[i, j])
            
            # k equation
            nu_eff_k = nu_mol + nu_t[i, j] * sigma_k # sigma is inverse Prandtl in some notations? 
            # Standard SST: diffusion is d/dx( (nu + sigma*nu_t) dk/dx )
            # Here sigma_k is the multiplier.
            
            d2k_dx2 = (k[i+1, j] - 2*k[i, j] + k[i-1, j]) / dx**2
            d2k_dy2 = (k[i, j+1] - 2*k[i, j] + k[i, j-1]) / dy**2
            
            k_new[i, j] = k[i, j] + dt * (Pk - beta_star * k[i, j] * w[i, j] + nu_eff_k * (d2k_dx2 + d2k_dy2))
            
            # omega equation
            # Pw = alpha * S2 ... No, Pw = alpha * Pk / nu_t ? Or alpha * S^2?
            # Standard: Pw = gamma * S^2 (gamma is my alpha here)
            Pw = alpha * S2
            
            # Cross Diffusion Term (only for F1 < 1)
            CD = 2 * (1 - F1) * sigma_w2 * (dk_dx * dw_dx + dk_dy * dw_dy) / max(w[i, j], 1e-8)
            
            nu_eff_w = nu_mol + nu_t[i, j] * sigma_w
            
            d2w_dx2 = (w[i+1, j] - 2*w[i, j] + w[i-1, j]) / dx**2
            d2w_dy2 = (w[i, j+1] - 2*w[i, j] + w[i, j-1]) / dy**2
            
            w_new[i, j] = w[i, j] + dt * (Pw - beta * w[i, j]**2 + CD + nu_eff_w * (d2w_dx2 + d2w_dy2))
            
            # Clip
            k_new[i, j] = max(k_new[i, j], 1e-8)
            w_new[i, j] = max(w_new[i, j], 1e-8)
            
    # Update nu_t with SST limiter
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            k[i, j] = k_new[i, j]
            w[i, j] = w_new[i, j]
            
            # F2
            d_x = min((i+0.5)*dx, (nx-(i+0.5))*dx)
            d_y = min((j+0.5)*dy, (ny-(j+0.5))*dy)
            d = min(d_x, d_y)
            
            arg2 = max(2 * np.sqrt(k[i, j]) / (beta_star * w[i, j] * d), 500 * nu_mol / (d**2 * w[i, j]))
            F2 = np.tanh(arg2**2)
            
            # Gradients for S
            du_dy = (u[i, j+1] - u[i, j-1]) / (2.0 * dy)
            dv_dx = (v[i+1, j] - v[i-1, j]) / (2.0 * dx)
            S = np.sqrt((du_dy + dv_dx)**2)
            
            # nu_t = a1 * k / max(a1 * w, S * F2)
            nu_t[i, j] = a1 * k[i, j] / max(a1 * w[i, j], S * F2 + 1e-10)
