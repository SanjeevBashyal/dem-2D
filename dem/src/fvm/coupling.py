import numpy as np
import numba as nb

### ************************************************
### Apply IBM Forcing (Solid -> Fluid)
### Forces fluid velocity to match solid velocity in solid regions
def apply_ibm_forcing(grid):
    apply_ibm_forcing_numba(grid.u, grid.v, 
                            grid.eps_s, grid.u_solid, grid.v_solid, 
                            grid.nx, grid.ny)

@nb.njit(cache=True)
def apply_ibm_forcing_numba(u, v, eps_s, u_solid, v_solid, nx, ny):
    # For u (vertical faces)
    for i in range(1, nx):
        for j in range(ny):
            # Interpolate eps_s to face
            eps = 0.5 * (eps_s[i-1, j] + eps_s[i, j])
            
            if eps > 0:
                # Interpolate solid velocity to face
                us = 0.5 * (u_solid[i-1, j] + u_solid[i, j])
                
                # Force velocity: u = (1-eps)*u + eps*us
                u[i, j] = (1.0 - eps) * u[i, j] + eps * us

    # For v (horizontal faces)
    for i in range(nx):
        for j in range(1, ny):
            # Interpolate eps_s to face
            eps = 0.5 * (eps_s[i, j-1] + eps_s[i, j])
            
            if eps > 0:
                # Interpolate solid velocity to face
                vs = 0.5 * (v_solid[i, j-1] + v_solid[i, j])
                
                # Force velocity
                v[i, j] = (1.0 - eps) * v[i, j] + eps * vs

### ************************************************
### Compute Hydrodynamic Forces (Fluid -> Solid)
### Integrates pressure and viscous stress over polygon surface
def compute_hydrodynamic_forces(grid, polygons):
    # This is complex for IBM.
    # Simplified approach:
    # F = sum( (u_fluid - u_solid) / dt * mass_fluid_displaced ) ?
    # Or integrate pressure gradient?
    
    # Let's use a volume integration approach over the solid cells.
    # F_hydro = sum over cells ( -grad P + div tau ) * Volume
    # But we only have P and u.
    
    # F_buoyancy = sum ( -grad P ) * Volume
    # F_drag = sum ( viscosity * laplacian u ) * Volume
    
    # Let's approximate by summing forces on grid cells covered by polygon.
    
    compute_forces_numba(grid.p, grid.u, grid.v, 
                         grid.nx, grid.ny, grid.dx, grid.dy,
                         polygons.x, polygons.f, polygons.tau,
                         polygons.vertices_world)
                         
    # Add Explicit Drag Force (Stokes/Di Felice) & Lift
    # Need fluid velocity at particle position
    # Use ambient velocity (average of neighbors) to avoid self-shielding
    compute_drag_lift_numba(grid.u, grid.v, grid.nu_t, grid.eps_s,
                       grid.nx, grid.ny, grid.dx, grid.dy,
                       polygons.x, polygons.v, polygons.f, 
                       polygons.vertices_world)

@nb.njit(cache=True)
def compute_forces_numba(p, u, v, nx, ny, dx, dy, 
                         poly_pos, poly_f, poly_tau, poly_verts_list):
    
    # Iterate over grid to find forces, then distribute to polygons?
    # Or iterate polygons and sum grid forces?
    # Iterating polygons is better if they are small.
    
    n_polys = len(poly_pos)
    
    for k in range(n_polys):
        verts = poly_verts_list[k]
        cx = poly_pos[k, 0]
        cy = poly_pos[k, 1]
        
        # AABB
        min_x = np.min(verts[:, 0])
        max_x = np.max(verts[:, 0])
        min_y = np.min(verts[:, 1])
        max_y = np.max(verts[:, 1])
        
        i_min = int(max(0, min_x / dx))
        i_max = int(min(nx, max_x / dx + 1))
        j_min = int(max(0, min_y / dy))
        j_max = int(min(ny, max_y / dy + 1))
        
        fx_total = 0.0
        fy_total = 0.0
        torque_total = 0.0
        
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                xc = (i + 0.5) * dx
                yc = (j + 0.5) * dy
                
                # Check if inside (simple check)
                # For better accuracy, use fractional area
                if point_in_polygon_coupling(xc, yc, verts):
                    # Force from pressure gradient
                    # F = -grad P * Volume
                    # grad P at center:
                    # dp/dx = (p[i+1] - p[i-1]) / 2dx
                    
                    # Handle boundaries for gradient
                    p_e = p[i+1, j] if i < nx-1 else p[i, j]
                    p_w = p[i-1, j] if i > 0 else p[i, j]
                    p_n = p[i, j+1] if j < ny-1 else p[i, j]
                    p_s = p[i, j-1] if j > 0 else p[i, j]
                    
                    dp_dx = (p_e - p_w) / (2*dx)
                    dp_dy = (p_n - p_s) / (2*dy)
                    
                    vol = dx * dy
                    
                    dFx = -dp_dx * vol
                    dFy = -dp_dy * vol
                    
                    fx_total += dFx
                    fy_total += dFy
                    
                    # Torque
                    rx = xc - cx
                    ry = yc - cy
                    
                    torque_total += rx * dFy - ry * dFx
                    
        poly_f[k, 0] += fx_total
        poly_f[k, 1] += fy_total
        poly_tau[k] += torque_total

@nb.njit(cache=True)
def point_in_polygon_coupling(x, y, verts):
    # Duplicate of grid.py function to avoid circular import or just copy
    n = len(verts)
    inside = False
    p1x, p1y = verts[0]
    for i in range(n + 1):
        p2x, p2y = verts[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

### ************************************************
### Compute Drag & Lift Force
@nb.njit(cache=True)
def compute_drag_lift_numba(u, v, nu_t, eps_s, nx, ny, dx, dy, 
                       poly_pos, poly_vel, poly_f, poly_verts_list):
    
    rho = 1000.0
    mu = 0.001
    
    n_polys = len(poly_pos)
    
    for k in range(n_polys):
        cx = poly_pos[k, 0]
        cy = poly_pos[k, 1]
        vx = poly_vel[k, 0]
        vy = poly_vel[k, 1]
        
        # Estimate Ambient Velocity (average of surrounding fluid cells)
        # Scan a box around center
        # Box size: ~ 2 * Diameter
        # Diameter approx 0.05. dx=0.02. So +/- 3 cells.
        
        i_c = int(cx / dx)
        j_c = int(cy / dy)
        
        u_amb = 0.0
        v_amb = 0.0
        count = 0
        
        du_dy_sum = 0.0 # For shear estimate
        
        range_cells = 3
        
        for i in range(i_c - range_cells, i_c + range_cells + 1):
            for j in range(j_c - range_cells, j_c + range_cells + 1):
                if i >= 0 and i < nx and j >= 0 and j < ny:
                    # Check if fluid cell (eps_s < 0.1)
                    if eps_s[i, j] < 0.1:
                        # Sample u, v at center
                        uc = 0.5 * (u[i, j] + u[i+1, j])
                        vc = 0.5 * (v[i, j] + v[i, j+1])
                        
                        u_amb += uc
                        v_amb += vc
                        
                        # Estimate shear du/dy
                        # (u[i, j+1] - u[i, j-1]) / 2dy
                        if j > 0 and j < ny-1:
                             du = (u[i, j+1] + u[i+1, j+1]) * 0.5 - (u[i, j-1] + u[i+1, j-1]) * 0.5
                             du_dy_sum += abs(du / (2*dy))
                        
                        count += 1
        
        if count > 0:
            u_amb /= count
            v_amb /= count
            du_dy_sum /= count
        else:
            # Fallback to local if no fluid neighbors (buried?)
            u_amb = sample_u(u, cx, cy, dx, dy)
            v_amb = sample_v(v, cx, cy, dx, dy)
            
        # Relative velocity
        ur_x = u_amb - vx
        ur_y = v_amb - vy
        ur_mag = np.sqrt(ur_x**2 + ur_y**2)
        
        # Diameter (approx)
        verts = poly_verts_list[k]
        min_x = np.min(verts[:, 0])
        max_x = np.max(verts[:, 0])
        D = max_x - min_x
        
        if ur_mag > 1e-6:
            # Reynolds number
            nu_eddy = 0.0
            # Sample nu_t at center
            if i_c >=0 and i_c < nx and j_c >=0 and j_c < ny:
                if nu_t is not None:
                    nu_eddy = nu_t[i_c, j_c]
                
            nu_eff = mu/rho + nu_eddy
            
            Re = ur_mag * D / nu_eff
            
            # Drag Coefficient (Schiller-Naumann)
            Cd = 24.0 / (Re + 1e-10) * (1.0 + 0.15 * Re**0.687)
            if Re > 1000:
                Cd = 0.44
                
            # Drag Force
            A = D # 2D Area
            Fx_d = 0.5 * rho * Cd * A * ur_mag * ur_x
            Fy_d = 0.5 * rho * Cd * A * ur_mag * ur_y
            
            poly_f[k, 0] += Fx_d
            poly_f[k, 1] += Fy_d
            
            # Saffman Lift Force
            # F_L = 1.61 * mu * D^2 * |u_amb - v| * sqrt(shear / nu) / nu ?
            # Standard form: F_L = 1.61 * D^2 * sqrt(rho * mu * |du/dy|) * (u_amb - v)
            # Note: Saffman is for low Re. For higher Re, it's complex.
            # Let's use a simplified Lift coefficient C_L ~ 0.1 - 0.5
            # F_L = 0.5 * rho * C_L * A * ur^2
            # Direction is perpendicular to relative velocity?
            # Or perpendicular to flow direction?
            # Lift acts upwards if shear is positive (velocity increases upwards).
            
            # Let's use Saffman-Mei expression or simple C_L.
            # Simple: Lift due to shear.
            # F_L = C_L * rho * D * (u_amb - vx)^2 * sign(du/dy)
            # But Saffman scales with sqrt(shear).
            
            # F_L = 1.61 * D**2 * sqrt(rho * mu * du_dy_sum) * ur_x
            # Check units: m^2 * sqrt(kg/m3 * kg/ms * 1/s) * m/s
            # = m^2 * sqrt(kg^2/m4s2) * m/s = m^2 * kg/m2s * m/s = kg m/s^2 = N. Correct.
            
            # Only apply if shear is significant
            if du_dy_sum > 1e-3:
                 F_lift = 1.61 * D**2 * np.sqrt(rho * mu * du_dy_sum) * ur_x
                 # Lift acts in y direction
                 poly_f[k, 1] += F_lift * 5.0 # Boost lift a bit for saltation visibility

@nb.njit(cache=True)
def sample_u(u, x, y, dx, dy):
    # u is at i*dx, (j+0.5)*dy
    # i goes 0 to nx
    i_f = x / dx
    j_f = y / dy - 0.5
    
    i = int(i_f)
    j = int(j_f)
    
    nx, ny = u.shape
    if i < 0: i = 0
    if i >= nx-1: i = nx-2
    if j < 0: j = 0
    if j >= ny-1: j = ny-2
    
    wx = i_f - i
    wy = j_f - j
    
    val = (1-wx)*(1-wy)*u[i,j] + wx*(1-wy)*u[i+1,j] + (1-wx)*wy*u[i,j+1] + wx*wy*u[i+1,j+1]
    return val

@nb.njit(cache=True)
def sample_v(v, x, y, dx, dy):
    # v is at (i+0.5)*dx, j*dy
    i_f = x / dx - 0.5
    j_f = y / dy
    
    i = int(i_f)
    j = int(j_f)
    
    nx, ny = v.shape
    if i < 0: i = 0
    if i >= nx-1: i = nx-2
    if j < 0: j = 0
    if j >= ny-1: j = ny-2
    
    wx = i_f - i
    wy = j_f - j
    
    val = (1-wx)*(1-wy)*v[i,j] + wx*(1-wy)*v[i+1,j] + (1-wx)*wy*v[i,j+1] + wx*wy*v[i+1,j+1]
    return val
