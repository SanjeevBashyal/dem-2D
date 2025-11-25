import numpy as np
import math
from dem.src.core.polygon_collision import broad_phase, sat_check, find_contact_points

class PolygonSolver:
    def __init__(self):
        # Contact history: {(id_a, id_b): {'delta_max': float, 'prev_delta': float}}
        self.contact_history = {}
        
        # Stiffness constants (should come from material, but hardcoding for now/prototype)
        self.k1 = 1.0e7 # Loading stiffness (Increased for less overlap)
        self.k2 = 2.0e7 # Unloading stiffness

    def solve(self, polygons, dt):
        # 1. Broad Phase
        pairs = broad_phase(polygons)
        
        current_contacts = set()
        
        for i, j in pairs:
            # 2. Narrow Phase (SAT)
            verts_a = polygons.vertices_world[i]
            verts_b = polygons.vertices_world[j]
            
            colliding, normal, depth = sat_check(verts_a, verts_b)
            
            if colliding:
                pair_id = tuple(sorted((i, j)))
                current_contacts.add(pair_id)
                
                # Ensure normal points from i to j? 
                # sat_check returns normal. We checked direction in sat_check.
                # If normal points A->B, force on A is -Fn * n.
                
                # Contact Points
                pts = find_contact_points(verts_a, verts_b, normal)
                if not pts:
                    continue
                    
                # Use average contact point for force application
                # (Simplification for now, ideally distribute force)
                cp = np.mean(pts, axis=0)
                
                # 3. Physics (Forces)
                self.apply_forces(polygons, i, j, normal, depth, cp, dt, pair_id)
                
        # Clean up old contacts
        # (Remove pairs that are no longer colliding)
        # This is important for hysteresis reset
        for pair_id in list(self.contact_history.keys()):
            if pair_id not in current_contacts:
                del self.contact_history[pair_id]

    def apply_forces(self, p, i, j, normal, delta, cp, dt, pair_id):
        # Retrieve history
        if pair_id not in self.contact_history:
            self.contact_history[pair_id] = {'delta_max': delta, 'prev_delta': delta}
        
        history = self.contact_history[pair_id]
        delta_max = history['delta_max']
        prev_delta = history['prev_delta']
        
        # Update delta_max
        if delta > delta_max:
            delta_max = delta
            history['delta_max'] = delta_max
            
        # 1. Normal Force (Walton-Braun)
        fn_mag = 0.0
        if delta > prev_delta:
            # Loading
            fn_mag = self.k1 * delta
        else:
            # Unloading
            f_max = self.k1 * delta_max
            fn_mag = f_max - self.k2 * (delta_max - delta)
            
        # Clamp Fn to be non-negative (no attraction)
        fn_mag = max(0.0, fn_mag)
        
        fn = fn_mag * normal
        
        # 2. Tangential Force (Friction)
        # Relative velocity at contact point
        # V_rel = Vb - Va + (wb x rb) - (wa x ra)
        # 2D cross product: w scalar, r vector -> (-w*ry, w*rx)
        
        va = p.v[i]
        vb = p.v[j]
        wa = p.omega[i]
        wb = p.omega[j]
        
        ra = cp - p.x[i]
        rb = cp - p.x[j]
        
        # Cross products (w x r) in 2D
        waxra = np.array([-wa * ra[1], wa * ra[0]])
        wbxrb = np.array([-wb * rb[1], wb * rb[0]])
        
        v_rel = vb + wbxrb - (va + waxra)
        
        # Tangential velocity
        vn_mag = np.dot(v_rel, normal)
        vt = v_rel - vn_mag * normal
        
        vt_mag = np.linalg.norm(vt)
        ft = np.zeros(2)
        
        if vt_mag > 1e-8:
            t_dir = vt / vt_mag
            
            # Simple spring friction or just Coulomb?
            # User said: "Apply a spring force... Cap it at Coulomb"
            # For simplicity, let's use viscous friction + Coulomb cap first, 
            # or just Coulomb for stability if no spring history.
            # User asked for spring force. That requires storing tangential displacement.
            # Let's stick to Coulomb for now to avoid complexity of tangential history 
            # unless strictly necessary. "Apply a spring force" suggests history.
            # Let's add 'shear_disp' to history.
            
            if 'shear_disp' not in history:
                history['shear_disp'] = np.zeros(2)
                
            # Integrate relative tangential velocity to get displacement
            history['shear_disp'] += vt * dt
            shear_disp = history['shear_disp']
            
            kt = 0.8 * self.k1 # Tangential stiffness
            ft_spring = -kt * shear_disp
            
            # Coulomb Cap
            mu = 0.5 # Friction coefficient
            ft_max = mu * fn_mag
            
            if np.linalg.norm(ft_spring) > ft_max:
                ft_spring = ft_max * (ft_spring / np.linalg.norm(ft_spring))
                # Slip occurred, reset/adjust displacement? 
                # Standard practice: adjust shear_disp to match the cap
                history['shear_disp'] = -ft_spring / kt
                
            ft = ft_spring
            
        # Total Force
        f_total = fn + ft # Force on B?
        # Normal points A->B.
        # Fn is force pushing B away. So +Fn on B.
        # Ft opposes motion of B relative to A.
        
        # Apply to particles
        # Force on A is opposite
        p.f[i] -= (fn + ft) # Wait, check signs.
        # Normal A->B. Fn pushes B. So +Fn on B. -Fn on A.
        # Ft opposes V_rel (Vb - Va). So it acts on B to oppose V_rel.
        # So +Ft on B?
        # v_rel is B relative to A. Friction on B should oppose v_rel.
        # If v_rel is positive, friction should be negative.
        # ft calculation: -kt * shear_disp.
        # shear_disp accumulates vt (v_rel_tangent).
        # So ft is opposite to displacement/velocity. Correct.
        
        p.f[j] += (fn + ft)
        p.f[i] -= (fn + ft)
        
        # 3. Torque
        # tau = r x F
        # 2D cross product: rx * Fy - ry * Fx
        
        # Torque on A
        # Force on A is -(fn + ft)
        fa = -(fn + ft)
        tau_a = ra[0] * fa[1] - ra[1] * fa[0]
        p.tau[i] += tau_a
        
        # Torque on B
        # Force on B is +(fn + ft)
        fb = (fn + ft)
        tau_b = rb[0] * fb[1] - rb[1] * fb[0]
        p.tau[j] += tau_b
        
        # Update prev_delta
        history['prev_delta'] = delta

    def compute_hydro_forces(self, grid, polygons, rho_fluid, mu_fluid):
        # Simple Drag and Buoyancy
        for i in range(polygons.np):
            if polygons.fixed[i]:
                continue
                
            # 1. Sample Fluid Velocity at Particle Center
            # (Simple bilinear interpolation)
            cx = polygons.x[i, 0]
            cy = polygons.x[i, 1]
            
            # Grid indices
            # u is at (i+0.5, j) -> x = (i+0.5)*dx, y = (j+0.5)*dy
            # Wait, u is staggered?
            # grid.u is (nx+1, ny). u[i,j] is at x=i*dx, y=(j+0.5)*dy
            # grid.v is (nx, ny+1). v[i,j] is at x=(i+0.5)*dx, y=j*dy
            
            # Interpolate u
            i_u = cx / grid.dx
            j_u = (cy - 0.5 * grid.dy) / grid.dy
            u_fluid = self.sample_grid(grid.u, i_u, j_u)
            
            # Interpolate v
            i_v = (cx - 0.5 * grid.dx) / grid.dx
            j_v = cy / grid.dy
            v_fluid = self.sample_grid(grid.v, i_v, j_v)
            
            # Interpolate Pressure/Buoyancy?
            # Or just use Archimedes: F_b = -rho * g * Volume
            # In 2D, Volume = Area.
            # But we need submerged area.
            # Check VOF (alpha) at center
            i_c = int(cx / grid.dx)
            j_c = int(cy / grid.dy)
            
            alpha = 0.0
            if 0 <= i_c < grid.nx and 0 <= j_c < grid.ny:
                alpha = grid.alpha[i_c, j_c]
            
            # If alpha > 0, apply buoyancy and drag
            if alpha > 0.01:
                # Area (Mass / Density)
                # Or recompute from vertices?
                # polygons.m[i] = area * density
                area = polygons.m[i] / polygons.mtr.density
                
                # Buoyancy (Vertical Up)
                # F_b = rho_fluid * g * Volume * alpha
                # g is vector now?
                # If solver.g is vector, buoyancy opposes it?
                # Usually F_b = -rho_fluid * g_vector * Volume
                # But we passed rho_fluid. We need g.
                # Let's assume standard gravity for buoyancy direction or pass g?
                # The method signature doesn't have g.
                # Let's assume vertical y for now or just drag.
                # Buoyancy is implicitly handled if we use pressure gradient force?
                # No, IBM usually adds explicit buoyancy if not resolving pressure on surface.
                # Let's add simple vertical buoyancy: F_b = rho_fluid * 9.81 * area * alpha
                # polygons.f[i, 1] += rho_fluid * 9.81 * area * alpha
                
                # Drag
                # F_d = 0.5 * Cd * rho * A * |u_rel| * u_rel
                # Or Stokes: F_d = 3 * pi * mu * D * u_rel
                # Diameter approx sqrt(area)
                D = 2.0 * math.sqrt(area / math.pi)
                
                u_rel = u_fluid - polygons.v[i, 0]
                v_rel = v_fluid - polygons.v[i, 1]
                
                # Reynolds number
                vel_mag = math.sqrt(u_rel**2 + v_rel**2)
                if vel_mag > 1e-8:
                    Re = rho_fluid * vel_mag * D / mu_fluid
                    Cd = 24.0 / Re * (1.0 + 0.15 * Re**0.687) # Schiller-Naumann
                    
                    drag_force_mag = 0.5 * rho_fluid * D * Cd * vel_mag * alpha # D is projected area in 2D? (Length)
                    
                    fx = drag_force_mag * (u_rel / vel_mag)
                    fy = drag_force_mag * (v_rel / vel_mag)
                    
                    polygons.f[i, 0] += fx
                    polygons.f[i, 1] += fy
                    
                    # Buoyancy (Archimedes)
                    # F_b = rho_fluid * Volume * g_eff
                    # We need g.
                    # Let's assume g = 9.81 up.
                    polygons.f[i, 1] += rho_fluid * 9.81 * area * alpha

    def sample_grid(self, field, i_f, j_f):
        # Bilinear interpolation
        i = int(i_f)
        j = int(j_f)
        
        nx, ny = field.shape
        
        # Clamp
        if i < 0: i = 0
        if i >= nx - 1: i = nx - 2
        if j < 0: j = 0
        if j >= ny - 1: j = ny - 2
        
        wx = i_f - i
        wy = j_f - j
        
        # Clamp weights
        wx = max(0.0, min(1.0, wx))
        wy = max(0.0, min(1.0, wy))
        
        val = (1 - wx) * (1 - wy) * field[i, j] + \
              wx * (1 - wy) * field[i+1, j] + \
              (1 - wx) * wy * field[i, j+1] + \
              wx * wy * field[i+1, j+1]
              
        return val
