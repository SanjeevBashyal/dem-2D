import numpy as np
import math
import numba as nb
from dem.src.core.sph_solver import kernel

### ************************************************
### Compute Coupling Forces
### fluid: FluidParticles object
### polygons: Polygons object
def compute_coupling_forces(fluid, polygons):
    if fluid.np == 0 or polygons.np == 0:
        return

    # Extract data for numba
    f_x = fluid.x
    f_v = fluid.v
    f_f = fluid.f
    f_h = fluid.h
    f_m = fluid.mass
    
    p_verts = polygons.vertices_world
    p_x     = polygons.x
    p_v     = polygons.v
    p_omega = polygons.omega
    p_f     = polygons.f
    p_tau   = polygons.tau
    
    # Numba-friendly list of arrays
    # We need to handle the list of vertices carefully.
    # Numba doesn't like list of arrays with different shapes if not typed perfectly.
    # For now, let's iterate in python or use a simplified approach.
    # Or pass the list to a numba function if it supports it (newer numba does).
    
    # Let's try to keep the loop in python for the polygon iteration, 
    # and call a numba function for the fluid iteration per polygon.
    # This avoids numba list issues.
    
    for i in range(polygons.np):
        verts = polygons.vertices_world[i]
        
        # AABB check first
        min_x = np.min(verts[:, 0]) - f_h
        max_x = np.max(verts[:, 0]) + f_h
        min_y = np.min(verts[:, 1]) - f_h
        max_y = np.max(verts[:, 1]) + f_h
        
        # Call numba function for this polygon
        compute_coupling_single_polygon(
            f_x, f_v, f_f, f_h, f_m,
            verts, p_x[i], p_v[i], p_omega[i], p_f, p_tau, i,
            min_x, max_x, min_y, max_y
        )

### ************************************************
### Compute Coupling for Single Polygon
@nb.njit(cache=True)
def compute_coupling_single_polygon(
    f_x, f_v, f_f, f_h, f_m,
    verts, p_pos, p_vel, p_omega, p_f, p_tau, p_idx,
    min_x, max_x, min_y, max_y
):
    n_fluid = len(f_x)
    n_verts = len(verts)
    
    # Stiffness for wall repulsion
    # Should be related to fluid stiffness B
    # K_wall ~ B * h? Or just a penalty.
    k_wall = 100000.0 
    
    for j in range(n_fluid):
        # Broad phase (AABB)
        fx = f_x[j, 0]
        fy = f_x[j, 1]
        
        if fx < min_x or fx > max_x or fy < min_y or fy > max_y:
            continue
            
        # Narrow phase: Distance to polygon
        dist, nx, ny, contact_x, contact_y = point_to_polygon_dist(fx, fy, verts)
        
        # If inside or close enough (dist < h)
        # Note: point_to_polygon_dist returns negative if inside? 
        # Let's assume it returns signed distance (negative inside) 
        # or positive distance to boundary.
        # We want repulsion if dist < h (and definitely if inside).
        
        # If dist is positive (outside) but < h: Repel
        # If dist is negative (inside): Repel strongly
        
        overlap = f_h - dist
        if overlap > 0:
            # Force magnitude
            # Penalty force: F = k * overlap
            # Or Lennard-Jones: (h/d)^4 - (h/d)^2 ...
            
            # Simple linear spring for now
            f_mag = k_wall * overlap
            
            # Direction: Normal from polygon to fluid
            # If inside, normal should point out.
            # If outside, normal points away from surface.
            
            # Force on Fluid
            f_fluid_x = f_mag * nx
            f_fluid_y = f_mag * ny
            
            f_f[j, 0] += f_fluid_x
            f_f[j, 1] += f_fluid_y
            
            # Reaction on Polygon
            # Force
            p_f[p_idx, 0] -= f_fluid_x
            p_f[p_idx, 1] -= f_fluid_y
            
            # Torque
            # r = contact_point - center_of_mass
            rx = contact_x - p_pos[0]
            ry = contact_y - p_pos[1]
            
            # tau = r x F_reaction
            # F_reaction = -F_fluid
            fr_x = -f_fluid_x
            fr_y = -f_fluid_y
            
            torque = rx * fr_y - ry * fr_x
            p_tau[p_idx] += torque

### ************************************************
### Point to Polygon Distance
### Returns: dist, nx, ny, contact_x, contact_y
### dist is signed: negative inside, positive outside
### normal points OUT of the polygon (towards fluid)
@nb.njit(cache=True)
def point_to_polygon_dist(px, py, verts):
    n_verts = len(verts)
    min_dist_sq = 1.0e20
    best_nx = 0.0
    best_ny = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    inside = True
    
    for i in range(n_verts):
        p1 = verts[i]
        p2 = verts[(i + 1) % n_verts]
        
        # Segment p1-p2
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        
        dx = x2 - x1
        dy = y2 - y1
        
        # Normal to edge (pointing out?)
        # Assuming CCW winding, normal is (dy, -dx)
        # Let's verify winding later. Assuming standard CCW.
        nx = dy
        ny = -dx
        
        # Normalize
        l = math.sqrt(nx*nx + ny*ny)
        if l > 0:
            nx /= l
            ny /= l
            
        # Project point onto line
        # v = p - p1
        vx = px - x1
        vy = py - y1
        
        # Dot with normal to check side
        dot_n = vx * nx + vy * ny
        if dot_n > 0:
            inside = False # Outside at least one edge
            
        # Distance to segment
        # t = dot(v, edge) / |edge|^2
        edge_len_sq = dx*dx + dy*dy
        t = (vx * dx + vy * dy) / edge_len_sq
        
        # Clamp t to segment
        if t < 0: t = 0
        if t > 1: t = 1
        
        # Closest point on segment
        cx = x1 + t * dx
        cy = y1 + t * dy
        
        # Distance squared
        d_sq = (px - cx)**2 + (py - cy)**2
        
        if d_sq < min_dist_sq:
            min_dist_sq = d_sq
            best_cx = cx
            best_cy = cy
            
            # Normal depends on where we are
            # If we are outside, normal is (p - c) / |p - c|
            # If we are inside, normal is the edge normal?
            # This is tricky for signed distance of arbitrary convex.
            
            # Let's use the edge normal if we are "closest" to that edge?
            # Or just p - c.
            
            # If t is 0 or 1 (vertex), p-c is correct.
            # If t is in (0, 1), p-c is parallel to normal.
            
            dist_vec_x = px - cx
            dist_vec_y = py - cy
            dist_len = math.sqrt(d_sq)
            
            if dist_len > 1e-8:
                best_nx = dist_vec_x / dist_len
                best_ny = dist_vec_y / dist_len
            else:
                best_nx = nx
                best_ny = ny

    dist = math.sqrt(min_dist_sq)
    if inside:
        dist = -dist
        # If inside, the normal calculated above (p-c) points towards the center?
        # No, p is inside, c is on boundary. p-c points OUT.
        # So best_nx is correct (pointing out).
        
    return dist, best_nx, best_ny, best_cx, best_cy
