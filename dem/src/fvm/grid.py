import numpy as np
import numba as nb
import math

class StaggeredGrid:
    def __init__(self, nx, ny, dx, dy):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        
        # Staggered Grid Arrays
        # u: (nx+1, ny) - Vertical faces
        # v: (nx, ny+1) - Horizontal faces
        # p: (nx, ny)   - Cell centers
        # alpha: (nx, ny) - VOF fraction
        
        self.u = np.zeros((nx + 1, ny))
        self.v = np.zeros((nx, ny + 1))
        self.p = np.zeros((nx, ny))
        self.alpha = np.zeros((nx, ny))
        
        # Solid Fraction (epsilon_s) and Solid Velocity
        # Defined at cell centers for simplicity, or staggered?
        # For IBM, we need solid velocity at u and v locations ideally.
        # Let's store at centers and interpolate, or store staggered.
        # Storing at centers is easier for fraction.
        self.eps_s = np.zeros((nx, ny)) 
        self.u_solid = np.zeros((nx, ny))
        self.v_solid = np.zeros((nx, ny))
        
        # Turbulence
        self.nu_t = np.zeros((nx, ny))
        
        # Temporary arrays for solver
        self.div = np.zeros((nx, ny))
        self.p_new = np.zeros((nx, ny))

    def reset_solid(self):
        self.eps_s[:] = 0.0
        self.u_solid[:] = 0.0
        self.v_solid[:] = 0.0

    def rasterize_polygons(self, polygons):
        self.reset_solid()
        
        # Extract polygon data
        # We need a robust way to check polygon overlap with grid cells.
        # For "Cut-Cell" accuracy, we need area fraction.
        # For "Immersed Boundary" (Direct Forcing), we can just check if center is inside.
        # Let's start with center check + supersampling for better epsilon?
        # Or just simple center check for now (eps = 0 or 1).
        # Better: AABB check then detailed check.
        
        rasterize_polygons_numba(
            self.eps_s, self.u_solid, self.v_solid,
            self.nx, self.ny, self.dx, self.dy,
            polygons.vertices_world, polygons.x, polygons.v, polygons.omega
        )

### ************************************************
### Rasterize Polygons (Numba)
@nb.njit(cache=True)
def rasterize_polygons_numba(eps_s, u_solid, v_solid, nx, ny, dx, dy, 
                             poly_verts_list, poly_pos, poly_vel, poly_omega):
    
    # Iterate over all polygons
    # Note: poly_verts_list is a list of arrays, numba might struggle if types vary.
    # Assuming it's a typed list or we iterate in python. 
    # Actually, passing list of numpy arrays to numba is tricky.
    # Let's assume we pass a flattened structure or handle it in python loop calling numba per poly.
    # BUT, for performance, we want the loop here.
    # Let's assume poly_verts_list is a List(float64[:,:]) which is supported in recent numba.
    
    n_polys = len(poly_pos)
    
    for k in range(n_polys):
        verts = poly_verts_list[k]
        cx = poly_pos[k, 0]
        cy = poly_pos[k, 1]
        vx = poly_vel[k, 0]
        vy = poly_vel[k, 1]
        omega = poly_omega[k]
        
        # AABB of polygon
        min_x = np.min(verts[:, 0])
        max_x = np.max(verts[:, 0])
        min_y = np.min(verts[:, 1])
        max_y = np.max(verts[:, 1])
        
        # Grid bounds
        i_min = int(max(0, min_x / dx))
        i_max = int(min(nx, max_x / dx + 1))
        j_min = int(max(0, min_y / dy))
        j_max = int(min(ny, max_y / dy + 1))
        
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                # Cell center
                xc = (i + 0.5) * dx
                yc = (j + 0.5) * dy
                
                # Check if point inside polygon
                if point_in_polygon(xc, yc, verts):
                    eps_s[i, j] = 1.0 # Simple binary mask for now
                    
                    # Solid velocity at this point
                    # v_point = v_cm + omega x r
                    rx = xc - cx
                    ry = yc - cy
                    
                    # omega x r = (-omega * ry, omega * rx)
                    u_point = vx - omega * ry
                    v_point = vy + omega * rx
                    
                    u_solid[i, j] = u_point
                    v_solid[i, j] = v_point

### ************************************************
### Point in Polygon (Ray Casting)
@nb.njit(cache=True)
def point_in_polygon(x, y, verts):
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
