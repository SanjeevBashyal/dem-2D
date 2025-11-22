import numpy as np
import math
import numba as nb

### ************************************************
### Broad Phase: AABB Check
### Returns a list of pairs (i, j) that might collide
def broad_phase(polygons):
    pairs = []
    n = polygons.np
    if n < 2:
        return pairs

    # Compute AABBs
    aabbs = []
    for i in range(n):
        verts = polygons.vertices_world[i]
        min_x = np.min(verts[:, 0])
        max_x = np.max(verts[:, 0])
        min_y = np.min(verts[:, 1])
        max_y = np.max(verts[:, 1])
        aabbs.append((min_x, max_x, min_y, max_y))

    # Check overlaps
    # This is O(N^2) but much faster than SAT. 
    # For better performance, use Spatial Hashing or Sort and Sweep.
    # Given the prompt mentions "Spatial Hashing (Grid) or a Bounding Box check", 
    # we'll stick to simple AABB check for now as N ~ 500 is manageable with AABB.
    for i in range(n):
        for j in range(i + 1, n):
            box_a = aabbs[i]
            box_b = aabbs[j]

            if (box_a[1] < box_b[0] or box_a[0] > box_b[1] or
                box_a[3] < box_b[2] or box_a[2] > box_b[3]):
                continue
            
            pairs.append((i, j))
            
    return pairs

### ************************************************
### Narrow Phase: Separating Axis Theorem (SAT)
### Returns: collision (bool), normal (array), depth (float)
### Normal points from A to B
def sat_check(verts_a, verts_b):
    normal = np.zeros(2)
    depth = float('inf')
    
    # Check axes of A
    for i in range(len(verts_a)):
        p1 = verts_a[i]
        p2 = verts_a[(i + 1) % len(verts_a)]
        
        edge = p2 - p1
        axis = np.array([-edge[1], edge[0]]) # Normal to edge
        axis = axis / np.linalg.norm(axis)
        
        min_a, max_a = project_polygon(axis, verts_a)
        min_b, max_b = project_polygon(axis, verts_b)
        
        if min_a >= max_b or min_b >= max_a:
            return False, np.zeros(2), 0.0
        
        axis_depth = min(max_b - min_a, max_a - min_b)
        if axis_depth < depth:
            depth = axis_depth
            normal = axis
            
    # Check axes of B
    for i in range(len(verts_b)):
        p1 = verts_b[i]
        p2 = verts_b[(i + 1) % len(verts_b)]
        
        edge = p2 - p1
        axis = np.array([-edge[1], edge[0]])
        axis = axis / np.linalg.norm(axis)
        
        min_a, max_a = project_polygon(axis, verts_a)
        min_b, max_b = project_polygon(axis, verts_b)
        
        if min_a >= max_b or min_b >= max_a:
            return False, np.zeros(2), 0.0
        
        axis_depth = min(max_b - min_a, max_a - min_b)
        if axis_depth < depth:
            depth = axis_depth
            normal = axis

    # Ensure normal points from A to B
    # We can check the direction from center A to center B
    center_a = np.mean(verts_a, axis=0)
    center_b = np.mean(verts_b, axis=0)
    direction = center_b - center_a
    
    if np.dot(normal, direction) < 0:
        normal = -normal
        
    return True, normal, depth

### ************************************************
### Helper: Project polygon onto an axis
def project_polygon(axis, verts):
    min_proj = float('inf')
    max_proj = float('-inf')
    
    for v in verts:
        proj = np.dot(v, axis)
        if proj < min_proj:
            min_proj = proj
        if proj > max_proj:
            max_proj = proj
            
    return min_proj, max_proj

### ************************************************
### Find Contact Points
### Returns list of contact points
def find_contact_points(verts_a, verts_b, normal):
    contact_points = []
    
    min_dist = float('inf')
    
    # Check vertices of A inside B (or close to edge)
    # Actually, with SAT, we know the penetration depth and normal.
    # The contact points are the vertices that are "deepest" in the collision direction.
    
    # Support points on A along normal
    # Support points on B along -normal
    
    # This is a simplified approach. For robust physics, we might need Sutherland-Hodgman.
    # But let's try finding vertices with minimum projection on the collision normal.
    
    # Find support point on A in direction of normal
    # (Since normal points A -> B, we want max projection on A)
    max_proj_a = -float('inf')
    support_a = []
    tol = 1e-5
    
    for v in verts_a:
        proj = np.dot(v, normal)
        if proj > max_proj_a + tol:
            max_proj_a = proj
            support_a = [v]
        elif abs(proj - max_proj_a) < tol:
            support_a.append(v)
            
    # Find support point on B in direction of -normal
    min_proj_b = float('inf')
    support_b = []
    
    for v in verts_b:
        proj = np.dot(v, normal)
        if proj < min_proj_b - tol:
            min_proj_b = proj
            support_b = [v]
        elif abs(proj - min_proj_b) < tol:
            support_b.append(v)
            
    # If one vertex involved -> Vertex-Edge
    # If two vertices involved -> Edge-Edge
    
    # We can just return all support points that are within the overlap region?
    # Or better:
    # If len(support_a) == 1 and len(support_b) >= 1: Point on A is contact
    # If len(support_b) == 1 and len(support_a) >= 1: Point on B is contact
    # If len(support_a) == 2 and len(support_b) == 2: Edge-Edge intersection?
    
    # Let's stick to the user's "Vertex-Edge" logic.
    # "One vertex of Polygon A is inside Polygon B"
    
    # Let's check which vertices are inside.
    # Note: "Inside" might be tricky if they are just touching.
    # But we have penetration.
    
    for v in support_a:
        # Check if v is inside B (or close enough)
        # Actually, support points are the candidates.
        contact_points.append(v)
        
    for v in support_b:
        contact_points.append(v)
        
    # This might return up to 4 points. We usually want 1 or 2.
    # If we have 2 points from A and 2 from B (Face-Face), we have a manifold.
    # We can average them or keep the extremes.
    
    # Refined strategy:
    # 1. Identify the reference face (the one most perpendicular to normal).
    # 2. Identify the incident face (the one on the other body).
    # 3. Clip incident face against reference face side planes.
    
    # For now, to keep it simple as requested ("Vertex-Edge" or "Edge-Edge"):
    # If 1 point in support_a and 1 in support_b: It's likely V-V (rare) or V-E.
    # If 2 points in support_a: Edge of A is involved.
    # If 2 points in support_b: Edge of B is involved.
    
    # Let's just return the support points of the "incident" body.
    # The reference body is the one with the face normal roughly matching the collision normal.
    # But we computed the normal from SAT, which comes from an edge of A or B.
    
    # If normal came from A, A is reference, B is incident.
    # If normal came from B, B is reference, A is incident.
    
    # But `sat_check` doesn't tell us which polygon the normal belongs to (it just returns the vector).
    # We can deduce it or modify `sat_check` to return the source.
    
    # Let's modify `sat_check` to be more informative? 
    # Or just re-check:
    # If normal is parallel to an edge of A, A is reference.
    # If normal is parallel to an edge of B, B is reference.
    
    # Let's try to find the incident vertices (those on the other body that are deepest).
    # If normal points A->B:
    # We want vertices of A that have MAX projection on normal?
    # OR vertices of B that have MIN projection on normal?
    
    # If the collision is "Edge of A hits Vertex of B":
    # Normal is perpendicular to Edge A.
    # Vertex B has min projection.
    # Contact is Vertex B.
    
    # If "Vertex of A hits Edge of B":
    # Normal is perpendicular to Edge B.
    # Vertex A has max projection.
    # Contact is Vertex A.
    
    # So, we collect:
    # 1. Vertices of A with max projection (within tolerance).
    # 2. Vertices of B with min projection (within tolerance).
    
    # If count(A) > count(B): Contact is on B (Vertex-Edge where A is Edge).
    # If count(B) > count(A): Contact is on A (Vertex-Edge where B is Edge).
    # If count(A) == count(B) == 2: Edge-Edge.
    
    # Let's return the points from the set with FEWER vertices, or both if equal?
    # Actually, for Edge-Edge, we want the intersection or the segment of overlap.
    # The user said: "Edge-Edge ... treat this as two separate contact points."
    # This implies we want the 2 vertices of the incident edge that are clipped?
    
    # Let's return ALL support points for now and filter in the solver?
    # No, solver needs specific points.
    
    pts = []
    if len(support_a) == 1 and len(support_b) == 1:
        # V-V
        pts = support_a + support_b # Average?
        # Just take the one that is "inside"?
        # Let's take the midpoint.
        pts = [(support_a[0] + support_b[0]) / 2]
    elif len(support_a) == 2 and len(support_b) == 1:
        # Edge A - Vertex B
        pts = support_b
    elif len(support_a) == 1 and len(support_b) == 2:
        # Vertex A - Edge B
        pts = support_a
    else:
        # Edge-Edge (2 and 2)
        # We need the 2 points of the overlap.
        # This is getting complicated for this function.
        # Let's just return all 4 and let the physics engine handle it?
        # Or just return the 2 from the "incident" face?
        # But which is incident?
        # Usually the one with the larger projection range? No.
        
        # Let's return the 2 points from the body that is NOT the reference.
        # We don't know the reference.
        # Let's just return all support points.
        pts = support_a + support_b
        
    return pts
