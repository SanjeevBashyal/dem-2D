
from dem.src.core.polygons import Polygons
import numpy as np

try:
    print("Initializing Polygons...")
    polygons = Polygons(material='steel')
    print("Polygons initialized.")
    
    verts = [[0,0], [1,0], [1,1], [0,1]]
    print("Adding particle...")
    polygons.add(verts, x=[0,0], v=[0,0], theta=0.0, omega=0.0)
    print("Particle added.")
    
except Exception as e:
    print(f"Caught error: {e}")
    import traceback
    traceback.print_exc()
