import numpy as np
import math
from pysph.solver.application import Application
from pysph.sph.scheme import WCSPHScheme, SchemeChooser
from pysph.sph.wc.edac import EDACScheme
from pysph.base.utils import get_particle_array
from pysph.base.kernels import QuinticSpline

# Custom imports
from dem.src.core.polygons import Polygons
from dem.src.core.polygon_solver import PolygonSolver
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolyPatch

class OpenChannelSPH(Application):
    def initialize(self):
        print("DEBUG: Initializing OpenChannelSPH")
        print("DEBUG: dir(self):", dir(self))
        # Geometry
        self.L = 2.0
        self.H = 1.0
        self.slope_deg = 10.0
        self.g_mag = 9.81
        self.gx = self.g_mag * math.sin(math.radians(self.slope_deg))
        self.gy = -self.g_mag * math.cos(math.radians(self.slope_deg))
        
        # SPH Parameters
        self.dx = 0.02
        self.hdx = 1.2
        self.h = self.hdx * self.dx
        self.ro = 1000.0
        self.nu = 1e-6
        self.c0 = 10.0 * np.sqrt(2 * 9.81 * 0.2) # Speed of sound approx
        self.gamma = 7.0
        self.alpha = 0.1
        
        # DEM Parameters
        self.polygons = Polygons(material="sand")
        self.dem_solver = PolygonSolver()
        self.dem_solver.k1 = 1.0e7
        self.dem_solver.k2 = 2.0e7
        
        # Add 3 particles
        self.polygons.add(self.create_random_poly(0.02), [0.5, 0.1], [0.0, 0.0], 0.0, 0.0, color='r')
        self.polygons.add(self.create_random_poly(0.02), [0.8, 0.1], [0.0, 0.0], 0.0, 0.0, color='g')
        self.polygons.add(self.create_random_poly(0.02), [1.1, 0.1], [0.0, 0.0], 0.0, 0.0, color='b')
        
        # Simulation
        self.dt = 1e-4
        self.tf = 2.0
        self.plot_freq = 100
        
    def create_random_poly(self, radius):
        n_sides = 5
        verts = []
        for k in range(n_sides):
            angle = 2 * math.pi * k / n_sides
            vx = radius * math.cos(angle)
            vy = radius * math.sin(angle)
            verts.append([vx, vy])
        return verts
        
    def create_particles(self):
        # 1. Fluid Particles
        # Initial block of water
        x_min, x_max = 0.0, 0.5
        y_min, y_max = 0.05, 0.25
        
        x, y = np.mgrid[x_min:x_max:self.dx, y_min:y_max:self.dx]
        x = x.ravel()
        y = y.ravel()
        
        m = self.dx * self.dx * self.ro
        
        fluid = get_particle_array(name='fluid', x=x, y=y, h=self.h, m=m, rho=self.ro)
        
        # 2. Wall Particles (Bed)
        # Wall at y=0
        wx, wy = np.mgrid[-0.1:self.L+0.1:self.dx, -0.1:0.0:self.dx]
        wx = wx.ravel()
        wy = wy.ravel()
        
        wall = get_particle_array(name='wall', x=wx, y=wy, h=self.h, m=m, rho=self.ro)
        
        # 3. DEM Boundary Particles
        # Generate particles on the surface of DEM polygons
        # We need to track which particle belongs to which polygon
        dx_dem = self.dx / 2.0 # Finer resolution for boundary
        
        bx, by, b_pid = [], [], []
        
        for i in range(self.polygons.np):
            verts = self.polygons.vertices_world[i]
            # Perimeter
            for j in range(len(verts)):
                p1 = verts[j]
                p2 = verts[(j+1)%len(verts)]
                dist = np.linalg.norm(p2 - p1)
                n_pts = int(dist / dx_dem) + 1
                for k in range(n_pts):
                    t = k / n_pts
                    p = p1 + t * (p2 - p1)
                    bx.append(p[0])
                    by.append(p[1])
                    b_pid.append(i)
                    
        dem_b = get_particle_array(name='dem_boundary', x=bx, y=by, h=self.h, m=m, rho=self.ro)
        dem_b.add_property('pid', type='int', data=b_pid)
        dem_b.add_property('fx', type='double', data=np.zeros_like(bx))
        dem_b.add_property('fy', type='double', data=np.zeros_like(by))
        
        self.scheme.setup_properties([fluid, wall, dem_b])
        
        return [fluid, wall, dem_b]

    def create_scheme(self):
        s = EDACScheme(
            ['fluid'], ['wall', 'dem_boundary'], dim=2, rho0=self.ro, c0=self.c0,
            h=self.h, pb=0.0, nu=self.nu, gy=self.gy, gx=self.gx
        )
        return s

    def configure_scheme(self):
        scheme = self.scheme
        kernel = QuinticSpline(dim=2)
        scheme.configure_solver(kernel=kernel, tf=self.tf, dt=self.dt, pfreq=self.plot_freq)

    def create_solver(self):
        return self.scheme.get_solver()

    def update_dem_boundary(self):
        # Update positions of dem_boundary particles based on current polygon positions
        # This is a simplification: we regenerate them or transform them?
        # Transforming is faster. We need local coordinates.
        # For now, let's just regenerate them roughly or assume rigid body motion if we stored local coords.
        # Let's regenerate for simplicity of implementation right now, though slow.
        # Optimization: Store local coords and transform.
        
        # Actually, let's just clear and rebuild.
        # Access the particle array
        dem_b = self.solver.particles[2] # Assuming order [fluid, wall, dem_boundary]
        # Wait, particles list might be reordered or dict?
        # self.solver.particles is a list.
        # Better find by name.
        for pa in self.solver.particles:
            if pa.name == 'dem_boundary':
                dem_b = pa
                break
        
        dx_dem = self.dx / 2.0
        bx, by, b_pid = [], [], []
        u, v = [], []
        
        for i in range(self.polygons.np):
            verts = self.polygons.vertices_world[i]
            # Polygon velocity
            v_poly = self.polygons.v[i]
            w_poly = self.polygons.omega[i]
            center = self.polygons.x[i]
            
            for j in range(len(verts)):
                p1 = verts[j]
                p2 = verts[(j+1)%len(verts)]
                dist = np.linalg.norm(p2 - p1)
                n_pts = int(dist / dx_dem) + 1
                for k in range(n_pts):
                    t = k / n_pts
                    p = p1 + t * (p2 - p1)
                    bx.append(p[0])
                    by.append(p[1])
                    b_pid.append(i)
                    
                    # Velocity at this point: v = v_cm + omega x r
                    r = p - center
                    # omega x r (2D) = (-omega * ry, omega * rx)
                    vx = v_poly[0] - w_poly * r[1]
                    vy = v_poly[1] + w_poly * r[0]
                    u.append(vx)
                    v.append(vy)
        
        # Resize if needed (PySPH arrays are resizable?)
        # Yes, but it's tricky. 
        # For now, let's assume constant number of particles if geometry doesn't change much?
        # No, geometry changes.
        # PySPH particle arrays can be resized.
        
        # Simpler approach: Just update x, y, u, v. 
        # If size changes, we might need to remove/add.
        # Let's assume we just overwrite for now and hope size matches roughly or handle resize.
        # Actually, `set_data` might work if we match keys.
        
        new_data = {'x': np.array(bx), 'y': np.array(by), 'u': np.array(u), 'v': np.array(v), 'pid': np.array(b_pid)}
        # We also need to keep other props like h, m, rho.
        # This is getting complicated for a quick implementation.
        
        # Alternative: Just update positions if we keep the same particles?
        # But we generated them on perimeter.
        # Let's just update x, y, u, v assuming the number of points on perimeter is constant (it is, based on edge length).
        # Wait, edge length is constant for rigid bodies.
        # So number of particles is constant!
        # We just need to ensure the order is preserved.
        
        # Re-generate in same order:
        idx = 0
        for i in range(self.polygons.np):
            verts = self.polygons.vertices_world[i]
            v_poly = self.polygons.v[i]
            w_poly = self.polygons.omega[i]
            center = self.polygons.x[i]
            
            for j in range(len(verts)):
                p1 = verts[j]
                p2 = verts[(j+1)%len(verts)]
                dist = np.linalg.norm(p2 - p1)
                n_pts = int(dist / dx_dem) + 1
                for k in range(n_pts):
                    t = k / n_pts
                    p = p1 + t * (p2 - p1)
                    
                    # Update arrays directly
                    dem_b.x[idx] = p[0]
                    dem_b.y[idx] = p[1]
                    
                    r = p - center
                    dem_b.u[idx] = v_poly[0] - w_poly * r[1]
                    dem_b.v[idx] = v_poly[1] + w_poly * r[0]
                    
                    idx += 1

    def compute_coupling_forces(self):
        # Aggregate forces from dem_boundary particles to polygons
        dem_b = None
        for pa in self.solver.particles:
            if pa.name == 'dem_boundary':
                dem_b = pa
                break
        
        # Reset polygon forces (handled in loop)
        # self.polygons.reset_forces() # Done in loop
        
        # SPH calculates acceleration 'au', 'av'. Force = m * a
        # Or pressure force?
        # EDAC/WCSPH computes accelerations.
        # Force on solid particle i: F_i = m_i * a_i
        # But 'a_i' computed by SPH for boundary particles includes pressure/viscous forces from fluid.
        
        fx = dem_b.m * dem_b.au
        fy = dem_b.m * dem_b.av
        
        for k in range(len(dem_b.x)):
            pid = dem_b.pid[k]
            f_vec = np.array([fx[k], fy[k]])
            
            # Add to polygon force
            self.polygons.f[pid] += f_vec
            
            # Add to polygon torque: T = r x F
            p = np.array([dem_b.x[k], dem_b.y[k]])
            center = self.polygons.x[pid]
            r = p - center
            torque = r[0] * f_vec[1] - r[1] * f_vec[0]
            self.polygons.tau[pid] += torque

    def run(self):
        self.setup([])
        solver = self.solver
        
        while solver.t < solver.tf:
            # 1. DEM Step (Predictor/Verlet 1)
            self.polygons.reset_forces()
            self.polygons.gravity([self.gx, self.gy])
            self.polygons.integrate_pos(self.dt)
            
            # 2. Update SPH Boundary Particles
            self.update_dem_boundary()
            
            # 3. SPH Step
            solver.integrator.step(solver.t, solver.dt)
            
            # 4. Coupling Forces
            self.compute_coupling_forces()
            
            # 5. DEM Step (Force Update & Corrector/Verlet 2)
            self.dem_solver.solve(self.polygons, self.dt) # Collisions
            self.polygons.integrate_vel(self.dt)
            
            # Update time
            solver.t += solver.dt
            solver.count += 1
            
            # 6. Output/Plot
            if solver.count % self.plot_freq == 0:
                print(f"Iter: {solver.count}, Time: {solver.t:.4f}")
                self.plot(solver)

    def plot(self, solver):
        plt.clf()
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_xlim(-0.1, self.L + 0.1)
        ax.set_ylim(-0.1, self.H)
        
        # Plot Fluid
        fluid = None
        for pa in solver.particles:
            if pa.name == 'fluid':
                fluid = pa
                break
        
        if fluid:
            plt.scatter(fluid.x, fluid.y, s=5, c=fluid.p, cmap='viridis', alpha=0.5)
            plt.colorbar(label='Pressure')
            
        # Plot Polygons
        for i in range(self.polygons.np):
            verts = self.polygons.vertices_world[i]
            c = self.polygons.color[i]
            poly = PolyPatch(verts, closed=True, facecolor=c, edgecolor='k', alpha=0.8)
            ax.add_patch(poly)
            
        plt.pause(0.001)
