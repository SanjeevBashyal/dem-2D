
import os
import sys
import numpy as np
import traceback

# Add path to find dem package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dem.app.sediment_transport_fvm_ibm import sediment_transport_app

class TestApp(sediment_transport_app):
    def create_packed_bed(self):
        print("Skipping particle initialization for test.")
        self.polygons.np = 0 # Ensure no particles

def test_profile_monitor():
    print("Testing Profile Monitor...")
    
    try:
        # Initialize app
        np.random.seed(42) 
        app = TestApp(run_name="test_profile", slope=0.01, u_inlet=1.0)
        
        # Override parameters for short run
        app.t_max = 0.05 
        app.dt = 1e-3    
        app.plot_freq = 10
        app.profile_start_time = 0.0 
        
        # Run loop manually
        print("Running loop...")
        while app.t < app.t_max:
            app.update()
            if app.it % 10 == 0:
                print(f"t={app.t:.2f}")
                
        # Save
        print("Saving profiles...")
        app.profile_monitor.save("profiles_test_profile.csv")
        
        # Check file
        if os.path.exists("profiles_test_profile.csv"):
            print("Success: profiles_test_profile.csv created.")
            data = np.loadtxt("profiles_test_profile.csv", delimiter=',', skiprows=1)
            print(f"Data shape: {data.shape}")
            # Check if values are non-zero (fluid should be moving)
            if np.sum(data[:, 1]) > 0:
                print("Velocity profile has non-zero values.")
            else:
                print("Warning: Velocity profile is all zero.")
        else:
            print("Error: File not created.")
            
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    test_profile_monitor()
