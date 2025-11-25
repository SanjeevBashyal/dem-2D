
import sys
import os
sys.path.append(os.getcwd())

from dem.app.sediment_transport_k_omega_standard import sediment_transport_k_omega_standard_app
import matplotlib.pyplot as plt
import numpy as np

def test_run():
    print("Initializing App...")
    app = sediment_transport_k_omega_standard_app("test_run_std", 0.01, 1.0)
    
    print("Running 200 steps...")
    try:
        for i in range(200):
            app.update()
            if i % 10 == 0:
                max_k = np.max(app.fluid_solver.k)
                max_w = np.max(app.fluid_solver.omega_t)
                print(f"Step {i}: Max k={max_k}, Max w={max_w}")
                
    except Exception as e:
        print(f"Crashed at step {i}")
        print(e)
        import traceback
        traceback.print_exc()
        
    print("Plotting...")
    app.plot(0)
    plt.close(app.fig)
    
    print("Test Complete.")

if __name__ == "__main__":
    test_run()
