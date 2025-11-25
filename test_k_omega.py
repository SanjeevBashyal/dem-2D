
import sys
import os
sys.path.append(os.getcwd())

from dem.app.sediment_transport_k_omega import sediment_transport_k_omega_app
import matplotlib.pyplot as plt

def test_run():
    print("Initializing App...")
    app = sediment_transport_k_omega_app("test_run", 0.01, 1.0)
    
    print("Running 100 steps...")
    for i in range(100):
        app.update()
        
    print("Plotting...")
    app.plot(0)
    plt.close(app.fig)
    
    print("Test Complete.")

if __name__ == "__main__":
    test_run()
