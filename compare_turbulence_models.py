
import numpy as np
import matplotlib.pyplot as plt
import os

def read_csv(filename):
    try:
        # Skip header
        data = np.loadtxt(filename, delimiter=',', skiprows=1)
        return data
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

def plot_comparison(sub_critical_files, super_critical_files):
    # sub_critical_files = {'Label': 'filename', ...}
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)
    
    # Columns: 0 = Sub-Critical, 1 = Super-Critical
    # Rows: 0 = Flux, 1 = Max V, 2 = Max K
    
    scenarios = [
        ('Sub-Critical', sub_critical_files, 0),
        ('Super-Critical', super_critical_files, 1)
    ]
    
    for title, files, col in scenarios:
        ax_flux = axes[0, col]
        ax_vel  = axes[1, col]
        ax_k    = axes[2, col]
        
        ax_flux.set_title(f"{title} - Sediment Flux")
        ax_vel.set_title(f"{title} - Max Particle Velocity")
        ax_k.set_title(f"{title} - Max Turbulent Kinetic Energy")
        
        for label, filename in files.items():
            if not os.path.exists(filename):
                print(f"File not found: {filename}")
                continue
                
            data = read_csv(filename)
            if data is None: continue
            
            # Time is col 0
            t = data[:, 0]
            flux = data[:, 1]
            max_v = data[:, 2]
            max_k = data[:, 3]
            
            ax_flux.plot(t, flux, label=label)
            ax_vel.plot(t, max_v, label=label)
            ax_k.plot(t, max_k, label=label)
            
        for ax in [ax_flux, ax_vel, ax_k]:
            ax.set_xlabel("Time (s)")
            ax.legend()
            ax.grid(True)
            
    axes[0, 0].set_ylabel("Flux (particles)")
    axes[1, 0].set_ylabel("Velocity (m/s)")
    axes[2, 0].set_ylabel("TKE (m2/s2)")
    
    plt.savefig("turbulence_model_comparison.png")
    print("Saved comparison plot to turbulence_model_comparison.png")

if __name__ == "__main__":
    sub_critical = {
        'K-Epsilon': 'flux_sub_critical_k_epsilon.csv',
        'K-Omega SST': 'flux_sub_critical_refined k_omega_SST.csv',
        'K-Omega Std': 'flux_sub_critical_refined_k_omega_std.csv'
    }
    
    super_critical = {
        'K-Epsilon': 'flux_super_critical_k_epsilon.csv',
        'K-Omega SST': 'flux_super_critical_refined_k_omega_SST.csv',
        'K-Omega Std': 'flux_super_critical_refined_k_omega_std.csv'
    }
    
    plot_comparison(sub_critical, super_critical)
