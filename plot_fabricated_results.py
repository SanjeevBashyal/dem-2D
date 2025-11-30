import numpy as np
import matplotlib.pyplot as plt
import os

def read_csv(filename):
    """Reads CSV file and returns data as numpy array."""
    try:
        # Skip header
        data = np.loadtxt(filename, delimiter=',', skiprows=1)
        return data
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

def moving_average(data, window_size=10):
    """Applies moving average smoothing."""
    if len(data) < window_size:
        return data
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, 'same')

def plot_fabricated_comparison(sub_critical_files, super_critical_files):
    """Generates time-series plots for fabricated data."""
    
    # Setup Data Structure
    scenarios = [
        ('Sub-Critical', sub_critical_files),
        ('Super-Critical', super_critical_files)
    ]
    
    # Time Series Plots
    fig_ts, axes_ts = plt.subplots(3, 2, figsize=(18, 12), constrained_layout=True)
    
    colors = {'K-Epsilon': 'blue', 'K-Omega SST': 'red', 'K-Omega Std': 'green'}
    linestyles = {'K-Epsilon': '-', 'K-Omega SST': '--', 'K-Omega Std': '-.'}

    for col, (title, files) in enumerate(scenarios):
        ax_flux = axes_ts[0, col]
        ax_vel  = axes_ts[1, col]
        ax_k    = axes_ts[2, col]
        
        ax_flux.set_title(f"{title} - Sediment Flux (Fabricated)")
        ax_vel.set_title(f"{title} - Max Particle Velocity")
        ax_k.set_title(f"{title} - Max Turbulent Kinetic Energy")
        
        for label, filename in files.items():
            if not os.path.exists(filename):
                print(f"File not found: {filename}")
                continue
                
            data = read_csv(filename)
            if data is None: continue
            
            # Data Columns: Time, Flux, Max_V, Max_K, Max_Eps
            t = data[:, 0]
            flux = data[:, 1]
            max_v = data[:, 2]
            max_k = data[:, 3]
            
            # Smoothing
            flux_smooth = moving_average(flux, window_size=5)
            max_v_smooth = moving_average(max_v, window_size=10)
            max_k_smooth = moving_average(max_k, window_size=10)
            
            # Plotting
            c = colors.get(label, 'black')
            ls = linestyles.get(label, '-')
            
            ax_flux.plot(t, flux_smooth, label=label, color=c, linestyle=ls, alpha=0.8)
            ax_vel.plot(t, max_v_smooth, label=label, color=c, linestyle=ls, alpha=0.8)
            ax_k.plot(t, max_k_smooth, label=label, color=c, linestyle=ls, alpha=0.8)
            
        # Formatting TS plots
        for ax in [ax_flux, ax_vel, ax_k]:
            ax.set_xlabel("Time (s)")
            ax.grid(True, alpha=0.3)
            ax.legend()

    axes_ts[0, 0].set_ylabel("Flux (particles/s)")
    axes_ts[1, 0].set_ylabel("Velocity (m/s)")
    axes_ts[2, 0].set_ylabel("TKE (m2/s2)")

    plt.savefig("fabricated_comparison.png", dpi=150)
    print("Saved plot: fabricated_comparison.png")

if __name__ == "__main__":
    # Define file paths
    sub_critical = {
        'K-Epsilon': 'fabricated_sub_k_epsilon.csv',
        'K-Omega SST': 'fabricated_sub_sst.csv',
        'K-Omega Std': 'fabricated_sub_std.csv'
    }
    
    super_critical = {
        'K-Epsilon': 'fabricated_super_k_epsilon.csv',
        'K-Omega SST': 'fabricated_super_sst.csv',
        'K-Omega Std': 'fabricated_super_std.csv'
    }
    
    print("Starting plotting...")
    plot_fabricated_comparison(sub_critical, super_critical)
    print("Plotting complete.")
