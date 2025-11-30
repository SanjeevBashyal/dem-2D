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

def calculate_statistics(data, start_fraction=0.5):
    """Calculates statistics for the developed flow phase."""
    n = len(data)
    start_idx = int(n * start_fraction)
    subset = data[start_idx:]
    
    stats = {
        'mean': np.mean(subset),
        'max': np.max(subset),
        'std': np.std(subset),
        'min': np.min(subset)
    }
    return stats

def plot_enhanced_comparison(sub_critical_files, super_critical_files):
    """Generates enhanced time-series and statistical plots."""
    
    # Setup Data Structure
    scenarios = [
        ('Sub-Critical', sub_critical_files),
        ('Super-Critical', super_critical_files)
    ]
    
    # 1. Time Series Plots
    fig_ts, axes_ts = plt.subplots(3, 2, figsize=(18, 12), constrained_layout=True)
    
    # 2. Bar Charts for Statistics
    fig_bar, axes_bar = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    
    stats_summary = []

    colors = {'K-Epsilon': 'blue', 'K-Omega SST': 'red', 'K-Omega Std': 'green'}
    linestyles = {'K-Epsilon': '-', 'K-Omega SST': '--', 'K-Omega Std': '-.'}

    for col, (title, files) in enumerate(scenarios):
        ax_flux = axes_ts[0, col]
        ax_vel  = axes_ts[1, col]
        ax_k    = axes_ts[2, col]
        
        ax_flux.set_title(f"{title} - Sediment Flux")
        ax_vel.set_title(f"{title} - Max Particle Velocity")
        ax_k.set_title(f"{title} - Max Turbulent Kinetic Energy")
        
        model_names = []
        means_vel = []
        means_k = []
        stds_vel = []
        
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
            
            # Statistics
            stats_v = calculate_statistics(max_v)
            stats_k = calculate_statistics(max_k)
            stats_flux = calculate_statistics(flux)
            
            model_names.append(label)
            means_vel.append(stats_v['mean'])
            means_k.append(stats_k['mean'])
            stds_vel.append(stats_v['std'])
            
            stats_summary.append({
                'Scenario': title,
                'Model': label,
                'Mean Velocity': stats_v['mean'],
                'Max Velocity': stats_v['max'],
                'Mean TKE': stats_k['mean'],
                'Max TKE': stats_k['max'],
                'Mean Flux': stats_flux['mean']
            })

        # Bar Chart for this scenario
        x = np.arange(len(model_names))
        width = 0.35
        
        ax_bar = axes_bar[col]
        rects1 = ax_bar.bar(x - width/2, means_vel, width, label='Mean Velocity', color='skyblue', yerr=stds_vel, capsize=5)
        
        ax_bar2 = ax_bar.twinx()
        rects2 = ax_bar2.bar(x + width/2, means_k, width, label='Mean TKE', color='orange', alpha=0.7)
        
        ax_bar.set_ylabel('Velocity (m/s)')
        ax_bar2.set_ylabel('TKE (m2/s2)')
        ax_bar.set_title(f'{title} - Statistical Comparison')
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(model_names)
        
        # Legend
        lines, labels = ax_bar.get_legend_handles_labels()
        lines2, labels2 = ax_bar2.get_legend_handles_labels()
        ax_bar.legend(lines + lines2, labels + labels2, loc='upper left')

        # Formatting TS plots
        for ax in [ax_flux, ax_vel, ax_k]:
            ax.set_xlabel("Time (s)")
            ax.grid(True, alpha=0.3)
            ax.legend()

    axes_ts[0, 0].set_ylabel("Flux (particles/s)")
    axes_ts[1, 0].set_ylabel("Velocity (m/s)")
    axes_ts[2, 0].set_ylabel("TKE (m2/s2)")

    plt.figure(fig_ts.number)
    plt.savefig("comparison_time_series.png", dpi=150)
    
    plt.figure(fig_bar.number)
    plt.savefig("comparison_statistics.png", dpi=150)
    
    print("Saved plots: comparison_time_series.png, comparison_statistics.png")
    
    return stats_summary

def generate_report(stats):
    """Generates a markdown report."""
    with open("comparison_summary.md", "w") as f:
        f.write("# RANS Turbulence Model Comparison Report\n\n")
        f.write("## Executive Summary\n")
        f.write("This report compares the performance of k-epsilon, k-omega standard, and k-omega SST models ")
        f.write("in sub-critical and super-critical sediment transport regimes.\n\n")
        
        f.write("## Statistical Summary (Developed Flow)\n")
        f.write("| Scenario | Model | Mean Velocity (m/s) | Max Velocity (m/s) | Mean TKE (m2/s2) | Mean Flux |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        
        for s in stats:
            f.write(f"| {s['Scenario']} | {s['Model']} | {s['Mean Velocity']:.3f} | {s['Max Velocity']:.3f} | ")
            f.write(f"{s['Mean TKE']:.4f} | {s['Mean Flux']:.2f} |\n")
            
        f.write("\n## Key Observations\n")
        f.write("1. **Velocity Prediction**: Compare the 'Mean Velocity' column. Higher velocity usually implies higher shear stress.\n")
        f.write("2. **Turbulence Levels**: 'Mean TKE' indicates the energy available for suspension. k-epsilon often over-predicts TKE near walls compared to k-omega SST.\n")
        f.write("3. **Stability**: The standard deviation (error bars in the bar chart) indicates the fluctuation/instability of the solution.\n")
        
        f.write("\n## Visualizations\n")
        f.write("### Time Series Evolution\n")
        f.write("![Time Series](comparison_time_series.png)\n\n")
        f.write("### Statistical Comparison\n")
        f.write("![Statistics](comparison_statistics.png)\n")

if __name__ == "__main__":
    # Define file paths (Relative to where script is run, assuming root of repo)
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
    
    print("Starting analysis...")
    stats = plot_enhanced_comparison(sub_critical, super_critical)
    generate_report(stats)
    print("Analysis complete. Report generated: comparison_summary.md")
