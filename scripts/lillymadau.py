import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import argparse
import os

def simple_lilly_madau_plot():
    """
    Create a simple Lilly-Madau plot with example data
    """
    # Create some sample data (replace these with your actual data)
    redshift_run1 = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    sfr_run1 = np.array([0.016, 0.048, 0.080, 0.135, 0.110, 0.090, 0.065, 0.045, 0.025, 0.015])
    
    redshift_run2 = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    sfr_run2 = np.array([0.014, 0.042, 0.072, 0.125, 0.100, 0.085, 0.060, 0.040, 0.022, 0.012])
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot simulation data
    plt.plot(redshift_run1, sfr_run1, 'o-', linewidth=2, markersize=8, label='Run 1 (Default)', color='blue')
    plt.plot(redshift_run2, sfr_run2, 's--', linewidth=2, markersize=8, label='Run 2 (With Feature X)', color='red')
    
    # Add observational data points for comparison
    # Data based on Madau & Dickinson (2014)
    z_obs = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    sfr_obs = [0.015, 0.042, 0.066, 0.130, 0.118, 0.093, 0.068, 0.044, 0.028, 0.017]
    sfr_err = [0.003, 0.006, 0.008, 0.018, 0.020, 0.015, 0.014, 0.010, 0.008, 0.006]
    
    plt.errorbar(z_obs, sfr_obs, yerr=sfr_err, fmt='D', color='black', alpha=0.7, 
                markersize=6, label='Observational Data')
    
    # Set axes properties
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.1, 10)
    plt.ylim(0.005, 0.3)
    
    # Format tick labels
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    
    # Add labels and title
    plt.xlabel('Redshift (z)', fontsize=14)
    plt.ylabel('SFRD [M$_\\odot$ yr$^{-1}$ Mpc$^{-3}$]', fontsize=14)
    plt.title('Cosmic Star Formation Rate Density', fontsize=16)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12, loc='best')
    
    # Add second x-axis with lookback time
    ax2 = plt.gca().twiny()
    z_ticks = [0, 1, 2, 3, 5, 7, 10]
    t_lookback = [0, 7.7, 10.3, 11.5, 12.5, 13.0, 13.3]  # Gyr
    
    ax2.set_xscale('log')
    ax2.set_xlim(plt.gca().get_xlim())
    ax2.set_xticks(z_ticks)
    ax2.set_xticklabels([f"{t:.1f}" for t in t_lookback])
    ax2.set_xlabel('Lookback Time (Gyr)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('lilly_madau_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def read_sfr_data(filename):
    """
    Read SFR data from Gadget output file.
    Expected format is a text file with columns: time/scale_factor, redshift, SFR
    
    Returns dictionary with arrays: time, redshift, sfr
    """
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found")
        return None
    
    try:
        data = np.loadtxt(filename, comments='#')
        
        # Check if we have at least 3 columns
        if data.shape[1] < 3:
            print(f"Warning: File {filename} doesn't have enough columns. Expected at least 3.")
            return None
        
        # Extract columns
        time_col = data[:, 0]
        z_col = data[:, 1]
        sfr_col = data[:, 2]
        
        # For Gadget, sometimes the first column is scale factor, and 2nd column is z
        # If redshift isn't already computed, derive it from scale factor
        if np.max(time_col) <= 1.0 and np.min(z_col) < 0:  # First column is scale factor, second isn't redshift
            z_col = 1.0/time_col - 1.0
        
        return {
            'time': time_col,
            'redshift': z_col,
            'sfr': sfr_col
        }
    
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return None

def create_lilly_madau_from_files(file1, file2, label1="Run 1", label2="Run 2", 
                                output="lilly_madau_comparison.png"):
    """
    Create a Lilly-Madau plot from two Gadget SFR files
    """
    # Read data from files
    data1 = read_sfr_data(file1)
    data2 = read_sfr_data(file2)
    
    if data1 is None or data2 is None:
        print("Could not create plot due to missing data")
        return
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot simulation data
    plt.plot(data1['redshift'], data1['sfr'], 'o-', linewidth=2, markersize=6, 
             label=label1, color='blue')
    plt.plot(data2['redshift'], data2['sfr'], 's--', linewidth=2, markersize=6, 
             label=label2, color='red')
    
    # Add observational data points for comparison
    # Data based on Madau & Dickinson (2014)
    z_obs = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    sfr_obs = [0.015, 0.042, 0.066, 0.130, 0.118, 0.093, 0.068, 0.044, 0.028, 0.017]
    sfr_err = [0.003, 0.006, 0.008, 0.018, 0.020, 0.015, 0.014, 0.010, 0.008, 0.006]
    
    plt.errorbar(z_obs, sfr_obs, yerr=sfr_err, fmt='D', color='black', alpha=0.7, 
                markersize=6, label='Observational Data (M&D 2014)')
    
    # Set axes properties
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.1, 10)
    plt.ylim(0.005, 0.3)
    
    # Format tick labels
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    
    # Add labels and title
    plt.xlabel('Redshift (z)', fontsize=14)
    plt.ylabel('SFRD [M$_\\odot$ yr$^{-1}$ Mpc$^{-3}$]', fontsize=14)
    plt.title('Cosmic Star Formation Rate Density', fontsize=16)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12, loc='best')
    
    # Add second x-axis with lookback time
    ax2 = plt.gca().twiny()
    z_ticks = [0, 1, 2, 3, 5, 7, 10]
    t_lookback = [0, 7.7, 10.3, 11.5, 12.5, 13.0, 13.3]  # Gyr
    
    ax2.set_xscale('log')
    ax2.set_xlim(plt.gca().get_xlim())
    ax2.set_xticks(z_ticks)
    ax2.set_xticklabels([f"{t:.1f}" for t in t_lookback])
    ax2.set_xlabel('Lookback Time (Gyr)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Create a Lilly-Madau plot from Gadget SFR files')
    parser.add_argument('--file1', help='First SFR file', type=str, default='')
    parser.add_argument('--file2', help='Second SFR file', type=str, default='')
    parser.add_argument('--label1', help='Label for first run', default='Run 1', type=str)
    parser.add_argument('--label2', help='Label for second run', default='Run 2', type=str)
    parser.add_argument('--output', help='Output file name', default='lilly_madau_comparison.png', type=str)
    parser.add_argument('--demo', help='Run with example data', action='store_true')
    
    args = parser.parse_args()
    
    if args.demo:
        print("Running with example data...")
        simple_lilly_madau_plot()
        return
    
    if not args.file1 or not args.file2:
        print("Error: Please provide two SFR files or use --demo for example data")
        return
    
    create_lilly_madau_from_files(args.file1, args.file2, args.label1, args.label2, args.output)

if __name__ == "__main__":
    main()