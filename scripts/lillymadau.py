#!/usr/bin/env python3
"""
Minimal Lilly-Madau plotter using only the SFR column
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def main():
    #parser = argparse.ArgumentParser(description='Create a simple Lilly-Madau plot from SFR data')
    #parser.add_argument('sfr_file', help='Gadget sfr.txt file')
    #parser.add_argument('--g3_file', help='Optional Gadget-3 sfr.txt file for comparison', default=None)
    #parser.add_argument('--output', help='Output image filename', default='lilly_madau.png')
    #args = parser.parse_args()
    
    # Check if file exists
    #if not os.path.exists(args.sfr_file):
    #    print(f"Error: File {args.sfr_file} not found!")
    #    return
    
    sfr_file = "../output/sfr.txt"
    g3_file = "~/gadget-3.27/output/sfr.txt"

    # Read SFR data - just the necessary columns
    try:
        data = np.loadtxt(sfr_file)
        scale_factor = data[:, 0]  # First column is scale factor
        sfr = data[:, 2]           # Third column is SFR in Msun/yr
        redshift = 1.0/scale_factor - 1.0  # Convert scale factor to redshift
    except Exception as e:
        print(f"Error reading {sfr_file}: {e}")
        return
    
    # Read G3 data if provided
    g3_redshift = None
    g3_sfr = None
    if g3_file and os.path.exists(g3_file):
        try:
            g3_data = np.loadtxt(g3_file)
            g3_scale_factor = g3_data[:, 0]
            g3_sfr = g3_data[:, 2]
            g3_redshift = 1.0/g3_scale_factor - 1.0
        except Exception as e:
            print(f"Error reading {g3_file}: {e}")
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot G4 data
    plt.plot(redshift, sfr, 'o-', label='Gadget-4', color='blue')
    
    # Plot G3 data if available
    if g3_redshift is not None and g3_sfr is not None:
        plt.plot(g3_redshift, g3_sfr, 's--', label='Gadget-3', color='red')
    
    # Plot observational data (Madau & Dickinson 2014)
    z_obs = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    sfr_obs = [0.015, 0.042, 0.066, 0.130, 0.118, 0.093, 0.068, 0.044, 0.028, 0.017]
    sfr_err = [0.003, 0.006, 0.008, 0.018, 0.020, 0.015, 0.014, 0.010, 0.008, 0.006]
    plt.errorbar(z_obs, sfr_obs, yerr=sfr_err, fmt='D', color='black', 
                 alpha=0.7, label='Observations (M&D 2014)')
    
    # Set up log scales
    #plt.xscale('log')
    plt.yscale('log')
    
    # Set axis limits
    plt.xlim(0.1, 10)
    plt.ylim(0.005, max(sfr)*1.5)
    
    # Add labels
    plt.xlabel('Redshift (z)', fontsize=14)
    plt.ylabel('SFR [M$_\\odot$ yr$^{-1}$]', fontsize=14)
    plt.title('Cosmic Star Formation History', fontsize=16)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    
    # Save and display
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Plot saved as {args.output}")
    plt.show()

if __name__ == "__main__":
    main()