#!/usr/bin/env python3
"""
Dust Diagnostics Analysis Script for Gadget-4 Simulations
=========================================================

This script analyzes dust diagnostics output from Gadget-4 simulations,
generating visualizations of dust properties and evolution.

Features:
- Dust-to-metal ratio analysis
- Species-specific dust properties
- Environment-dependent dust analysis (temperature, density)
- Time evolution of dust properties
- Correlation with feedback mechanisms

Usage:
    python plot_dust_diagnostics.py [--feedback_file FEEDBACK_FILE]

The script analyzes dust_diagnostics.csv and optionally correlates with
feedback data if a feedback diagnostics file is provided.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as colors
import os
import argparse
from scipy import stats

# Set up argument parsing
parser = argparse.ArgumentParser(description='Analyze dust diagnostics from Gadget-4 simulations')
parser.add_argument('--feedback_file', type=str, help='Path to feedback diagnostics CSV file for correlation analysis')
args = parser.parse_args()

# Configure plot style
plt.style.use('bmh')
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

# Check if the dust diagnostics CSV file exists
dust_csv_path = "dust_diagnostics.csv"
if not os.path.exists(dust_csv_path):
    raise FileNotFoundError(f"Could not find {dust_csv_path}. Please check file path.")

# Read the dust diagnostics CSV
dust_df = pd.read_csv(
    dust_csv_path,
    comment="#",
    names=["silicate", "carbon", "iron", "dtm_ratio", "temp", "dens", "depletion"],
    engine="python",
    on_bad_lines="skip"
)

# Print basic statistics
print(f"Loaded {len(dust_df)} dust diagnostic records.")
print("\nDust Statistics:")
print(f"Mean dust-to-metal ratio: {dust_df['dtm_ratio'].mean():.3f}")
print(f"Median dust-to-metal ratio: {dust_df['dtm_ratio'].median():.3f}")
print(f"Mean depletion fraction: {dust_df['depletion'].mean():.3f}")

# Calculate derived quantities
dust_df['total_dust'] = dust_df['silicate'] + dust_df['carbon'] + dust_df['iron']
dust_df['silicate_frac'] = dust_df['silicate'] / dust_df['total_dust'].clip(lower=1e-10)
dust_df['carbon_frac'] = dust_df['carbon'] / dust_df['total_dust'].clip(lower=1e-10)
dust_df['iron_frac'] = dust_df['iron'] / dust_df['total_dust'].clip(lower=1e-10)

# Print composition statistics
print("\nDust Composition:")
print(f"Silicate fraction: {dust_df['silicate_frac'].mean():.3f}")
print(f"Carbon fraction: {dust_df['carbon_frac'].mean():.3f}")
print(f"Iron fraction: {dust_df['iron_frac'].mean():.3f}")

# Create figure with GridSpec for flexible layout
fig = plt.figure(figsize=(20, 16))
gs = GridSpec(4, 3, figure=fig)

# 1) Histogram of dust-to-metal ratios
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(dust_df['dtm_ratio'], bins=30, alpha=0.7, color='darkblue', 
         histtype='stepfilled', edgecolor='black')
ax1.set_xlabel('Dust-to-Metal Ratio')
ax1.set_ylabel('Number of Gas Particles')
ax1.set_title('Distribution of Dust-to-Metal Ratios')
ax1.grid(True, linestyle='--', alpha=0.7)
# Add vertical line for mean and median
ax1.axvline(dust_df['dtm_ratio'].mean(), color='red', linestyle='-', label=f'Mean: {dust_df["dtm_ratio"].mean():.3f}')
ax1.axvline(dust_df['dtm_ratio'].median(), color='green', linestyle='--', label=f'Median: {dust_df["dtm_ratio"].median():.3f}')
ax1.legend()

# 2) Dust composition breakdown
ax2 = fig.add_subplot(gs[0, 1])
species_avg = [dust_df['silicate_frac'].mean(), dust_df['carbon_frac'].mean(), dust_df['iron_frac'].mean()]
labels = ['Silicate', 'Carbon', 'Iron']
colors = ['brown', 'black', 'darkred']
ax2.pie(species_avg, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
ax2.set_title('Average Dust Composition by Species')

# 3) Dust-to-metal ratio vs. temperature
ax3 = fig.add_subplot(gs[0, 2])
# For temperature, using log scale and applying temp cut to focus on relevant range
temp_mask = dust_df['temp'] > 1.0  # Filter out unrealistic temperatures
sc = ax3.scatter(dust_df.loc[temp_mask, 'temp'], dust_df.loc[temp_mask, 'dtm_ratio'], 
                 c=np.log10(dust_df.loc[temp_mask, 'dens']),
                 cmap='viridis', alpha=0.6, s=5, edgecolor='none')
ax3.set_xscale('log')
ax3.set_xlabel('Temperature [K]')
ax3.set_ylabel('Dust-to-Metal Ratio')
ax3.set_title('Dust-to-Metal Ratio vs. Temperature')
ax3.grid(True, linestyle='--', alpha=0.7)
cbar = plt.colorbar(sc, ax=ax3)
cbar.set_label('log₁₀(Density [g/cm³])')

# 4) Dust species mass distribution
ax4 = fig.add_subplot(gs[1, 0])
mass_bins = np.logspace(np.log10(max(1e-10, min(dust_df['total_dust'].min(), dust_df['silicate'].min(), dust_df['carbon'].min(), dust_df['iron'].min()))),
                       np.log10(max(dust_df['total_dust'].max(), dust_df['silicate'].max(), dust_df['carbon'].max(), dust_df['iron'].max())),
                       30)
ax4.hist(dust_df['silicate'], bins=mass_bins, alpha=0.6, color='brown', label='Silicate', histtype='step', linewidth=2)
ax4.hist(dust_df['carbon'], bins=mass_bins, alpha=0.6, color='black', label='Carbon', histtype='step', linewidth=2)
ax4.hist(dust_df['iron'], bins=mass_bins, alpha=0.6, color='darkred', label='Iron', histtype='step', linewidth=2)
ax4.hist(dust_df['total_dust'], bins=mass_bins, alpha=0.6, color='blue', label='Total Dust', histtype='step', linewidth=2)
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_xlabel('Dust Mass')
ax4.set_ylabel('Number of Gas Particles')
ax4.set_title('Distribution of Dust Species Mass')
ax4.legend()
ax4.grid(True, linestyle='--', alpha=0.7)

# 5) Depletion vs. density
ax5 = fig.add_subplot(gs[1, 1])
# For density, using log scale
dens_mask = dust_df['dens'] > 0  # Filter out non-positive densities
sc = ax5.scatter(dust_df.loc[dens_mask, 'dens'], dust_df.loc[dens_mask, 'depletion'], 
                 c=dust_df.loc[dens_mask, 'temp'],
                 cmap='plasma', alpha=0.6, s=5, edgecolor='none')
ax5.set_xscale('log')
ax5.set_xlabel('Density [g/cm³]')
ax5.set_ylabel('Metal Depletion Fraction')
ax5.set_title('Metal Depletion vs. Density')
ax5.grid(True, linestyle='--', alpha=0.7)
cbar = plt.colorbar(sc, ax=ax5)
cbar.set_label('Temperature [K]')

# 6) Phase space: Temperature vs. Density colored by DTM ratio
ax6 = fig.add_subplot(gs[1, 2])
valid_mask = (dust_df['temp'] > 1.0) & (dust_df['dens'] > 0)
sc = ax6.scatter(dust_df.loc[valid_mask, 'dens'], dust_df.loc[valid_mask, 'temp'], 
                 c=dust_df.loc[valid_mask, 'dtm_ratio'],
                 cmap='viridis', alpha=0.6, s=5, edgecolor='none')
ax6.set_xscale('log')
ax6.set_yscale('log')
ax6.set_xlabel('Density [g/cm³]')
ax6.set_ylabel('Temperature [K]')
ax6.set_title('Phase Space Colored by Dust-to-Metal Ratio')
ax6.grid(True, linestyle='--', alpha=0.7)
cbar = plt.colorbar(sc, ax=ax6)
cbar.set_label('Dust-to-Metal Ratio')

# 7) Species abundance vs. total dust mass
ax7 = fig.add_subplot(gs[2, 0])
ax7.scatter(dust_df['total_dust'], dust_df['silicate_frac'], color='brown', label='Silicate', alpha=0.5, s=5)
ax7.scatter(dust_df['total_dust'], dust_df['carbon_frac'], color='black', label='Carbon', alpha=0.5, s=5)
ax7.scatter(dust_df['total_dust'], dust_df['iron_frac'], color='darkred', label='Iron', alpha=0.5, s=5)
ax7.set_xscale('log')
ax7.set_xlabel('Total Dust Mass')
ax7.set_ylabel('Species Mass Fraction')
ax7.set_title('Dust Species Fractions vs. Total Dust Mass')
ax7.legend()
ax7.grid(True, linestyle='--', alpha=0.7)

# 8 & 9) These plots will be filled conditionally if feedback data is available

# If feedback diagnostics file is provided, correlate with dust properties
if args.feedback_file and os.path.exists(args.feedback_file):
    print(f"\nLoading feedback diagnostics from: {args.feedback_file}")
    
    # Read the feedback data
    feedback_df = pd.read_csv(
        args.feedback_file,
        comment="#",
        names=["delta_u", "delta_v", "rel_inc", "r", "n_ngb", "h_star", "E_ratio", "feedback_type", "time"],
        engine="python",
        on_bad_lines="skip"
    )
    
    # Filter to get only neighbor records (gas particles that received feedback)
    neighbors = feedback_df.dropna(subset=["delta_u"])
    
    print(f"Loaded {len(neighbors)} feedback neighbor records.")
    
    # Plot 8: DTM ratio by feedback type (if feedback data available)
    ax8 = fig.add_subplot(gs[2, 1])
    
    # Create synthetic combined dataset
    # This is a simplification since we don't have direct links between feedback and dust particles
    # In real application, you'd need particle IDs to properly match them
    
    # Bin dust particles by temperature for comparison with feedback types
    temp_bins = [0, 100, 1000, 10000, 1e6, 1e10]
    temp_labels = ['<100K', '100-1000K', '1K-10K', '10K-1M', '>1M']
    dust_df['temp_bin'] = pd.cut(dust_df['temp'], bins=temp_bins, labels=temp_labels)
    
    # Calculate average DTM ratio by temperature bin
    dtm_by_temp = dust_df.groupby('temp_bin')['dtm_ratio'].mean().reset_index()
    
    # Count feedback events by type
    feedback_counts = neighbors['feedback_type'].value_counts()
    
    # Plot DTM ratio by temperature bin
    ax8.bar(dtm_by_temp['temp_bin'], dtm_by_temp['dtm_ratio'], color='skyblue', alpha=0.7)
    ax8.set_xlabel('Gas Temperature Range')
    ax8.set_ylabel('Average Dust-to-Metal Ratio')
    ax8.set_title('DTM Ratio by Gas Temperature')
    ax8.grid(True, linestyle='--', alpha=0.7)
    plt.setp(ax8.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 9: Feedback types and dust species correlation
    ax9 = fig.add_subplot(gs[2, 2])
    
    # Calculate average dust species fractions by temperature bin
    species_by_temp = dust_df.groupby('temp_bin')[['silicate_frac', 'carbon_frac', 'iron_frac']].mean().reset_index()
    
    # Plot grouped bar chart
    bar_width = 0.25
    index = np.arange(len(species_by_temp))
    
    ax9.bar(index - bar_width, species_by_temp['silicate_frac'], bar_width, 
            color='brown', label='Silicate', alpha=0.7)
    ax9.bar(index, species_by_temp['carbon_frac'], bar_width, 
            color='black', label='Carbon', alpha=0.7)
    ax9.bar(index + bar_width, species_by_temp['iron_frac'], bar_width, 
            color='darkred', label='Iron', alpha=0.7)
    
    ax9.set_xlabel('Gas Temperature Range')
    ax9.set_ylabel('Average Species Fraction')
    ax9.set_title('Dust Composition by Gas Temperature')
    ax9.set_xticks(index)
    ax9.set_xticklabels(species_by_temp['temp_bin'])
    ax9.legend()
    ax9.grid(True, linestyle='--', alpha=0.7)
    plt.setp(ax9.get_xticklabels(), rotation=45, ha='right')
    
    # Plot additional feedback-related info if available
    if len(neighbors) > 0:
        # 10) Feedback velocity kicks vs DTM ratio by temperature
        ax10 = fig.add_subplot(gs[3, 0])
        
        # Calculate average velocity kick by feedback type
        vkick_by_type = neighbors.groupby('feedback_type')['delta_v'].mean().reset_index()
        
        # Plot bar chart
        ax10.bar(vkick_by_type['feedback_type'], vkick_by_type['delta_v'], color='lightgreen', alpha=0.7)
        ax10.set_xlabel('Feedback Type')
        ax10.set_ylabel('Average Velocity Kick')
        ax10.set_title('Average Velocity Kick by Feedback Type')
        ax10.grid(True, linestyle='--', alpha=0.7)
        
        # 11) Energy input vs dust production proxy
        ax11 = fig.add_subplot(gs[3, 1])
        
        # Calculate average thermal energy by feedback type
        energy_by_type = neighbors.groupby('feedback_type')['delta_u'].mean().reset_index()
        
        # Plot bar chart
        ax11.bar(energy_by_type['feedback_type'], energy_by_type['delta_u'], color='salmon', alpha=0.7)
        ax11.set_xlabel('Feedback Type')
        ax11.set_ylabel('Average Thermal Energy Input')
        ax11.set_title('Thermal Energy Input by Feedback Type')
        ax11.grid(True, linestyle='--', alpha=0.7)
        
        # 12) Combined metric
        ax12 = fig.add_subplot(gs[3, 2])
        
        # Create a plot showing which feedback types might correlate with dust production
        # Using feedback type distribution
        ax12.pie(feedback_counts, labels=feedback_counts.index, autopct='%1.1f%%', 
                 startangle=90, colors=['blue', 'red', 'green', 'gray'])
        ax12.set_title('Distribution of Feedback Events by Type')
else:
    # If no feedback data, show alternative plots
    
    # 8) DTM ratio by density bins
    ax8 = fig.add_subplot(gs[2, 1])
    
    # Create density bins
    dens_bins = np.logspace(np.log10(max(1e-30, dust_df['dens'].min())), 
                          np.log10(dust_df['dens'].max()), 
                          6)
    dust_df['dens_bin'] = pd.cut(dust_df['dens'], bins=dens_bins)
    
    # Calculate average DTM ratio by density bin
    dtm_by_dens = dust_df.groupby('dens_bin')['dtm_ratio'].mean().reset_index()
    
    # Plot
    ax8.bar(range(len(dtm_by_dens)), dtm_by_dens['dtm_ratio'], color='skyblue', alpha=0.7)
    ax8.set_xlabel('Density Bin (increasing →)')
    ax8.set_ylabel('Average Dust-to-Metal Ratio')
    ax8.set_title('DTM Ratio by Gas Density')
    ax8.set_xticks(range(len(dtm_by_dens)))
    ax8.set_xticklabels([f'{i+1}' for i in range(len(dtm_by_dens))])
    ax8.grid(True, linestyle='--', alpha=0.7)
    
    # 9) Species composition by density bin
    ax9 = fig.add_subplot(gs[2, 2])
    
    # Calculate average species fractions by density bin
    species_by_dens = dust_df.groupby('dens_bin')[['silicate_frac', 'carbon_frac', 'iron_frac']].mean().reset_index()
    
    # Plot stacked bar chart
    bottom = np.zeros(len(species_by_dens))
    
    for species, color in zip(['silicate_frac', 'carbon_frac', 'iron_frac'], 
                             ['brown', 'black', 'darkred']):
        ax9.bar(range(len(species_by_dens)), species_by_dens[species], 
                bottom=bottom, color=color, alpha=0.7, 
                label=species.replace('_frac', '').capitalize())
        bottom += species_by_dens[species]
    
    ax9.set_xlabel('Density Bin (increasing →)')
    ax9.set_ylabel('Species Fraction')
    ax9.set_title('Dust Composition by Gas Density')
    ax9.set_xticks(range(len(species_by_dens)))
    ax9.set_xticklabels([f'{i+1}' for i in range(len(species_by_dens))])
    ax9.legend()
    ax9.grid(True, linestyle='--', alpha=0.7)
    
    # 10) DTM ratio vs. depletion correlation
    ax10 = fig.add_subplot(gs[3, 0])
    
    # Calculate correlation
    corr, p_value = stats.pearsonr(dust_df['dtm_ratio'], dust_df['depletion'])
    
    # Plot correlation
    ax10.scatter(dust_df['dtm_ratio'], dust_df['depletion'], 
                alpha=0.5, s=5, color='purple')
    ax10.set_xlabel('Dust-to-Metal Ratio')
    ax10.set_ylabel('Metal Depletion Fraction')
    ax10.set_title(f'DTM Ratio vs. Depletion (r={corr:.3f}, p={p_value:.3e})')
    ax10.grid(True, linestyle='--', alpha=0.7)
    
    # Add regression line
    slope, intercept = np.polyfit(dust_df['dtm_ratio'], dust_df['depletion'], 1)
    x = np.array([dust_df['dtm_ratio'].min(), dust_df['dtm_ratio'].max()])
    y = slope * x + intercept
    ax10.plot(x, y, 'r-', label=f'y = {slope:.3f}x + {intercept:.3f}')
    ax10.legend()
    
    # 11) Temperature distribution
    ax11 = fig.add_subplot(gs[3, 1])
    
    # Plot temperature histogram
    temp_mask = dust_df['temp'] > 1.0
    ax11.hist(dust_df.loc[temp_mask, 'temp'], bins=30, 
              alpha=0.7, color='orange', histtype='stepfilled', 
              edgecolor='black', log=True)
    ax11.set_xscale('log')
    ax11.set_xlabel('Temperature [K]')
    ax11.set_ylabel('Number of Gas Particles')
    ax11.set_title('Temperature Distribution of Gas with Dust')
    ax11.grid(True, linestyle='--', alpha=0.7)
    
    # 12) Density distribution
    ax12 = fig.add_subplot(gs[3, 2])
    
    # Plot density histogram
    dens_mask = dust_df['dens'] > 0
    ax12.hist(dust_df.loc[dens_mask, 'dens'], bins=30, 
              alpha=0.7, color='green', histtype='stepfilled', 
              edgecolor='black', log=True)
    ax12.set_xscale('log')
    ax12.set_xlabel('Density [g/cm³]')
    ax12.set_ylabel('Number of Gas Particles')
    ax12.set_title('Density Distribution of Gas with Dust')
    ax12.grid(True, linestyle='--', alpha=0.7)

# Add a title for the entire figure
fig.suptitle("Dust Diagnostics Analysis", fontsize=20, y=0.98)

# Adjust layout and save
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('dust_diagnostics_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\nAnalysis Summary:")
print(f"DTM ratio range: {dust_df['dtm_ratio'].min():.3f} - {dust_df['dtm_ratio'].max():.3f}")
print(f"Temperature range: {dust_df['temp'].min():.1f} - {dust_df['temp'].max():.1e} K")
print(f"Density range: {dust_df['dens'].min():.1e} - {dust_df['dens'].max():.1e} g/cm³")

# Save statistics to a summary file
with open('dust_analysis_summary.txt', 'w') as f:
    f.write("Dust Analysis Summary\n")
    f.write("=====================\n\n")
    f.write(f"Total gas particles analyzed: {len(dust_df)}\n\n")
    
    f.write("Dust-to-Metal Ratio Statistics:\n")
    f.write(f"  Mean: {dust_df['dtm_ratio'].mean():.3f}\n")
    f.write(f"  Median: {dust_df['dtm_ratio'].median():.3f}\n")
    f.write(f"  Min: {dust_df['dtm_ratio'].min():.3f}\n")
    f.write(f"  Max: {dust_df['dtm_ratio'].max():.3f}\n\n")
    
    f.write("Dust Composition (Average):\n")
    f.write(f"  Silicate: {dust_df['silicate_frac'].mean():.3f}\n")
    f.write(f"  Carbon: {dust_df['carbon_frac'].mean():.3f}\n")
    f.write(f"  Iron: {dust_df['iron_frac'].mean():.3f}\n\n")
    
    f.write("Environmental Conditions:\n")
    f.write(f"  Temperature range: {dust_df['temp'].min():.1f} - {dust_df['temp'].max():.1e} K\n")
    f.write(f"  Density range: {dust_df['dens'].min():.1e} - {dust_df['dens'].max():.1e} g/cm³\n\n")
    
    if args.feedback_file and os.path.exists(args.feedback_file):
        f.write("Feedback Statistics:\n")
        for fb_type, count in feedback_counts.items():
            f.write(f"  {fb_type}: {count} events\n")

print("\nAnalysis complete! Summary saved to dust_analysis_summary.txt")