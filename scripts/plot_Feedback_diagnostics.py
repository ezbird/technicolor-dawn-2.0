#!/usr/bin/env python3
"""
Feedback Diagnostics with Type Separation (SNII, SNIa, AGB)
==========================================================

This script analyzes feedback diagnostics output from Gadget-4 simulations,
separating the analysis by feedback type (SNII, SNIa, AGB), which is 
particularly useful for dust models that treat each mechanism differently.

Features:
- Separate analysis for SNII, SNIa, and AGB feedback
- Enhanced visualizations with consistent color schemes
- Dust-relevant metrics including transport potential
- Energy distribution and conservation analysis
- CSV export of key metrics for further analysis

Usage:
    python plot_feedback_diagnostics.py

Requirements:
    - pandas
    - numpy
    - matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
plt.style.use('bmh')

# Check if the CSV file exists
csv_path = "../feedback_diagnostics.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Could not find {csv_path}. Please check file path.")

# Read the CSV (skipping comments)
df = pd.read_csv(
    csv_path,
    comment="#",
    names=["delta_u","delta_v","rel_inc","r","n_ngb","h_star","E_ratio","feedback_type","time"],
    engine="python",
    on_bad_lines="skip"    # skip any lines that don't match fields
)

# Coerce columns to appropriate types
numeric_cols = ["delta_u","delta_v","rel_inc","r","n_ngb","h_star","E_ratio","time"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# If feedback_type is not in the columns or all values are empty, create a default
if "feedback_type" not in df.columns or df["feedback_type"].isna().all():
    print("Warning: feedback_type column not found or empty. Using default type.")
    df["feedback_type"] = "Unknown"

# Define feedback types and their properties (colors, markers, etc.)
feedback_types = {
    "SNII": {"color": "blue", "label": "Type II Supernova", "linestyle": "-", "marker": "o"},
    "SNIa": {"color": "red", "label": "Type Ia Supernova", "linestyle": "--", "marker": "s"},
    "AGB": {"color": "green", "label": "AGB Stars", "linestyle": ":", "marker": "^"},
    "Unknown": {"color": "gray", "label": "Unknown", "linestyle": "-.", "marker": "x"}
}

# Split neighbor vs. star records
neighbors = df.dropna(subset=["delta_u"])
stars = df.dropna(subset=["n_ngb"])

# Print summary statistics
print(f"Total records: {len(df)}")
print(f"Neighbor records: {len(neighbors)}")
print(f"Star records: {len(stars)}")

# Count by feedback type
if not neighbors.empty:
    print("\nNeighbor records by feedback type:")
    neighbor_counts = neighbors["feedback_type"].value_counts()
    for fb_type, count in neighbor_counts.items():
        print(f"  {fb_type}: {count}")

if not stars.empty:
    print("\nStar events by feedback type:")
    star_counts = stars["feedback_type"].value_counts()
    for fb_type, count in star_counts.items():
        print(f"  {fb_type}: {count}")

# Create figure with GridSpec for more flexible layout
fig = plt.figure(figsize=(20, 16))
gs = GridSpec(4, 3, figure=fig)

# 1) Histogram of Δu (thermal energy kicks) by feedback type
ax1 = fig.add_subplot(gs[0, 0])
for fb_type, props in feedback_types.items():
    subset = neighbors[neighbors["feedback_type"] == fb_type]
    if len(subset) > 0:
        ax1.hist(subset["delta_u"], bins=30, alpha=0.6, color=props["color"], 
                 label=props["label"], log=True, histtype='step', linewidth=2,
                 linestyle=props["linestyle"])
ax1.set_xscale("log")
ax1.set_xlabel("Δu (post-feedback specific internal energy)")
ax1.set_ylabel("Count")
ax1.set_title("Δ Thermal Energy Injection by Feedback Type")
ax1.legend()

# 2) Histogram of Δv (velocity kicks) by feedback type
ax2 = fig.add_subplot(gs[0, 1])
for fb_type, props in feedback_types.items():
    subset = neighbors[neighbors["feedback_type"] == fb_type]
    if len(subset) > 0:
        ax2.hist(subset["delta_v"], bins=30, alpha=0.6, color=props["color"], 
                 label=props["label"], log=True, histtype='step', linewidth=2,
                 linestyle=props["linestyle"])
ax2.set_xscale("log")
ax2.set_xlabel("Δv (feedback velocity kick)")
ax2.set_ylabel("Count")
ax2.set_title("Velocity Kick Distribution by Feedback Type")
ax2.legend()

# 3) Relative energy increase vs. distance by feedback type
ax3 = fig.add_subplot(gs[0, 2])
for fb_type, props in feedback_types.items():
    subset = neighbors[neighbors["feedback_type"] == fb_type]
    if len(subset) > 0:
        ax3.scatter(subset["r"], subset["rel_inc"], s=5, alpha=0.5, 
                    color=props["color"], label=props["label"], marker=props["marker"])
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_xlabel("Distance r [kpc]")
ax3.set_ylabel("Δu / u_before")
ax3.set_title("Fractional Thermal Energy Increase vs Distance")
ax3.legend()

# 4) Histogram of neighbor-counts by feedback type
ax4 = fig.add_subplot(gs[1, 0])
for fb_type, props in feedback_types.items():
    subset = stars[stars["feedback_type"] == fb_type]
    if len(subset) > 0:
        ax4.hist(subset["n_ngb"], bins=20, alpha=0.6, color=props["color"], 
                 label=props["label"], histtype='step', linewidth=2,
                 linestyle=props["linestyle"])
ax4.set_xlabel("Neighbors per Event")
ax4.set_ylabel("Events")
ax4.set_title("Number of Neighbors Receiving Feedback")
ax4.legend()

# 5) Histogram of feedback kernel radii by feedback type
ax5 = fig.add_subplot(gs[1, 1])
for fb_type, props in feedback_types.items():
    subset = stars[stars["feedback_type"] == fb_type]
    if len(subset) > 0:
        ax5.hist(subset["h_star"], bins=20, alpha=0.6, color=props["color"], 
                 label=props["label"], histtype='step', linewidth=2,
                 linestyle=props["linestyle"])
ax5.set_xlabel("Kernel Radius h [kpc]")
ax5.set_ylabel("Events")
ax5.set_title("Feedback Kernel Radii Distribution")
ax5.legend()

# 6) Energy conservation ratio per star by feedback type
ax6 = fig.add_subplot(gs[1, 2])
for fb_type, props in feedback_types.items():
    subset = stars[stars["feedback_type"] == fb_type]
    if len(subset) > 0:
        ax6.hist(subset["E_ratio"], bins=20, alpha=0.6, color=props["color"], 
                 label=props["label"], histtype='step', linewidth=2,
                 linestyle=props["linestyle"])
ax6.set_xlabel("E_applied / E_input")
ax6.set_ylabel("Counts")
ax6.set_title("Energy Conservation Ratio by Feedback Type")
ax6.legend()

# 7) Cumulative energy distribution by radius
ax7 = fig.add_subplot(gs[2, 0])
for fb_type, props in feedback_types.items():
    subset = neighbors[neighbors["feedback_type"] == fb_type]
    if len(subset) > 0:
        # Sort by distance
        subset = subset.sort_values(by="r")
        # Calculate cumulative energy
        subset = subset.copy()  # Avoid SettingWithCopyWarning
        subset["energy"] = subset["delta_u"]
        cumul_energy = subset["energy"].cumsum() / subset["energy"].sum()
        ax7.plot(subset["r"], cumul_energy, color=props["color"], label=props["label"],
                 linestyle=props["linestyle"], marker=props["marker"], markevery=max(1, len(subset)//20))
ax7.set_xscale("log")
ax7.set_xlabel("Distance r [kpc]")
ax7.set_ylabel("Cumulative Energy Fraction")
ax7.set_title("Cumulative Energy Distribution by Radius")
ax7.grid(True, which="both", linestyle="--", linewidth=0.5)
ax7.legend()

# 8) Feedback events vs simulation time
ax8 = fig.add_subplot(gs[2, 1])
if "time" in df.columns and not stars["time"].isna().all():
    for fb_type, props in feedback_types.items():
        subset = stars[stars["feedback_type"] == fb_type]
        if len(subset) > 0 and not subset["time"].isna().all():
            ax8.hist(subset["time"], bins=20, alpha=0.6, color=props["color"], 
                     label=props["label"], histtype='step', linewidth=2,
                     linestyle=props["linestyle"])
    ax8.set_xlabel("Simulation Time")
    ax8.set_ylabel("Number of Events")
    ax8.set_title("Feedback Events Over Simulation Time")
    ax8.legend()
else:
    # Alternative: Distribution of velocity vs thermal energy
    for fb_type, props in feedback_types.items():
        subset = neighbors[neighbors["feedback_type"] == fb_type]
        if len(subset) > 0:
            ax8.scatter(subset["delta_u"], subset["delta_v"], s=5, alpha=0.5, 
                        color=props["color"], label=props["label"], marker=props["marker"])
    ax8.set_xscale("log")
    ax8.set_yscale("log")
    ax8.set_xlabel("Δu (thermal energy)")
    ax8.set_ylabel("Δv (velocity kick)")
    ax8.set_title("Correlation Between Thermal and Kinetic Feedback")
    ax8.legend()

# 9) Energy partition (thermal vs. kinetic) by distance
ax9 = fig.add_subplot(gs[2, 2])
for fb_type, props in feedback_types.items():
    subset = neighbors[neighbors["feedback_type"] == fb_type]
    if len(subset) > 0:
        # Calculate the ratio of kinetic to thermal energy
        subset = subset.copy()  # Avoid SettingWithCopyWarning
        subset["e_ratio"] = (subset["delta_v"]**2) / (2 * subset["delta_u"])
        
        # Bin by radius
        r_bins = np.logspace(np.log10(subset["r"].min()), np.log10(subset["r"].max()), 10)
        bin_centers = np.sqrt(r_bins[:-1] * r_bins[1:])
        binned_ratios = []
        
        for i in range(len(r_bins)-1):
            mask = (subset["r"] >= r_bins[i]) & (subset["r"] < r_bins[i+1])
            if mask.sum() > 0:
                binned_ratios.append(subset.loc[mask, "e_ratio"].mean())
            else:
                binned_ratios.append(np.nan)
        
        # Plot the mean ratio vs. radius
        valid_idx = ~np.isnan(binned_ratios)
        if np.any(valid_idx):
            ax9.plot(bin_centers[valid_idx], np.array(binned_ratios)[valid_idx], 
                    linestyle=props["linestyle"], marker=props["marker"], color=props["color"],
                    label=props["label"])

ax9.set_xscale("log")
ax9.set_xlabel("Distance r [kpc]")
ax9.set_ylabel("Ekinetic / Ethermal Ratio")
ax9.set_title("Energy Partition vs. Distance")
ax9.grid(True, which="both", linestyle="--", linewidth=0.5)
ax9.legend()

# 10) Velocity component distribution (for dust transport analysis)
ax10 = fig.add_subplot(gs[3, 0])
# Create a composite metric for dust transport potential: v_kick * r
for fb_type, props in feedback_types.items():
    subset = neighbors[neighbors["feedback_type"] == fb_type]
    if len(subset) > 0:
        subset = subset.copy()  # Avoid SettingWithCopyWarning
        subset["transport_potential"] = subset["delta_v"] * subset["r"]
        ax10.hist(subset["transport_potential"], bins=30, alpha=0.6, 
                 color=props["color"], label=props["label"], 
                 histtype='step', linewidth=2, linestyle=props["linestyle"])
ax10.set_xscale("log")
ax10.set_xlabel("Dust Transport Potential (v × r)")
ax10.set_ylabel("Count")
ax10.set_title("Dust Transport Potential Distribution")
ax10.legend()

# 11) Distance distribution by feedback type
ax11 = fig.add_subplot(gs[3, 1])
for fb_type, props in feedback_types.items():
    subset = neighbors[neighbors["feedback_type"] == fb_type]
    if len(subset) > 0:
        ax11.hist(subset["r"], bins=30, alpha=0.6, 
                 color=props["color"], label=props["label"], 
                 histtype='step', linewidth=2, linestyle=props["linestyle"])
ax11.set_xlabel("Distance r [kpc]")
ax11.set_ylabel("Count")
ax11.set_title("Distance Distribution by Feedback Type")
ax11.legend()

# 12) Feedback efficiency summary
ax12 = fig.add_subplot(gs[3, 2])
fb_names = []
avg_energies = []
avg_velocities = []
event_counts = []
transport_potentials = []

for fb_type, props in feedback_types.items():
    subset_n = neighbors[neighbors["feedback_type"] == fb_type]
    subset_s = stars[stars["feedback_type"] == fb_type]
    
    if len(subset_n) > 0 and len(subset_s) > 0:
        fb_names.append(fb_type)
        avg_energies.append(subset_n["delta_u"].mean())
        avg_velocities.append(subset_n["delta_v"].mean())
        event_counts.append(len(subset_s))
        
        # Calculate dust transport potential
        transport_pot = (subset_n["delta_v"] * subset_n["r"]).mean()
        transport_potentials.append(transport_pot)

x = np.arange(len(fb_names))
width = 0.35

# Create grouped bar chart
if len(fb_names) > 0:
    # Normalize values for better visualization
    max_energy = max(avg_energies) if avg_energies else 1
    max_velocity = max(avg_velocities) if avg_velocities else 1
    max_transport = max(transport_potentials) if transport_potentials else 1
    
    norm_energies = [e/max_energy for e in avg_energies]
    norm_velocities = [v/max_velocity for v in avg_velocities]
    norm_transport = [t/max_transport for t in transport_potentials]
    
    # Plot bars
    bar_width = 0.25
    ax12.bar(x - bar_width, norm_energies, bar_width, label='Rel. Thermal Energy', 
            color='lightblue', edgecolor='blue')
    ax12.bar(x, norm_velocities, bar_width, label='Rel. Velocity Kick', 
            color='lightgreen', edgecolor='green')
    ax12.bar(x + bar_width, norm_transport, bar_width, label='Rel. Transport Potential', 
            color='lightsalmon', edgecolor='red')
    
    # Add labels
    ax12.set_xlabel('Feedback Type')
    ax12.set_ylabel('Relative Value (normalized)')
    ax12.set_title('Feedback Efficiency Comparison')
    ax12.set_xticks(x)
    ax12.set_xticklabels(fb_names)
    ax12.legend()
    
    # Add event count labels
    for i, count in enumerate(event_counts):
        ax12.annotate(f'{count} events', 
                    xy=(x[i], 0),
                    xytext=(0, -20),
                    textcoords='offset points',
                    ha='center')
    
    # Add actual values as text
    for i in range(len(fb_names)):
        ax12.annotate(f'E:{avg_energies[i]:.2e}', 
                    xy=(x[i] - bar_width, norm_energies[i]),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=8)
        ax12.annotate(f'v:{avg_velocities[i]:.2e}', 
                    xy=(x[i], norm_velocities[i]),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=8)
        ax12.annotate(f'TP:{transport_potentials[i]:.2e}', 
                    xy=(x[i] + bar_width, norm_transport[i]),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=8)

else:
    ax12.text(0.5, 0.5, "No data available for comparison", 
            horizontalalignment='center', verticalalignment='center',
            transform=ax12.transAxes)

# Add a title for the entire figure
fig.suptitle("Feedback Diagnostics by Type (SNII, SNIa, AGB)", fontsize=20, y=0.98)

# Adjust layout and save
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('feedback_diagnostics_by_type.png', dpi=300, bbox_inches='tight')
plt.show()

# Print dust-relevant summary statistics
print("\n=== Dust-Relevant Statistics ===")
for fb_type in feedback_types:
    subset = neighbors[neighbors["feedback_type"] == fb_type]
    if len(subset) > 0:
        mean_energy = subset["delta_u"].mean()
        mean_velocity = subset["delta_v"].mean()
        mean_distance = subset["r"].mean()
        transport_potential = (subset["delta_v"] * subset["r"]).mean()
        
        print(f"\n{fb_type} Feedback:")
        print(f"  Mean thermal energy: {mean_energy:.3e}")
        print(f"  Mean velocity kick: {mean_velocity:.3f}")
        print(f"  Mean distance: {mean_distance:.3f} kpc")
        print(f"  Dust transport potential (v*r): {transport_potential:.3e}")
        
        # Calculate distances at which certain fractions of energy are deposited
        r_sorted = subset.sort_values(by="r")
        energy_cumsum = r_sorted["delta_u"].cumsum() / r_sorted["delta_u"].sum()
        
        # Find radius containing 50% and 90% of energy
        r50_idx = (energy_cumsum >= 0.5).idxmax() if not energy_cumsum.empty else None
        r90_idx = (energy_cumsum >= 0.9).idxmax() if not energy_cumsum.empty else None
        
        if r50_idx is not None and r90_idx is not None:
            r50 = r_sorted.loc[r50_idx, "r"]
            r90 = r_sorted.loc[r90_idx, "r"]
            print(f"  R50 (50% energy radius): {r50:.3f} kpc")
            print(f"  R90 (90% energy radius): {r90:.3f} kpc")
            
        # If we have timing information
        if "time" in subset.columns and not subset["time"].isna().all():
            time_range = subset["time"].max() - subset["time"].min()
            print(f"  Time span of events: {time_range:.3f} code units")

# Export key metrics to a CSV for further analysis
print("\nExporting metrics summary to 'feedback_metrics_summary.csv'")

# Create a summary DataFrame
summary_data = []
for fb_type in feedback_types:
    subset_n = neighbors[neighbors["feedback_type"] == fb_type]
    subset_s = stars[stars["feedback_type"] == fb_type]
    
    if len(subset_n) > 0 and len(subset_s) > 0:
        # Calculate distances at which certain fractions of energy are deposited
        r_sorted = subset_n.sort_values(by="r")
        energy_cumsum = r_sorted["delta_u"].cumsum() / r_sorted["delta_u"].sum()
        
        # Find radius containing 50% and 90% of energy
        r50_idx = (energy_cumsum >= 0.5).idxmax() if not energy_cumsum.empty else None
        r90_idx = (energy_cumsum >= 0.9).idxmax() if not energy_cumsum.empty else None
        
        r50 = r_sorted.loc[r50_idx, "r"] if r50_idx is not None else np.nan
        r90 = r_sorted.loc[r90_idx, "r"] if r90_idx is not None else np.nan
        
        # Collect statistics
        summary_data.append({
            'feedback_type': fb_type,
            'events': len(subset_s),
            'affected_particles': len(subset_n),
            'mean_thermal_energy': subset_n["delta_u"].mean(),
            'mean_velocity': subset_n["delta_v"].mean(),
            'mean_distance': subset_n["r"].mean(),
            'transport_potential': (subset_n["delta_v"] * subset_n["r"]).mean(),
            'r50_energy': r50,
            'r90_energy': r90,
            'energy_conservation_ratio': subset_s["E_ratio"].mean(),
            'mean_neighbors': subset_s["n_ngb"].mean(),
            'mean_kernel_radius': subset_s["h_star"].mean()
        })

# Create and save the summary DataFrame
if summary_data:
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('feedback_metrics_summary.csv', index=False)
    print("Summary exported successfully!")
else:
    print("No data available for summary export.")

print("\nAnalysis complete!")