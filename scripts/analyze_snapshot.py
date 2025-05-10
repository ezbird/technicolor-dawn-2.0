import argparse
import h5py
import numpy as np

def analyze_snapshot(path, floor=None):
    # Load snapshot
    with h5py.File(path, 'r') as f:
        gas = f['PartType0']
        utherm = gas['InternalEnergy'][:]
        density = gas['Density'][:]
    
    total = utherm.size
    nan_mask = ~np.isfinite(utherm)
    nan_count = nan_mask.sum()

    # Floor analysis if provided
    floor_count = None
    if floor is not None:
        floor_count = (utherm <= floor).sum()

    # Compute basic statistics for utherm and density
    def stats(arr):
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return {
                'min': np.nan, 'p1': np.nan, 'p25': np.nan, 'median': np.nan,
                'p75': np.nan, 'p99': np.nan, 'max': np.nan,
                'mean': np.nan, 'std': np.nan
            }
        return {
            'min': valid.min(),
            'p1': np.percentile(valid, 1),
            'p25': np.percentile(valid, 25),
            'median': np.median(valid),
            'p75': np.percentile(valid, 75),
            'p99': np.percentile(valid, 99),
            'max': valid.max(),
            'mean': valid.mean(),
            'std': valid.std()
        }

    u_stats = stats(utherm)
    rho_stats = stats(density)

    # Print summary
    print(f"Analyzing snapshot: {path}\n")
    print(f"Total gas particles: {total}")
    print(f"NaNs in InternalEnergy: {nan_count} ({nan_count/total:.2%})")
    if floor_count is not None:
        print(f"u <= floor ({floor:.3e}): {floor_count} ({floor_count/total:.2%})")
    print("\nInternalEnergy statistics (code units):")
    print(f"  min={u_stats['min']:.3e}, 1%={u_stats['p1']:.3e}, 25%={u_stats['p25']:.3e}, "
          f"median={u_stats['median']:.3e}, 75%={u_stats['p75']:.3e}, 99%={u_stats['p99']:.3e}, max={u_stats['max']:.3e}")
    print(f"  mean={u_stats['mean']:.3e}, std={u_stats['std']:.3e}")

    print("\nDensity statistics (code units):")
    print(f"  min={rho_stats['min']:.3e}, 1%={rho_stats['p1']:.3e}, 25%={rho_stats['p25']:.3e}, "
          f"median={rho_stats['median']:.3e}, 75%={rho_stats['p75']:.3e}, 99%={rho_stats['p99']:.3e}, max={rho_stats['max']:.3e}")
    print(f"  mean={rho_stats['mean']:.3e}, std={rho_stats['std']:.3e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Gadget-4 snapshot gas properties")
    parser.add_argument("snapshot", help="Path to the HDF5 snapshot file")
    parser.add_argument("--floor", type=float,
                        help="Code-unit threshold to measure utherm floor fraction")
    args = parser.parse_args()
    analyze_snapshot(args.snapshot, args.floor)
