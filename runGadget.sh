#!/bin/bash

start_total=$(date +%s)

# reset everything
echo "[runGadget] Resetting output directory..."
start=$(date +%s)
rm -rf output
end=$(date +%s)
echo "[runGadget] Output reset took $((end - start)) seconds."

# get latest from repository
echo "[runGadget] Pulling latest from repository..."
start=$(date +%s)
git pull origin
end=$(date +%s)
echo "[runGadget] Git pull took $((end - start)) seconds."

# build with maximum parallelism
echo "[runGadget] Cleaning and building..."
start=$(date +%s)
make clean
make -j24  # Use all 24 cores for compilation
end=$(date +%s)
echo "[runGadget] Build took $((end - start)) seconds."

# Check available memory and CPU info
echo "[runGadget] System info:"
echo "  Available cores: $(nproc)"
echo "  Available memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo "  Load average: $(uptime | awk -F'load average:' '{print $2}')"

# run Gadget with optimal MPI configuration
echo "[runGadget] Running Gadget with 20 cores (leaving 4 for system)..."
start=$(date +%s)

# Optimal MPI settings for performance
export OMP_NUM_THREADS=1  # Disable OpenMP threading to avoid conflicts
export I_MPI_PIN_DOMAIN=auto  # Let MPI handle CPU affinity
export I_MPI_FABRICS=shm:ofi  # Use shared memory + libfabric for fast local communication

# Run with 20 cores, leaving 4 for system overhead and I/O
mpirun -np 20 \
    --bind-to core \
    --map-by core \
    --report-bindings \
    ./Gadget4 param.txt | tee output.log

end=$(date +%s)
echo "[runGadget] Gadget run took $((end - start)) seconds."

# make output animations in parallel
echo "[runGadget] Generating output animations in parallel..."
start=$(date +%s)

# Run plotting scripts in parallel using background processes
python3 plotOutputFrames.py &
PID1=$!

python3 plot_Rho_vs_T.py &
PID2=$!

python3 plotFeedbackMap.py &
PID3=$!

# Wait for all plotting to finish
wait $PID1
echo "[runGadget] plotOutputFrames.py completed"

wait $PID2  
echo "[runGadget] plot_Rho_vs_T.py completed"

wait $PID3
echo "[runGadget] plotFeedbackMap.py completed"

end=$(date +%s)
echo "[runGadget] All animation scripts took $((end - start)) seconds."

# Compress output for storage efficiency
echo "[runGadget] Compressing large output files..."
start=$(date +%s)
find output -name "*.hdf5" -size +100M -exec gzip {} \; &
COMPRESS_PID=$!

end_total=$(date +%s)
echo "[runGadget] Total script runtime: $((end_total - start_total)) seconds."

# Optional: wait for compression to finish in background
echo "[runGadget] Compression running in background (PID: $COMPRESS_PID)"