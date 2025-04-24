#!/bin/bash

which_snapshot=$1
start_total=$(date +%s)

if [ -z "$which_snapshot" ]; then
    echo "[runGadget] ERROR: Please provide a snapshot number!"
    echo "Usage: ./runGadget.sh 30"
    exit 1
fi

# get latest from repository
echo "[runGadget] Pulling latest from repository..."
start=$(date +%s)
git pull origin
end=$(date +%s)
echo "[runGadget] Git pull took $((end - start)) seconds."

# build
echo "[runGadget] Cleaning and building..."
start=$(date +%s)
make clean
make -j4
end=$(date +%s)
echo "[runGadget] Build took $((end - start)) seconds."

# run Gadget
echo "[runGadget] Running Gadget from snapshot $which_snapshot..."
start=$(date +%s)
mpirun -np 4 ./Gadget4 param.txt 2 "$which_snapshot" | tee output.log
end=$(date +%s)
echo "[runGadget] Gadget run took $((end - start)) seconds."

# make output animations
echo "[runGadget] Generating output animations..."
start=$(date +%s)
#python3 plotOutputAnimation.py
#python3 plotFeedbackMap.py
end=$(date +%s)
echo "[runGadget] Animation scripts took $((end - start)) seconds."

end_total=$(date +%s)
echo "[runGadget] Total script runtime: $((end_total - start_total)) seconds."
