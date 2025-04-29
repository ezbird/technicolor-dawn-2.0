#!/bin/bash

start_total=$(date +%s)

# reset everything
echo "[runPlots] Resetting output directory..."
start=$(date +%s)
rm -r output
end=$(date +%s)
echo "[runPlots] Output reset took $((end - start)) seconds."

# run Gadget
echo "[runGadget] Making frames of all particle types..."
start=$(date +%s)
python3 plotOutputFrames.py
end=$(date +%s)
echo "[runGadget] Gadget run took $((end - start)) seconds."

end_total=$(date +%s)
echo "[runGadget] Total script runtime: $((end_total - start_total)) seconds."
