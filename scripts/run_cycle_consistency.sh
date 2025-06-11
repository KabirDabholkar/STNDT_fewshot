#!/bin/bash

# Array of variants to analyze
variants=("mc_maze_20" "mc_rtt_20" "area2_bump_20" "dmfc_rsg_20")

# Loop through each variant
for variant in "${variants[@]}"; do
    echo "Running cycle consistency analysis for $variant"
    python src/validate_nlb_fewshot.py "$variant" --mode cycle_consistency
    echo "Completed analysis for $variant"
    echo "----------------------------------------"
done

echo "All cycle consistency analyses completed!" 