#!/bin/bash
# Advanced high-resolution coverage with custom RFICP parameters

echo "Running advanced RFICP analysis with custom parameters..."
python ../src/RFICP.py ../data/terrain.pcd \
    --downsample 50000 \
    --downsample-method voxel \
    --poisson-depth 10 \
    --line-spacing 0.2 \
    --point-spacing 0.05 \
    --altitude 1.0 \
    --aridity-resolution 200 \
    --rficp-k 1.5 \
    --rficp-c-target 0.15 \
    --rficp-lambda 1.2 \
    --rficp-R 2.0 \
    --rficp-sigma 0.8 \
    --v-min 0.3 \
    --v-max 3.0 \
    --visualize all

echo "Advanced analysis complete!"