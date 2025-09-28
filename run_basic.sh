#!/bin/bash
# Basic coverage mission with default parameters

echo "Running basic RFICP terrain analysis..."
python ../src/RFICP.py ../data/terrain.pcd \
    --line-spacing 0.5 \
    --altitude 2.0 \
    --visualize combined

echo "Analysis complete!"