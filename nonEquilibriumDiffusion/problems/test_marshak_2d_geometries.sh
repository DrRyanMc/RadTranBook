#!/bin/bash

echo "Testing Marshak Wave 2D with different geometries"
echo "================================================="

echo ""
echo "1. Testing Cartesian geometry (x-y)..."
/usr/local/bin/python3 marshak_wave_multigroup_2d.py --geometry cartesian --nx 20 --ny 4 > /dev/null 2>&1 &
PID1=$!

# Wait a bit and check if process started
sleep 2
if ps -p $PID1 > /dev/null; then
    echo "   ✓ Cartesian simulation started (PID: $PID1)"
    echo "   Simulation running in background..."
else
    echo "   ✗ Cartesian simulation failed to start"
fi

echo ""
echo "2. Testing Cylindrical geometry (r-z)..."
/usr/local/bin/python3 marshak_wave_multigroup_2d.py --geometry cylindrical --nx 20 --ny 4 > /dev/null 2>&1 &
PID2=$!

# Wait a bit and check if process started
sleep 2
if ps -p $PID2 > /dev/null; then
    echo "   ✓ Cylindrical simulation started (PID: $PID2)"
    echo "   Simulation running in background..."
else
    echo "   ✗ Cylindrical simulation failed to start"
fi

echo ""
echo "Setup complete. Both simulations are running."
echo "Expected outputs:"  
echo "  - marshak_wave_2d_comparison_cartesian.png"
echo "  - marshak_wave_2d_comparison_cylindrical.png"
echo ""
echo "Processes:"
echo "  Cartesian: $PID1"
echo "  Cylindrical: $PID2"
