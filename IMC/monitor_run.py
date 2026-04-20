import time
import os
import subprocess

print("Monitoring MarshakWave_FC_vs_CF.py execution...")
print("Press Ctrl+C to stop monitoring (script will continue running)\n")

last_size = 0
while True:
    # Check if process is running
    result = subprocess.run(['pgrep', '-f', 'MarshakWave_FC_vs_CF'], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Process finished!")
        break
    
    # Check output file
    if os.path.exists('marshak_run.out'):
        size = os.path.getsize('marshak_run.out')
        if size > last_size:
            # Get last 3 lines
            with open('marshak_run.out', 'r') as f:
                lines = f.readlines()
                if lines:
                    last_lines = lines[-3:]
                    print(f"Size: {size} bytes, Last update:")
                    for line in last_lines:
                        print("  ", line.rstrip())
            last_size = size
    
    time.sleep(30)

# Check for output files
print("\nChecking for output PDFs...")
import glob
pdfs = sorted(glob.glob('marshak_wave_*_*temp.pdf'))
if pdfs:
    print(f"Found {len(pdfs)} PDF files:")
    for pdf in pdfs:
        print(f"  - {pdf}")
else:
    print("No PDF files found yet")
    print("\nLast few lines of output:")
    if os.path.exists('marshak_run.out'):
        with open('marshak_run.out', 'r') as f:
            lines = f.readlines()
            for line in lines[-20:]:
                print(line.rstrip())
