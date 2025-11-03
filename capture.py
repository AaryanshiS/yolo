import os
import subprocess
import time

import pandas as pd
import pyshark

OUTPUT_FILE = "live130925.csv"
INTERFACE = "Wi-Fi"  # change to your interface

print(f"🚀 Starting capture on {INTERFACE}. Writing to {OUTPUT_FILE}")

# Remove old file if exists
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

cap = pyshark.LiveCapture(interface=INTERFACE, display_filter="ip")

packets = []


# --- Auto-start dashboard after short delay ---
def launch_dashboard():
    time.sleep(5)  # wait 5 seconds before starting dashboard
    print("📊 Launching dashboard...")
    subprocess.Popen(["python", "dashlogger.py"])  # run dashboard in parallel


# Launch dashboard in background
import threading

threading.Thread(target=launch_dashboard, daemon=True).start()

# --- Capture loop ---
for packet in cap.sniff_continuously():
    try:
        # Common fields
        timestamp = packet.sniff_time.timestamp()
        length = int(packet.length)
        protocol = packet.highest_layer
        src = getattr(packet.ip, "src", "N/A")
        dst = getattr(packet.ip, "dst", "N/A")

        # Append row
        packets.append(
            {
                "Time": timestamp,
                "Source": src,
                "Destination": dst,
                "Protocol": protocol,
                "Length": length,
            }
        )

        # Write every 20 packets
        if len(packets) >= 20:
            df = pd.DataFrame(packets)
            if os.path.exists(OUTPUT_FILE):
                df.to_csv(OUTPUT_FILE, mode="a", header=False, index=False)
            else:
                df.to_csv(OUTPUT_FILE, index=False)
            packets = []
            print(f"✅ Wrote 20 packets at {time.strftime('%H:%M:%S')}")

    except Exception:
        continue
