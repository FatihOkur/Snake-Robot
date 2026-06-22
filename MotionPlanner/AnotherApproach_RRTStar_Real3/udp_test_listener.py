import socket
import json

# Must match the settings in threeD_measure.py
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

# Set up the UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"[*] UDP Listener Active. Waiting for Checkpoint data on {UDP_IP}:{UDP_PORT}...")
print("[*] Press Ctrl+C to stop.\n")

try:
    while True:
        data, addr = sock.recvfrom(1024) # 1024 byte buffer
        try:
            # Decode the raw bytes into a string, then parse the JSON
            message = data.decode("utf-8")
            payload = json.loads(message)
            
            print(f"[SUCCESS] Checkpoint Received from {addr}:")
            print(json.dumps(payload, indent=4))
            print("-" * 40)
            
        except json.JSONDecodeError:
            print(f"[WARN] Received non-JSON data: {data}")
            
except KeyboardInterrupt:
    print("\n[*] Listener shut down by user.")
    sock.close()