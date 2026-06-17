import serial
import json
import time
import struct
import sys

# --- CONFIGURATION ---
JSON_FILE = 'robot_path_commands.json'
UART_PORT = '/dev/ttyAMA0'  # Sadece özel (exclusive) ve izole edilmiş UART portu
BAUD_RATE = 115200

SYNC_BYTE_1 = 0xAA          
SYNC_BYTE_2 = 0xBB

def calculate_checksum(payload_bytes):
    """Calculates a simple 8-bit checksum for data integrity."""
    return sum(payload_bytes) & 0xFF

def main():
    # 1. Load the Trajectory
    try:
        with open(JSON_FILE, 'r') as f:
            raw_trajectory = json.load(f)
        print(f"[INFO] Loaded {len(raw_trajectory)} raw steps from {JSON_FILE}")
    except Exception as e:
        print(f"[ERROR] Could not load JSON: {e}")
        sys.exit(1)

    # 1.5 Stutter Filter: Remove mid-path steps where distance is 0.0 AND no angles change
    trajectory = []
    for i, cmd in enumerate(raw_trajectory):
        dist_head = float(cmd["dc_motor_commands"]["segment1_head_distance_units"])
        dist_link2 = float(cmd["dc_motor_commands"]["segment3_link2_distance_units"])
        
        q1_yaw = float(cmd["servo_yaw_commands"]["q1_deg"])
        q2_yaw = float(cmd["servo_yaw_commands"]["q2_deg"])
        q3_yaw = float(cmd["servo_yaw_commands"]["q3_deg"])

        angle_changed = False
        if trajectory:
            last_cmd = trajectory[-1]
            last_q1 = float(last_cmd["servo_yaw_commands"]["q1_deg"])
            last_q2 = float(last_cmd["servo_yaw_commands"]["q2_deg"])
            last_q3 = float(last_cmd["servo_yaw_commands"]["q3_deg"])
            if abs(q1_yaw - last_q1) > 0.1 or abs(q2_yaw - last_q2) > 0.1 or abs(q3_yaw - last_q3) > 0.1:
                angle_changed = True
        else:
            angle_changed = True # Always keep the very first step to initialize pose
        
        # Keep the step if it requires movement, OR if angle changed, OR if it's the very last step
        if dist_head != 0.0 or dist_link2 != 0.0 or angle_changed or i == (len(raw_trajectory) - 1):
            trajectory.append(cmd)
            
    total_steps = len(trajectory)
    print(f"[INFO] Filtered trajectory down to {total_steps} active steps.")

    # 2. Open UART Connection
    try:
        # KORUMA 1: exclusive=True ile Linux'un araya girmesini kesin olarak yasaklıyoruz.
        ser = serial.Serial(UART_PORT, BAUD_RATE, timeout=1, exclusive=True)
        print(f"[INFO] Connected to STM32 on {UART_PORT} at {BAUD_RATE} baud.")
        
        # Temizlik: Pi açılırken hatta birikmiş olabilecek çöpleri temizle
        ser.reset_input_buffer()
        time.sleep(1) 
    except Exception as e:
        print(f"[ERROR] Could not open UART. Port may be busy or disabled: {e}")
        sys.exit(1)

    step_index = 0
    print("[INFO] Waiting for STM32 to request the first step...")

    # 3. Main Communication Loop
    try:
        while step_index < total_steps:
            
            # KORUMA 2: Anlık kablo kopmalarına ve gürültülere karşı okuma zırhı
            try:
                incoming_bytes = ser.readline()
            except serial.SerialException as e:
                print(f"\n[UYARI] Anlık bağlantı kopması veya gürültü tespit edildi: {e}")
                print("[UYARI] Port kurtarılıyor, iletişim devam edecek...\n")
                ser.reset_input_buffer()
                time.sleep(0.1)
                continue # Hatayı yoksay ve döngüye kaldığı yerden devam et
            
            if incoming_bytes:
                # Gelen byteları metne çevirip gereksiz karakterleri atıyoruz
                incoming_str = incoming_bytes.decode('utf-8', errors='ignore').strip()
                
                # Sadece içinde "REQ" geçiyor mu diye bakıyoruz
                if "REQ" in incoming_str:
                    print(f"[TX] STM32 requested data. Sending Step {step_index + 1}/{total_steps}...")
                    
                    cmd = trajectory[step_index]
                    
                    # Verileri Çek
                    duration_ms = int(cmd["step_duration_ms"])
                    dist_head = float(cmd["dc_motor_commands"]["segment1_head_distance_units"])
                    dist_link2 = float(cmd["dc_motor_commands"]["segment3_link2_distance_units"])
                    
                    q1_yaw = float(cmd["servo_yaw_commands"]["q1_deg"])
                    q2_yaw = float(cmd["servo_yaw_commands"]["q2_deg"])
                    q3_yaw = float(cmd["servo_yaw_commands"]["q3_deg"])
                    
                    q1_pitch = float(cmd["servo_pitch_commands"]["q1_pitch_deg"])
                    q2_pitch = float(cmd["servo_pitch_commands"]["q2_pitch_deg"])
                    q3_pitch = float(cmd["servo_pitch_commands"]["q3_pitch_deg"])

                    # Binary formatta paketle (<H 8f)
                    payload = struct.pack('<H 8f', 
                                          duration_ms, 
                                          dist_head, dist_link2, 
                                          q1_yaw, q2_yaw, q3_yaw, 
                                          q1_pitch, q2_pitch, q3_pitch)

                    checksum = calculate_checksum(payload)
                    packet = bytes([SYNC_BYTE_1, SYNC_BYTE_2]) + payload + bytes([checksum])
                    
                    # KORUMA 3: Anlık kopmalara karşı yazma zırhı
                    try:
                        ser.write(packet)
                        ser.flush()
                        step_index += 1
                        
                        # KORUMA 4: Makinalı tüfek gibi hızlı gitmeyi engelleyen fren ve temizleyici
                        time.sleep(0.01)             # 10ms nefes aldırır (elektriksel oturma süresi)
                        ser.reset_input_buffer()     # Birikmiş spam REQ mesajlarını çöpe atar
                        
                    except serial.SerialException:
                        print("[HATA] Veri gönderilirken port koptu! Sistem bir sonraki REQ ile tekrar deneyecek.")
                        time.sleep(0.1)

                elif incoming_str:
                    # Gelen veri REQ değilse (örneğin STM32 bir hata mesajı basıyorsa) bunu ekrana yazdır.
                    print(f"[STM32 DEBUG] {incoming_str}")

        # Trajectory Complete
        print("[INFO] Trajectory complete. Sending END dummy packet.")
        dummy_payload = struct.pack('<H 8f', 65535, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        dummy_checksum = calculate_checksum(dummy_payload)
        dummy_packet = bytes([SYNC_BYTE_1, SYNC_BYTE_2]) + dummy_payload + bytes([dummy_checksum])
        
        try:
            ser.write(dummy_packet)
            ser.flush()
        except:
            pass # Sonda koparsa görmezden gel
            
        ser.close()

    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user. Closing port.")
        ser.close()

if __name__ == "__main__":
    main()