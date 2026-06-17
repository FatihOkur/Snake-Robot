import serial
import time

# Pi 5 üzerindeki tüm olası UART portlarının listesi
ports_to_try = [
    '/dev/serial0', 
    '/dev/ttyAMA0', 
    '/dev/ttyAMA10',
    '/dev/ttyS0'
]

opened_ports = {}

print("--- UART PORT AVCISI ---")
for p in ports_to_try:
    try:
        # Portu açmayı dene
        ser = serial.Serial(p, 115200, timeout=0.1)
        opened_ports[p] = ser
        print(f"[OK] Port açıldı: {p}")
    except Exception as e:
        print(f"[HATA] Açılamadı: {p}")

if not opened_ports:
    print("\nHiçbir port açılamadı! Kabloları veya boot ayarlarını kontrol edin.")
    exit()

print("\n[DİNLENİYOR] Açılan tüm portlar dinleniyor...")
print("Lütfen şimdi STM32 üzerindeki SİYAH RESET TUŞUNA basın.\n")

try:
    while True:
        # Açık olan tüm portları sırayla kontrol et
        for name, ser in opened_ports.items():
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)
                print(f"!!! BİNGO !!! Veri şu porttan geldi -> {name}")
                print(f"Gelen Veri: {data}\n")
                
except KeyboardInterrupt:
    print("\nÇıkış yapıldı.")