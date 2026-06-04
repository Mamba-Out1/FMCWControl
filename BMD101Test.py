import serial

ser = serial.Serial("COM5",57600)

while True:
    print(ser.readline())