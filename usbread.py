import serial
import time
import pickle

x = -54
y = -71
it = 1
name = "new_" + str(x) + "_" + str(y) + "_" + str(it) + ".dat"

ser = serial.Serial(port = "COM3", baudrate=20000000,
                           bytesize=8, timeout=1, stopbits=serial.STOPBITS_ONE)
master_array = []

def merge(buffer):
    buffer = buffer[:-2]
    seg = ([int(i) for i in buffer.split(' ')])
    master_array.append(seg)



print("go")

start = time.time()
read = ser.read(1).decode("utf-8")
for i in range(20000):
    buffer = ""
    if read == "<" or buffer[-1] == "<":
        buffer = ser.read(16).decode("utf-8")
        merge(buffer)
        #print(buffer)
        
    else:
        read = ser.read(1).decode("utf-8")

master_array.append([x,y])

pickle.dump(master_array, open(name, 'wb'))

print(start - time.time())
print(len(master_array))
print(master_array[0])
print(master_array[1])