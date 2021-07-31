import serial
from time import sleep
import sys

COM_PORT = 'COM5'  # need to change the port name
BAUD_RATES = 9600
ser = serial.Serial(COM_PORT, BAUD_RATES)
sol="rLDfRdzRsbrUzUzsxUqxu"
# sol="UlfbUlfb"
# sol="UxquxszuzuRBsrzDrFdlR"
# print(len(sol))
i = 0
try:
    while i<len(sol):
        if i==0:
            sleep(1.5)
        choice = sol[i]
        if choice == 'r':    #R
            print('r')
            ser.write(b'r')  # 訊息必須是位元組類型
            sleep(1.5)              # 暫停0.5秒，再執行底下接收回應訊息的迴圈
        elif choice == 'R':
            print('R')
            ser.write(b'R')
            sleep(1.5)
        elif choice == 'q':
            print('q')
            ser.write(b'q')
            sleep(1.5)
        elif choice == 'L':     #L
            print('L')
            ser.write(b'L')
            sleep(1.5)
        elif choice == 'l':
            print('l')
            ser.write(b'l')
            sleep(1.5)
        elif choice == 'a':
            print('a')
            ser.write(b'a')
            sleep(1.5)
        elif choice == 'F':     #F
            print('F')
            ser.write(b'F')
            sleep(1.5)
        elif choice == 'f':
            print('f')
            ser.write(b'f')
            sleep(1.5)
        elif choice == 'x':
            print('x')
            ser.write(b'x')
            sleep(1.5)
        elif choice == 'B':     #B    
            print('B')
            ser.write(b'B')
            sleep(1.5)
        elif choice == 'b':
            print('b')
            ser.write(b'b')
            sleep(1.5)
        elif choice == 'z':
            print('z')
            ser.write(b'z')
            sleep(1.5)
        elif choice == 'U':     #U    
            print('U')
            ser.write(b'U')
            sleep(1.5)
        elif choice == 'u':
            print('u')
            ser.write(b'u')
            sleep(1.5)
        elif choice == 'w':
            print('w')
            ser.write(b'w')
            sleep(1.5)
        elif choice == 'D':     #D    
            print('D')
            ser.write(b'D')
            sleep(1.5)
        elif choice == 'd':
            print('d')
            ser.write(b'd')
            sleep(1.5)
        elif choice == 's':
            print('s')
            ser.write(b's')
            sleep(1.5)
        else:
            print('bye!!')
            ser.close()
            sys.exit()

        # while ser.in_waiting:
        #     mcu_feedback = ser.readline().decode()  # 接收回應訊息並解碼
        #     print('控制板回應：', mcu_feedback)
        i = i + 1

except KeyboardInterrupt:
    ser.close()
    print('再見！')
