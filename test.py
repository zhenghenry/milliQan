from periphery import I2C
from time import sleep
import sys
import os

i2c = I2C('/dev/i2c-6')
def run(i2c,time,steps,Filename):
	f = open(Filename + "_input.txt", "w+")
	time = float(time)
	volt = 0x0000
	while(int(volt) < 26214):
		i2c.transfer(0x11,[I2C.Message([0x3F,volt>>8, volt & 0xFF00>>8], read = False)])
		volt = volt + steps
		os.system("cat /sys/bus/iio/devices/iio:device0/in_voltage8_vpvn_raw >>" + Filename + "_output.txt")
		f.write('%0.2f\n'%(float(volt)))
		f.flush()
		os.fsync(f.fileno())
		sleep(time)
		if(int(volt) > 26214 - steps - 1):
			while(int(volt)>0):
				volt = volt - steps
				i2c.transfer(0x11,[I2C.Message([0x3F,volt>>8, volt & 0xFF00>>8], read = False)])
				print('%.2f' % (float(volt)))
				os.system("cat /sys/bus/iio/devices/iio:device0/in_voltage8_vpvn_raw >>" + Filename + "_output.txt")
				f.write('%0.2f\n'%(float(volt)))
				f.flush()
				os.fsync(f.fileno())
				sleep(time)

Time = sys.argv[1]
Time = float(Time)
Steps = int(sys.argv[2])
file_name = sys.argv[3]
file_list = os.listdir(".")
txt_list = []
for names in file_list:
	if names.endswith(".txt"):
		txt_list.append(names)
if file_name + "_output.txt" in txt_list:
	raise Exception("File already exists!")
while(True):
	run(i2c,Time,Steps,file_name)
