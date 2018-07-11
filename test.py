from periphery import I2C
from time import sleep
import sys

i2c = I2C('/dev/i2c-6')
def run(i2c,time,steps):
	time = float(time)
	volt = 0x0000
	while(int(volt) < 2**16):
		i2c.transfer(0x20,[I2C.Message([0x3F,volt>>8, volt & 0xFF00>>8], read = False)])
		volt = volt + steps
		print('%3f V' %(float(volt)*1.25/(2.**15)))
		sleep(time)
		if(int(volt) > 2**16 - steps - 1):
			while(int(volt)>0):
				volt = volt - steps
				i2c.transfer(0x20,[I2C.Message([0x3F,volt>>8, volt & 0xFF00>>8], read = False)])
				print('%3f V' % (float(volt)*1.25/(2.**15)))
				sleep(time)

Time = sys.argv[1]
Time = float(Time)
Steps = sys.argv[2]
Steps = int(Steps)
while(True):
	run(i2c,Time,Steps)
