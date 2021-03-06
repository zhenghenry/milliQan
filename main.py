from periphery import I2C
from time import sleep
import sys
import os

def set_DAC(i2c, voltage):
	i2c.transfer(0x11, [I2C.Message([0x30, voltage >> 8, voltage & 0xFF00 >> 8])])

def read_ADC():
	f = os.popen("cat /sys/bus/iio/devices/iio:device0/in_voltage8_vpvn_raw")
	voltage = float(f.read().split('\n')[0])
	print(voltage)
	return voltage
def read_Temp():
	f = os.popen("cat /sus/bus/iio/devices/iio:device0/in_temp0_raw")
	raw_temp = float(f.read().split('\n')[0])
	f.close()
	f = os.popen("cat /sus/bus/iio/devices/iio:device0/in_temp0_scale")
	scale_temp = float(f.read().split('\n')[0])
	f.close()
	f = os.popen("cat /sus/bus/iio/devices/iio:device0/in_temp0_offset")
	offset_temp = float(f.read().split('\n')[0])
	f.close()
	return (raw_temp + offset_temp)*scale_temp
