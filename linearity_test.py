import main.py as m
from time import sleep
from periphery import I2C
from time import clock

def linearity_test(stepsize, timestep, data_rate, filename):
	file_list = os.listdir(".")
	txt_list = []
	for names in file_list:
		if names.endswith(".txt"):
			txt_list.append(names)
	if file_name + "_output.txt" in txt_list:
		raise Exception("File already exists!")
	i2c = I2C('/dev/i2c-6')
	timestep = float(timestep)
	data_rate = float(data_rate)
	volt = 0x0000
	steps = 0
	increasing = True
	timestamp = 0
	while(int(volt) < 52428):
		adc_output = m.read_ADC()
		f_output = open(filename + "_output.txt", "w+")
		f_input = open(filename + "_input.txt", "w+")
		f_output.write("%0.2f, %0.2f\n"%(adc_output, timestamp))
		f_input.write("%0.2f\n"%(float(volt)*1.25/(2.**15)))
		f_output.flush()
		f_input.flush()
		os.fsync(f_output.fileno())
		os.fsync(f_input.fileno())
		steps += 1
		timestamp = timestamp + data_rate
		sleep(data_rate)
		if steps == int(timestep/data_rate):
			if int(volt) > 52428 - stepsize - 1:
				increasing = False
			if int(volt) < stepsize + 1:
				break
			if increasing == True:
				volt = volt + stepsize
				m.set_DAC(i2c, volt)
			if increasing == False:
				volt = volt - stepsize
				m.set_DAC(i2c, volt)
			steps = 0
	f_output.close()
	f_input.close()