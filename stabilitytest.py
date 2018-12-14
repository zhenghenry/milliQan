import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import matplotlib.colors as c
from itertools import cycle
from scipy import optimize

cycol1 = cycle('bgrcmk')
cycol2 = cycle('bgrcmk')

filename = 'v6trial'
outval, time = np.loadtxt(filename + '_output.txt', delimiter = ',', unpack = True)
inval = np.loadtxt(filename + '_input.txt', delimiter = ',', unpack = True)
inval = inval*1.25/2**15
inval = inval[15960:39300]
outval = outval[15960:39300]
time = time[15960:39300]
previnval = 0.
measuredvoltage = np.array([])
setvoltage = np.array([])
rmse = np.array([])
rms = np.array([])
percentrms = np.array([])
percentrmse = np.array([])
dacval = np.array([])
meansquared = 0
mean = 0
means = np.array([])
meanlist = np.array([])
measureddiff = np.array([])
meandiff = np.array([])
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

fig4 = plt.figure()
ax4 = fig4.add_subplot(111)

for i in range(int(len(inval))):
	measuredvoltage = np.append(measuredvoltage, outval[i]) 
	setvoltage = np.append(setvoltage, inval[i])
	if len(measuredvoltage)%60 == 0:
		mean = np.mean(measuredvoltage)
		meanlist = np.append(meanlist, mean)	
		print(mean)		
		measureddiff = abs(measuredvoltage - setvoltage)
		for k in range(60):
			meandiff = np.append(meandiff, abs(np.mean(measuredvoltage - setvoltage)))
			means = np.append(means, mean)
		mse = mean_squared_error(measureddiff,meandiff)			
		mean_mse = mean_squared_error(measuredvoltage, means)
		if mean < 0.002 or mse == 0:
			percentrms = np.append(percentrms, 1.)
			percentrmse = np.append(percentrmse, 1/60.)
		else:
			percentrms = np.append(percentrms, np.sqrt(mse*60.)/mean)
			percentrmse = np.append(percentrmse, np.sqrt(1/60. + mean**2/mse))			

		ax1.hist(measureddiff - meandiff, bins = 15, alpha = 0.5, color = next(cycol1))
		ax2.hist(measuredvoltage, bins = 15, color = next(cycol2))
		rmse = np.append(rmse, np.sqrt(mse)/abs(np.mean(measuredvoltage - setvoltage)))
		meansquared = 0
		measuredvoltage = np.array([])
		means = np.array([])
		setvoltage = np.array([])
		dacval = np.append(dacval, inval[i])
		meandiff = np.array([])
yerr_lower = np.zeros(len(dacval))
yerr_upper = np.zeros(len(dacval))
for i in range(len(yerr_upper)):
	yerr_upper[i] = 1/np.sqrt(120)
	yerr_lower[i] = rmse[i]
ax1.set_xlabel("ADC Output Voltage (V)")
ax1.set_ylabel("Counts")
ax2.set_xlabel("ADC Output Voltage (V)")
ax2.set_ylabel("Counts")
ax1.set_title("ADC Output Histogram Centered")
ax2.set_title("ADC Output Histogram")

def fitfunc(p, x):
    return p[0] * x + p[1]


def residual(p, x, y, dy):
    return (fitfunc(p, x) - y) / dy


p01 = [1., 1.]
pf1, cov1, info1, mesg1, success1 = optimize.leastsq(residual, p01,
                                                     args=(dacval, meanlist, np.sqrt(mean_mse)), full_output=1)

if cov1 is None:
    print('Fit did not converge')
    print('Success code:', success1)
    print(mesg1)
else:
    print('Fit Converged')
    chisq1 = sum(info1['fvec'] * info1['fvec'])
    dof1 = len(dacval) - len(pf1)
    pferr1 = [np.sqrt(cov1[i, i]) for i in range(len(pf1))]
    print('Converged with chi-squared', chisq1)
    print('Number of degrees of freedom, dof =', dof1)
    print('Reduced chi-squared', chisq1 / dof1)
    print('Inital guess values:')
    print('  p0 =', p01)
    print('Best fit values:')
    print('  pf =', pf1)
    print('Uncertainties in the best fit values:')
    print('  pferr =', pferr1)

    ax4.errorbar(dacval, meanlist, np.sqrt(mean_mse), markersize = 5, capsize = 2, fmt='k.', label='Data')
    f = np.linspace(dacval.min(), dacval.max(), 5000)
    ax4.plot(f, fitfunc(pf1, f), 'r-', label='Fit')
    ax4.set_title("HV Board DAC and ADC Voltage Stability Test")
    ax4.set_xlabel("DAC Input Voltage(V)")
    ax4.set_ylabel("Mean ADC Output Valaue (V)")
    ax4.set_ylim(1400.,2000.)
    ax4.legend()

    textfit1 = '$f =  Ax + B$ \n'\
               '$A = (%.3f \pm %.3f) $\n'\
               '$B = (%.3f \pm %.3f)$ V\n'\
               '$N = %i$ (dof) \n'\
               '$\chi^2/N = % .2f$' % ( pf1[0], pferr1[0],pf1[1], pferr1[1],  dof1,
                chisq1 / dof1)
    ax4.text(0.1, .4, textfit1, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top')
    #ax1.set_xlim([1e-6, 7*1e-6])
    plt.show()

fig5 = plt.figure()
ax5 = fig5.add_subplot(111)
ax5.errorbar(time, outval, yerr = np.sqrt(mean_mse), markersize = 5, capsize = 2, fmt = 'k.') 
ax5.set_xlabel("DAC Input Voltage (V)")
ax5.set_ylabel("Mean ADC Output Valaue (V)")
ax5.set_title("HV Board DAC and ADC Voltage Stability Test")

plt.show()
plt.close('all')

