#!/usr/local/bin/python3

# This script compares the results from Aither's Sod's shock tube simulations to the exact
# solution. The procedure for determining the exact solution is outlined in "Modern Compressible
# Flow" by Anderson.

import numpy as np
import math
from scipy.optimize import fsolve
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.size'] = 18

# function definitions
def SoS(pressure, density, gamma):
    return math.sqrt(gamma * pressure / density)

def VelScale(pressure, density):
    return math.sqrt(pressure / density)

def NonDimTemperature(pressure, density, gamma):
    return pressure * gamma / density

def Temperature(pressure, density, R):
    return pressure / (R * density)

def VelExpansion(sos4, position, time, gamma):
    return 2. / (gamma + 1.) * (sos4 + position / time)
    
def SoSExpansion(sos4, position, time, gamma):
    vel = VelExpansion(sos4, position, time, gamma)
    return sos4 * (1. - 0.5 * (gamma - 1.) * vel / sos4)
    
def TemperatureExpansion(t4, sos4, position, time, gamma):
    vel = VelExpansion(sos4, position, time, gamma)
    return t4 * (1. - 0.5 * (gamma - 1.) * vel / sos4)**2.
    
def PressureExpansion(p4, sos4, position, time, gamma):
    vel = VelExpansion(sos4, position, time, gamma)
    return p4 * (1. - 0.5 * (gamma - 1.) * vel / sos4)**(2.*gamma/(gamma-1.))

def DensityExpansion(r4, sos4, position, time, gamma):
    vel = VelExpansion(sos4, position, time, gamma)
    return r4 * (1. - 0.5 * (gamma - 1.) * vel / sos4)**(2./(gamma-1.))
    
def InRegion4(sos4, position, time):
    insideRegion4 = False
    vel = position / time    
    if (vel <= -sos4):
        insideRegion4 = True
    return insideRegion4

def InExpansion(sos4, v3, sos3, position, time):
    insideExpansion = False
    vel = position / time
    if (vel >= -sos4 and vel <= v3 - sos3):
        insideExpansion = True
    return insideExpansion

def InRegion3(vp, v3, sos3, position, time):
    insideRegion3 = False
    vel = position / time
    if (vel >= v3 - sos3 and vel <= vp):
        insideRegion3 = True
    return insideRegion3
    
def InRegion2(vp, vs, position, time):
    insideRegion2 = False
    vel = position / time
    if (vel >= vp and vel <= vs):
        insideRegion2 = True
    return insideRegion2
    
def InRegion1(vs, position, time):
    insideRegion1 = False
    vel = position / time
    if (vel >= vs):
        insideRegion1 = True
    return insideRegion1


def main():
    # -----------------------------------------------------------------------
    gamma = 1.4  # ratio of specific heats
    R = 287.     # ideal gas constant
    
    # high pressure state
    p_high = 101325.
    r_high = 1.225
    v_high = 0.
    
    # low pressure state
    p_low = 10132.5
    r_low = 0.153125
    v_low = 0.
    
    time = 0.1   # time for comparison
    num = 1001    # number of points through tube
    lend = -0.5   # left end of simulation
    rend = 0.5    # right end of simulation

    # simulation data files
    fmuscl = "muscl_vanalbada_coarse.csv"
    ffirst = "first_order_coarse.csv"
    fweno = "weno.csv"
    
    # -----------------------------------------------------------------------

    # nondimensionalization
    # density nondimensionalized by high pressure region density
    # pressure nondimensionalized by high pressure region pressure
    # velocity nondimensionalized by sqrt(Pinf / Rinf)
    #    nondim P factor = Pinf; nondim r factor = Rinf
    #    in sos nondimensionalization P is nondimensionalized by Rinf * Ainf^2
    #    therefore Pinf = Rinf * Vinf^2
    # temperature nondimensionalized by Tinf / gamma because nondimensional
    # equation of state substitutes 1/gamma for R
    # since velocity was nondimensionlized by sqrt(Pinf/ Rinf) instead of the
    # speed of sound (sqrt(gamma * Pinf / Rinf), time must be scaled by gamma as well

    nTime = time / gamma

    # get simulation data
    muscl = np.genfromtxt(fmuscl, skip_header=1, delimiter=",")
    # nondimensionalize simulation data
    muscl[:,0] /= r_high
    muscl[:,2] /= p_high
    muscl[:,3] /= Temperature(p_high, r_high, R) / gamma
    muscl[:,4] /= VelScale(p_high, r_high)

    first = np.genfromtxt(ffirst, skip_header=1, delimiter=",")
    first[:,0] /= r_high
    first[:,2] /= p_high
    first[:,3] /= Temperature(p_high, r_high, R) / gamma
    first[:,4] /= VelScale(p_high, r_high)

    weno = np.genfromtxt(fweno, skip_header=1, delimiter=",")
    weno[:,0] /= r_high
    weno[:,2] /= p_high
    weno[:,3] /= Temperature(p_high, r_high, R) / gamma
    weno[:,4] /= VelScale(p_high, r_high)
    
    # exact data
    # left state (high pressure)
    r4 = r_high / r_high
    p4 = p_high / p_high
    t4 = NonDimTemperature(p4, r4, gamma)
    v4 = v_high
    
    # right state (low pressure)
    r1 = r_low / r_high
    p1 = p_low / p_high
    t1 = NonDimTemperature(p1, r1, gamma)
    v1 = v_low     
    
    gm1 = gamma - 1.
    gp1 = gamma + 1.
    x = np.linspace(lend, rend, num)
    
    # calculate p2/p1
    a1 = SoS(p1, r1, gamma)
    a4 = SoS(p4, r4, gamma)
    func = lambda p2_p1 : p4/p1 - p2_p1 * (1. - (gm1 * a1/a4 * (p2_p1 - 1.)) / 
        math.sqrt(2.*gamma * (2.*gamma + gp1 * (p2_p1 - 1.))))**(-2.*gamma/gm1)
        
    # calculate state at region 2
    p2 = p1 * fsolve(func, 1.0)
    t2 = t1 * p2/p1 * (gp1/gm1 + p2/p1) / (1. + (gp1/gm1) * (p2/p1))
    r2 = r1 * (1. + (gp1/gm1) * (p2/p1)) / (gp1/gm1 + p2/p1)
    # calcualte velocity ahead of shock relative to wave
    vs = a1 * math.sqrt(gp1/(2.*gamma) * (p2/p1 - 1.) + 1.)
    # calculate velocity of contact wave
    vp = a1/gamma * (p2/p1 - 1.) * math.sqrt((2.*gamma/gp1) / (p2/p1 + gm1/gp1))
    v2 = vp
    a2 = SoS(p2, r2, gamma)
    
    # calculate state at region 3
    v3 = v2
    p3 = p4 * (p2/p1) / (p4/ p1)
    r3 = r4 * (p3/p4)**(1./gamma)
    t3 = t4 * (p3/p4)**(gm1/gamma)
    a3 = SoS(p3, r3, gamma)

    # assign states
    density = np.zeros(num)
    pressure = np.zeros(num)
    temperature = np.zeros(num)
    velocity = np.zeros(num)
    regions = np.zeros(num)
    
    for ii in range(0, num):
        if (InRegion1(vs, x[ii], nTime)):
            density[ii] = r1
            pressure[ii] = p1
            temperature[ii] = t1
            velocity[ii] = v1
            regions[ii] = 1
        elif (InRegion2(vp, vs, x[ii], nTime)):
            density[ii] = r2
            pressure[ii] = p2
            temperature[ii] = t2
            velocity[ii] = v2
            regions[ii] = 2
        elif (InRegion3(vp, v3, a3, x[ii], nTime)):
            density[ii] = r3
            pressure[ii] = p3
            temperature[ii] = t3
            velocity[ii] = v3
            regions[ii] = 3
        elif (InExpansion(a4, v3, a3, x[ii], nTime)):
            density[ii] = DensityExpansion(r4, a4, x[ii], nTime, gamma)
            pressure[ii] = PressureExpansion(p4, a4, x[ii], nTime, gamma)
            temperature[ii] = TemperatureExpansion(t4, a4, x[ii], nTime, gamma)
            velocity[ii] = VelExpansion(a4, x[ii], nTime, gamma)
            regions[ii] = 3.5
        elif (InRegion4(a4, x[ii], nTime)):
            density[ii] = r4
            pressure[ii] = p4
            temperature[ii] = t4
            velocity[ii] = v4
            regions[ii] = 4
        else:
            print("ERROR: Region not found!!!")
            print("Position:", x[ii])

    # plot results
    x = x + 0.5
    legend = ["Exact", "First", "MUSCL", "WENO"]
    fig, ax = plt.subplots(2, 2, figsize=(12,8))
    fig.subplots_adjust(top=0.75)
    ax[0,0].plot(x, density, 'k', lw=2)
    ax[0,0].plot(first[:,-3], first[:,0], 'g', lw=2)
    ax[0,0].plot(muscl[:,-3], muscl[:,0], 'r', lw=2)
    ax[0,0].plot(weno[:,-3], weno[:,0], 'b', lw=2)    
    ax[0,0].grid('on')
    ax[0,0].set_xlabel("Position (m)")
    ax[0,0].set_ylabel("Normalized Density")
    ax[0,0].set_xlim([x[0], x[-1]])
    ax[0,0].set_ylim([0, 1.1])
    leg = ax[0,0].legend(legend, loc='best')
    leg.draggable()
    
    ax[0,1].plot(x, pressure, 'k', lw=2)
    ax[0,1].plot(first[:,-3], first[:,2], 'g', lw=2)
    ax[0,1].plot(muscl[:,-3], muscl[:,2], 'r', lw=2)
    ax[0,1].plot(weno[:,-3], weno[:,2], 'b', lw=2)
    ax[0,1].grid('on')
    ax[0,1].set_xlabel("Position (m)")
    ax[0,1].set_ylabel("Normalized Pressure")
    ax[0,1].set_xlim([x[0], x[-1]])
    ax[0,1].set_ylim([0, 1.1])
    leg = ax[0,1].legend(legend, loc='best')
    leg.draggable()
    
    ax[1,0].plot(x, temperature, 'k', lw=2)
    ax[1,0].plot(first[:,-3], first[:,3], 'g', lw=2)
    ax[1,0].plot(muscl[:,-3], muscl[:,3], 'r', lw=2)
    ax[1,0].plot(weno[:,-3], weno[:,3], 'b', lw=2)
    ax[1,0].grid('on')
    ax[1,0].set_xlabel("Position (m)")
    ax[1,0].set_ylabel("Normalized Temperature")
    ax[1,0].set_xlim([x[0], x[-1]])
    ax[1,0].set_ylim([0.7, 1.7])
    leg = ax[1,0].legend(legend, loc='best')
    leg.draggable()
    
    ax[1,1].plot(x, velocity, 'k', lw=2)
    ax[1,1].plot(first[:,-3], first[:,4], 'g', lw=2)
    ax[1,1].plot(muscl[:,-3], muscl[:,4], 'r', lw=2)
    ax[1,1].plot(weno[:,-3], weno[:,4], 'b', lw=2)
    ax[1,1].grid('on')
    ax[1,1].set_xlabel("Position (m)")
    ax[1,1].set_ylabel("Normalized Velocity")
    ax[1,1].set_xlim([x[0], x[-1]])
    ax[1,1].set_ylim([0, 1])
    leg = ax[1,1].legend(legend, loc='best')
    leg.draggable()
    
    plt.suptitle("Shock Tube at Nondimensional Time: {}".format(time), y=1.0)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    fig.subplots_adjust(top=0.75)
    ax.plot(x, density, 'k', lw=2)
    ax.plot(first[:,-3], first[:,0], 'g', lw=2)
    ax.plot(muscl[:,-3], muscl[:,0], 'r', lw=2)
    ax.plot(weno[:,-3], weno[:,0], 'b', lw=2)    
    ax.grid('on')
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Normalized Density")
    ax.set_xlim([0.3, 0.7])
    ax.set_ylim([0, 1.1])
    leg = ax.legend(legend, loc='best')
    leg.draggable()
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()
