def Gau(x,a,b,c):
    import numpy as np
    return a*np.exp(-(x-b)**2/(2.*(c**2)))

def GauFit(x, y, p0):
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(Gau, x, y, p0=p0)
    return popt, pcov

def getFWHM(Angles, Energy, p0):
    import numpy as np
    Energy = np.swapaxis(Energy, 0, 1)
    x_points=[] 
    y_points=[]
    args1=Angles>-(3*np.pi/16)
    args2=Angles<(3*np.pi/16)
    for j in Angles[args1 & args2]:
        arg=np.argwhere(Angles==j)[0][0]
        args=Energy[:,arg]>0
        point=np.where(Energy[:,arg]==Energy[args,arg][-1])[0]
        if point.size==0: point=0
        else: point=np.max(point)
        x_points.append(Angles[arg])
        y_points.append(Energy[point])

    if len(p0) != 3:
        raise ValueError("Initial parameters for Gaussian fit must be a list of 3 elements")
    try: popt, pcov = GauFit(x_points, y_points, p0)
    except RuntimeError:
        print("Couldn't fit curve")
        return np.nan, np.nan
    finally:
        a_fit, b_fit, c_fit = popt
        return 2*np.sqrt(2*np.log(2))*abs(c_fit), round((2*np.sqrt(2*np.log(2))*abs(c_fit))*180/np.pi,2)

def getCDSurf(x, y, den, spot, steps):
    import numpy as np
    den_time=np.zeros((len(x), steps))
    cd_sur=[]
    for i in range(steps):
        y_arg=np.argwhere(abs(y)<=spot/2)
        den_time[:,i]=np.mean(den[i][:,y_arg:y_arg],axis=1)
        try: cd_sur=np.append(cd_sur,x[np.argwhere(den_time[:,i]>=1)[0]])
        except IndexError: cd_sur=np.append(cd_sur,np.nan)
    return cd_sur

def GoTrans(Surf, Tau, Time):
    import numpy as np
    arg=np.argwhere(Surf>1)[-1][0]
    Trans=False
    if Time[arg]<2.4*Tau*1e15:
        Trans=True
        return Trans, Time[arg] 
    else: return Trans, np.nan

def PrintPercentage(current_value, max_value):
    if max_value == 0:
        raise ValueError("Max value cannot be zero")
    percentage = round((current_value / max_value) * 100, 1)
    print('|' + '#' * int(percentage) + ' ' * (100 - int(percentage)) + f'| {percentage}%')