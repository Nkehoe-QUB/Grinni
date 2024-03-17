def Gau(x,a,b,c):
    import numpy as np
    return a*np.exp(-(x-b)**2/(2.*(c**2)))

def GauFit(x, y, p0):
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(Gau, x, y, p0=p0)
    return popt, pcov 

def getFWHM(Angles, NumDen, Energy):
    import numpy as np
    NumDen = np.swapaxes(NumDen, 0, 1)
    x_points=[] 
    y_points=[]
    args1=Angles>-(3*np.pi/16)
    args2=Angles<(3*np.pi/16)
    for j in Angles[args1 & args2]:
        arg=np.argwhere(Angles==j)[0][0]
        args=NumDen[:,arg]>0
        point=np.where(NumDen[:,arg]==NumDen[args,arg][-1])[0]
        if point.size==0: point=0
        else: point=np.max(point)
        x_points.append(Angles[arg])
        y_points.append(Energy[point])

    p0=[np.max(y_points),0.0,0.111]
    try: popt, pcov = GauFit(x_points, y_points, p0)
    except RuntimeError:
        print("Couldn't fit curve")
        return np.nan, np.nan
    else:
        a_fit, b_fit, c_fit = popt
        return 2*np.sqrt(2*np.log(2))*abs(c_fit), round((2*np.sqrt(2*np.log(2))*abs(c_fit))*180/np.pi,2)

def getCDSurf(x, y, den, spot, steps):
    import numpy as np
    import matplotlib.pyplot as plt
    den_time=np.zeros((len(x), steps))
    cd_sur=[]
    y_arg=np.where(abs(y)<=spot/2)[0]
    np.set_printoptions(threshold = np.inf)
    for i in range(steps):
        den_time[:,i]=np.mean(den[i][:,y_arg],axis=1)
        # plt.figure()
        # plt.plot(x,np.mean(den[i][:,y_arg],axis=1)) 
        # plt.savefig('den_time_'+str(i)+'.png')
        # plt.close()
        # print(den_time[:,i])
        try: cd_sur=np.append(cd_sur,x[np.argwhere(den_time[:,i]>=1)[0]-1])
        except IndexError: cd_sur=np.append(cd_sur,np.nan)
    # print(den_time)
    return cd_sur, den_time

def GoTrans(Surf, Tau, Time):
    import numpy as np
    try: arg=np.argwhere(Surf>=1)[0]
    except IndexError: return False, np.nan
    else:
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

def MakeMovie(GraphFolder, OutputFolder, initialfile, finalfile, quantity):
    import pathlib
    import cv2
    from cv2 import VideoWriter, VideoWriter_fourcc
    import numpy as np
    import os
    h = cv2.imread(os.path.join(GraphFolder, quantity + '_' + str(initialfile) + '.png'))
    height = h.shape[0]
    width = h.shape[1]
    FPS = 1.0
    fourcc = VideoWriter_fourcc(*'XVID')
    folder_path = OutputFolder
    if not(os.path.exists(folder_path) and os.path.isdir(folder_path)):
        os.mkdir(folder_path)
    video = VideoWriter(os.path.join(OutputFolder, quantity + '.mp4'), fourcc, float(FPS), (width, height))
    for filenumber in range(initialfile, finalfile):
        filename = os.path.join(GraphFolder, quantity + '_' + str(filenumber) + '.png')
        filepath = pathlib.Path(filename)
        if filepath.exists():
            h = cv2.imread(filename)
            video.write(np.uint8(h))
        else:
             print(filename + 'image does not exist')
    video.release()