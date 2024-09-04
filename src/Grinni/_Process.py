from ._Utils import Gau, getFWHM, getCDSurf, GoTrans, PrintPercentage, MakeMovie, MovingAverage

class Process():
    def __init__(self, SimName=".", x_spot=0, Tau=0, Ped=None, Dim=2, Log=True, Movie=True):
        ########### Constants ##################################
        self.c = 299792458. 
        self.me = 9.11e-31
        self.epsilon0 = 8.854187e-12
        self.e = 1.602176e-19
        self.amu = 1.673776e-27
        self.massNeutron = 1838. # in units of electron mass
        self.massProton = 1836.

        self.P_r = self.me * self.c
        self.MeV_to_J = 1.6e-13
        self.micro = 1e-6
        self.nano = 1e-9
        self.pico = 1e-12
        self.femto = 1e-15
        ########################################################
        try: import happi
        except ImportError:
            raise ImportError("happi is not installed")
        import numpy as np
        from cmcrameri import cm as cmaps
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        import matplotlib.gridspec as gridspec
        import os
        import pandas as pd
        import re
        self.pd = pd
        self.os=os
        self.happi = happi
        self.np = np
        self.plt = plt
        self.cmaps = cmaps
        self.plt.rcParams["axes.labelsize"] = 16
        self.plt.rcParams["axes.titlesize"] = 16
        self.plt.rcParams["xtick.labelsize"] = 14
        self.plt.rcParams["ytick.labelsize"] = 14
        self.plt.rcParams["legend.fontsize"] = 14
        self.cm = colors
        self.gs = gridspec
        self.re = re
        self.SimName = SimName
        self.x_spot = x_spot
        self.Dim = Dim
        self.Log = Log
        self.Movie = Movie
        self.Units = ["um", "fs", "MeV", "V/m", "kg*m/s", 'um^-3*MeV^-1', 'um^-3*kg^-1*(m/s)^-1', 'T']
        file_path = f'{self.SimName}/smilei.py'
        self.box_x = None
        self.box_y = None
        with open(file_path, 'r') as file:
            for line in file:
                xmatch = re.search(r'box_x\s*=\s*(\d+).\s*\*\s*micro', line)
                if xmatch is None: re.search(r'box_x\s*=\s*(\d+)\s*\*\s*micro', line)
                if xmatch:
                    self.box_x = float(xmatch.group(1))
                    break
            for line in file:
                ymatch = re.search(r'box_y\s*=\s*(\d+).\s*\*\s*micro', line)
                if ymatch is None: re.search(r'box_y\s*=\s*(\d+)\s*\*\s*micro', line)
                if ymatch:
                    self.box_y = float(ymatch.group(1))
                    break
        print(f"\nBox size is {self.box_x}x{self.box_y} micrometers")
        self.Simulation = self.happi.Open(self.SimName, verbose=False)
        if self.Simulation == "Invalid Smilei simulation":
            raise ValueError(f"Simulation {self.SimName} does not exist")
        else: print(f"\nSimulation {self.SimName} loaded")
        if x_spot == 0 or Tau == 0:
            raise ValueError("No spot size or simulation size or Tau value was provided")
        self.x_spot = x_spot 
        if self.x_spot > 1:
            print("\nx_spot is in meters, converting to micrometers")
            self.x_spot = self.x_spot*self.micro
        self.Tau = Tau
        if self.Tau > 1:
            print("\nTau is in seconds, converting to femtoseconds")
            self.Tau = self.Tau*self.femto
        self.t0=((self.x_spot/self.c)+((2*self.Tau)/(2*self.np.sqrt(self.np.log(2)))))/self.femto
        if Ped is not None: 
            print("\nAdding Ped to t0")
            if Ped > 1:
                print("\nPed is in seconds, converting to picoseconds")
                Ped = Ped*self.pico
            self.t0 = self.t0 + (Ped/self.femto)
        self.simulation_path = self.os.path.abspath(self.SimName)
        self.raw_path = self.os.path.join(self.simulation_path,  "Raw")
        if not(self.os.path.exists(self.raw_path) and self.os.path.isdir(self.raw_path)):
            self.os.mkdir(self.raw_path)
        print(f"\nGraphs will be saved in {self.raw_path}")
        self.pros_path = self.os.path.join(self.simulation_path, "Processed")
        if not(self.os.path.exists(self.pros_path) and self.os.path.isdir(self.pros_path)):
            self.os.mkdir(self.pros_path)
        print(f"\nVideos will be saved in {self.pros_path}")
    
    def moving_average(self, x, w):
        return self.np.convolve(x, self.np.ones(w), 'valid') / w

    def GetData(self, Diag, Name, Field=None, units=None, Data=True, Axis=True, ProsData=True, x_offset=None, y_offset=None):
        # Check if Diag is a valid diagnostic
        if not self.Simulation.getDiags(Diag):
            raise ValueError(f"Diag '{Diag}' is not a valid diagnostic")
        # Check if Name is a valid diagnostic
        if Name not in self.Simulation.getDiags(Diag)[1]:
            raise ValueError(f"Name {Name} is not a valid diagnostic")
        # Check if units is a list of strings
        if units is not None:
            if not (isinstance(units, list) and all(isinstance(unit, str) for unit in units)):
                raise TypeError("units must be a list of strings")
        
        # Get the data
        if x_offset is not None and x_offset < 1:
            x_offset = x_offset/self.micro
        if y_offset is not None and y_offset < 1:
            y_offset = y_offset/self.micro
        if Diag == "ParticleBinning":
            if units is None:
                MetaData = self.Simulation.ParticleBinning(Name)
            elif units is not None:
                MetaData = self.Simulation.ParticleBinning(Name, units=units)
            if self.Dim == 2:
                axis_names=['x', 'y', 'user_function0', 'ekin', 'px', 'py']
            elif self.Dim == 3:
                axis_names=['x', 'y', 'z', 'user_function0', 'ekin', 'px', 'py']
            
        elif Diag == "Fields":
            if units is None:
                MetaData = self.Simulation.Field(Name, Field)
            elif units is not None:
                MetaData = self.Simulation.Field(Name, Field, units=units)
            axis_names=['x', 'y']
        else: raise ValueError(f"Diag '{Diag}' is not a valid diagnostic")

        Values = self.np.array(MetaData.getData())
        self.TimeSteps = self.np.array(MetaData.getTimesteps())
        if Diag == "Fields":
            self.max_number = float('-inf')  # Initialize max_number to negative infinity
            for array in Values:
                current_max = self.np.max(array)
                if current_max > self.max_number:
                    self.max_number = current_max
        axis ={}
        axis["Time"] = self.np.round(MetaData.getTimes()-self.t0,2)
        bin_size = None
        for axis_name in axis_names:
            axis_data = self.np.array(MetaData.getAxis(axis_name))
            if len(axis_data)==0:
                    continue
            elif axis_name == "x":
                axis_data = axis_data - x_offset if x_offset is not None else axis_data  
            elif axis_name == "y":
                axis_data = axis_data - y_offset if y_offset is not None else axis_data - ((axis_data[0]+axis_data[-1])/2)
            elif axis_name == "z":
                axis_data = axis_data  
            elif axis_name == "user_function0":
                bin_size=(axis_data[1]-axis_data[0]) if bin_size is None else bin_size*(axis_data[1]-axis_data[0])
            elif axis_name == "ekin":
                if "carbon" in Name:
                    Z=12
                elif "proton" in Name:
                    Z=1
                elif "electron" in Name:
                    Z=1
                axis_data = self.np.array(MetaData.getAxis('ekin', timestep=self.TimeSteps[0])/Z)
                for t in self.TimeSteps[1:]:
                    axis_data=self.np.vstack((axis_data, self.np.array(MetaData.getAxis('ekin', timestep=t)/Z)))
                if ProsData:
                    Values = Values * (self.box_x * self.box_y)
            elif axis_name == "px":
                if "x-px" in Name:
                    bin_size = axis['x'][1]-axis['x'][0]
                axis_data = self.np.array(MetaData.getAxis('px', timestep=self.TimeSteps[0]),ndmin=2)
                for t in self.TimeSteps[1:]:
                    axis_data = self.np.vstack((axis_data,self.np.array(MetaData.getAxis('px', timestep=t))))
            elif axis_name == "py":
                axis_data = self.np.array(MetaData.getAxis('py', timestep=self.TimeSteps[0]),ndmin=2)
                for t in self.TimeSteps[1:]:
                    axis_data = self.np.vstack((axis_data,self.np.array(MetaData.getAxis('py', timestep=t))))
            axis[axis_name] = axis_data
        if Data: 
            if Axis: return Values * bin_size if bin_size is not None else Values, axis
            else: return Values * bin_size if bin_size is not None else Values
        elif Axis: return axis
        else: raise ValueError("No data or axis requested")
        
    def DensityPlot(self, Species=[], E_las=False, E_avg=False, Field=None, EMax=None, Colours=None, CBMin=None, CBMax=None, File=None):
        if not Species and (E_las and E_avg) is None:
            raise ValueError("No species or field were provided")
        if Species and not isinstance(Species, list):
            Species = [Species]
        if Colours is not None and not isinstance(Colours, list):
            if not isinstance(Colours, str):
                raise ValueError("Colours must be a list of strings")
            elif Colours == "jet":
                Colours = None
            elif len(Colours) != len(Species):
                print("Number of colours must match number of species\nSetting colours to 'jet'")
                Colours = None
            else: Colours = [Colours]
        if (E_las or E_avg) and Field is None:
            raise ValueError("No field was provided")
        if E_las:
            Ey, E_axis = self.GetData("Fields", "instant fields", Field=Field, units=self.Units, x_offset=self.x_spot)
        elif E_avg:
            Ey, E_axis = self.GetData("Fields", "average fields", Field=Field, units=self.Units, x_offset=self.x_spot)
        
        den_to_plot={}
        axis={}
        TempFile=File if File is not None else "density"
        if Species:
            for type in Species:
                Diag=type + ' density'
                if Diag not in self.Simulation.getDiags("ParticleBinning")[1] and type != "rel electron":
                    raise ValueError(f"Diagnostic '{Diag}' is not a valid density diagnostic")
                if type == "rel electron":
                    elec_den, axis[type] = self.GetData("ParticleBinning", "electron density", units=self.Units, x_offset=self.x_spot)
                    en_den = self.GetData("ParticleBinning", "electron energy density", Axis=False, x_offset=self.x_spot)
                    den_to_plot[type] = self.np.array(elec_den) / ((self.np.array(en_den) / self.np.array(elec_den)) + 1)
                    continue
                den_to_plot[type], axis[type] = self.GetData("ParticleBinning", Diag, units=self.Units, x_offset=self.x_spot)
        
        if Species: print(f"\nPlotting {Species} densities")
        else: print(f"\nPlotting {Field} field")
        FinalFile = self.TimeSteps.size
        for i in range(self.TimeSteps.size):
            fig, ax = self.plt.subplots(num=1,clear=True)
            if E_las or E_avg:
                try: field = self.np.swapaxes(Ey[i], 0, 1)
                except IndexError: 
                    FinalFile = i
                    continue
                else:
                    FUnit = 'V/m' if 'E' in Field else 'T'
                    cax1=ax.pcolormesh(E_axis['x'], E_axis['y'], field, cmap=self.cmaps.vik, norm=self.cm.CenteredNorm(halfrange=self.max_number if EMax is None else EMax))
                    cbar1 = fig.colorbar(cax1, aspect=50)
                    cbar1.set_label(f"{Field} [{FUnit}]")
            if Species:
                for type in Species:
                    SaveFile=TempFile if File is not None else f"{type}_" + TempFile
                    den = self.np.swapaxes(den_to_plot[type][i], 0, 1)
                    cax=ax.pcolormesh(axis[type]['x'], axis[type]['y'], den, cmap=self.cmaps.batlow_r if Colours is None else getattr(self.cmaps, Colours[Species.index(type)]), norm=self.cm.LogNorm(vmin=1e-2 if CBMin is None else CBMin, vmax=1e3 if CBMax is None else CBMax))
                    if (Colours is not None) and (len(Colours) > 1) and (not E_las or not E_avg):
                        cbar=fig.colorbar(cax, aspect=50)
                        cbar.set_label(f"N$_{{{type}}}$ [$N_c$]")
                if (Colours is None) or (len(Colours) == 1):
                    cbar=fig.colorbar(cax, aspect=50)
                    cbar.set_label('N [$N_c$]')
            ax.grid(True)
            ax.set_xlabel(r'x [$\mu$m]')
            ax.set_ylabel(r'y [$\mu$m]')
            if Species: ax.set_title(f"{axis[type]['Time'][i]}fs")
            else: ax.set_title(f"{E_axis['Time'][i]}fs")
            fig.tight_layout()
            if not Species: SaveFile=TempFile if File is not None else f"{Field}_" + TempFile
            self.plt.savefig(self.raw_path + "/" + SaveFile + "_" + str(i) + ".png",dpi=200)
            if self.Log: 
                PrintPercentage(i, self.TimeSteps.size -1 )
        print(f"\nDensities saved in {self.raw_path}")
        if self.Movie:
            MakeMovie(self.raw_path, self.pros_path, 0, FinalFile, SaveFile)
            print(f"\nMovies saved in {self.pros_path}")
            
    def SpectraPlot(self, Species=[], XMax=None, YMin=None, YMax=None, File=None, ProsData=True):
        if not Species:
            raise ValueError("No species were provided")
        if not isinstance(Species, list):
            Species = [Species]
        spect_to_plot={}
        axis={}
        label={}
        TempFile=File if File is not None else "energies"
        x_max = 0
        for type in Species:
            Diag=type + ' spectra'
            if Diag not in self.Simulation.getDiags("ParticleBinning")[1] and type != "rel electron":
                raise ValueError(f"Diagnostic '{type}' is not a valid spectra diagnostic")
            if type == "rel electron":
                elec_den, axis[type] = self.GetData("ParticleBinning", "electron density", units=self.Units, ProsData=ProsData)
                en_den = self.GetData("ParticleBinning", "electron energy density", Axis=False)
                spect_to_plot[type] = self.np.array(elec_den) / ((self.np.array(en_den) / self.np.array(elec_den)) + 1)
                continue
            spect_to_plot[type], axis[type] = self.GetData("ParticleBinning", Diag, units=self.Units, ProsData=ProsData)
            label[type] = type
        
        print(f"\nPlotting {Species} spectra")
        for type in Species:
            dfs = []
            for i in range(self.TimeSteps.size):
                if ProsData:
                    spect_to_plot[type][i] = MovingAverage(spect_to_plot[type][i], 3)
                if self.np.max(axis[type]['ekin'][i][~self.np.isnan(axis[type]['ekin'][i])]) > x_max:
                    x_max = self.np.max(axis[type]['ekin'][i][~self.np.isnan(axis[type]['ekin'][i])])
                df =self.pd.DataFrame({
                    'Time':axis[type]['Time'][i],
                    'Energy':axis[type]['ekin'][i],
                    'dNdE':spect_to_plot[type][i]
                    })
                dfs.append(df)
            dfs = self.pd.concat(dfs)
            with open(self.os.path.join(self.simulation_path, f"{type.replace(' ','_')}_energy.csv"), 'w') as file:
                dfs.to_csv(file, index=False)

            print(f"\n{type} energies saved in {self.simulation_path}")
        print(x_max)
        for i in range(self.TimeSteps.size):
            fig, ax = self.plt.subplots(num=1,clear=True)
            for type in Species:
                SaveFile=TempFile if File is not None else f"{type}_" + TempFile
                ax.plot(axis[type]['ekin'][i], spect_to_plot[type][i], label=f"{label[type]}")
            ax.set_xlabel('E [$MeV$]')
            ax.set_xlim(0,x_max if XMax is None else XMax)
            ax.set_ylim(1e4 if YMin is None else YMin,1e11 if YMax is None else YMax)
            ax.set_ylabel('dNdE [1/MeV]' if ProsData else 'dNdE [1/MeV $\\mu$m$^3$]')
            ax.set_yscale('log')
            ax.grid(True)
            ax.legend()
            ax.set_title(f"{axis[type]['Time'][i]}fs")
            fig.tight_layout()
            self.plt.savefig(self.raw_path + '/' + SaveFile + '_' + str(i) + '.png',dpi=200)
            if self.Log: 
                PrintPercentage(i, self.TimeSteps.size -1 )
        print(f"\nSpectra saved in {self.raw_path}")
        if self.Movie:
            MakeMovie(self.raw_path, self.pros_path, 0, self.TimeSteps.size, SaveFile)
            print(f"\nMovies saved in {self.pros_path}")

    def PhaseSpacePlot(self, Species=[], Phase=None, CBMin=None, CBMax=None, YMin=None, YMax=None, XMin=None, XMax=None, File=None):
        if not Species:
            raise ValueError("No phase spaces were provided")
        if not isinstance(Species, list):
            Species = [Species]
        if Phase is None:
            print("No phase space was provided, defaulting to x-px")
            Phase = "x-px"
        phase_to_plot={}
        axis={}
        label={}
        TempFile=File if File is not None else f"{Phase}_phase"
        for type in Species:
            Diag=type + ' ' + Phase + ' phase space'
            if Diag not in self.Simulation.getDiags("ParticleBinning")[1]:
                raise ValueError(f"Diagnostic '{Diag}' is not a valid phase space diagnostic")
            phase_to_plot[type], axis[type] = self.GetData("ParticleBinning", Diag, units=self.Units, x_offset=self.x_spot)
            label[type] = type
        
        print(f"\nPlotting {Species} phase spaces")
        Phase = Phase.split('-')
        for p in range(len(Phase)):
            if Phase[p] == "energy":
                Phase[p] = "ekin"
                break
        for type in Species:
            max0 = 0
            min0 = 0
            max1 = 0
            min1 = 0
            InitialFile=0
            for i in range(self.TimeSteps.size):
                if Phase[1] == "py" :
                    if self.np.max(axis[type]["py"][i]) > max1:
                        max1 = self.np.max(axis[type]["py"][i])
                    if self.np.min(axis[type]["py"][i]) < min1:
                        min1 = self.np.min(axis[type]["py"][i])
                else:
                    if self.np.max(axis[type][Phase[1]][i]) > max0:
                        max0 = self.np.max(axis[type][Phase[1]][i])
                    if self.np.min(axis[type][Phase[1]][i]) < min0:
                        min0 = self.np.min(axis[type][Phase[1]][i])
            for i in range(self.TimeSteps.size):
                fig, ax = self.plt.subplots(num=1,clear=True)
                SaveFile=TempFile if File is not None else f"{type}_" + TempFile
                momentum = self.np.swapaxes(phase_to_plot[type][i], 0, 1)
                X = axis[type][Phase[0]] if Phase[0] not in ["px" or "py"] else axis[type][Phase[0]][i]
                Y = axis[type][Phase[1]][i]
                try: cax = ax.pcolormesh(X, Y, momentum, cmap=self.cmaps.batlow_r, norm=self.cm.LogNorm(vmin=1e2 if CBMin is None else CBMin, vmax=1e10 if CBMax is None else CBMax))
                except ValueError: 
                    InitialFile+=1
                    continue
                cbar = fig.colorbar(cax, aspect=50)
                if Phase[1] == "px":
                    ax.set_xlabel(r'x [$\mu$m]')
                    ax.set_ylabel('px [kgm/s]')
                    ax.set_ylim(min0 if YMin is None else YMin, max0 if YMax is None else YMax)
                    cbar.set_label('dndpx [(kgm/s)$^{-1}$]')
                elif Phase[1] == "py":
                    ax.set_ylabel('py [kgms$^{-1}$]')
                    ax.set_xlabel('px [kgms$^{-1}$]')
                    ax.set_ylim(min1 if YMin is None else YMin, max1 if YMax is None else YMax)
                    ax.set_xlim(min0 if XMin is None else XMin, max0 if XMax is None else XMax)
                elif Phase[1] == "ekin":
                    ax.set_xlabel(r'x [$\mu$m]')
                    ax.set_ylabel('Energy [MeV]')
                    ax.set_ylim(min0 if YMin is None else YMin, max0 if YMax is None else YMax)
                    cbar.set_label('dndE [MeV$^{-1}$]')
                ax.set_title(f"{axis[type]['Time'][i]}fs")
                fig.tight_layout()
                self.plt.savefig(self.raw_path + '/' + SaveFile + '_' + str(i) + '.png',dpi=200)
                if self.Log: 
                    PrintPercentage(i, self.TimeSteps.size -1 )
            print(f"\nPhase spaces saved in {self.raw_path}")
            if self.Movie:
                MakeMovie(self.raw_path, self.pros_path, InitialFile, self.TimeSteps.size, SaveFile)
                print(f"\nMovies saved in {self.pros_path}")
    
    def AnglePlot(self, Species=[], CBMin=None, CBMax=None, XMax=None, YMin=None, YMax=None, LasAngle=None, File=None):
        if not Species:
            raise ValueError("No species were provided")
        if not isinstance(Species, list):
            Species = [Species]
        if not isinstance(XMax, list):
            if XMax is not None:
                XMax = [XMax]
        if XMax is not None:
            if len(XMax) < len(Species) and len(XMax) != 1:
                raise ValueError("XMax must be a list of the same length as Species or a single value")
        if YMin is not None and YMin < -self.np.pi:
            YMin = self.np.radians(YMin)
        if YMax is not None and YMax > self.np.pi:
            YMax = self.np.radians(YMax)
        if YMin is None:
            if YMax is not None:
                YMin = -YMax
        else:
            if YMax is None:
                if YMin > 0:
                    YMin = -YMin
                YMax = -YMin
        angle_to_plot={}
        axis={}
        label={}
        InitalFile=0
        TempFile=File if File is not None else "angles"
        for type in Species:
            Diag=type + ' angle'
            if Diag not in self.Simulation.getDiags("ParticleBinning")[1]:
                raise ValueError(f"Diagnostic '{type}' is not a valid angle diagnostic")
            angle_to_plot[type], axis[type] = self.GetData("ParticleBinning", Diag, units=self.Units, x_offset=10)
            label[type] = type
        
        print(f"\nPlotting {Species} angles")
        EMax=[]
        for type in Species:
            x_max=0
            dfs = []
            for i in range(self.TimeSteps.size):
                for j in range(len(axis[type]['user_function0'])):
                    df = self.pd.DataFrame({
                        'Time': axis[type]['Time'][i],
                        'Angles': axis[type]['user_function0'][j],
                        'Energy': axis[type]['ekin'][i],
                        'dNdE': angle_to_plot[type][i][j]
                    })
                    dfs.append(df)
                if self.np.max(axis[type]['ekin'][i]) > x_max:
                    x_max = self.np.max(axis[type]['ekin'][i][~self.np.isnan(axis[type]['ekin'][i])])
            dfs = self.pd.concat(dfs)
            with open(self.os.path.join(self.simulation_path, f"{type.replace(' ','_')}_ang_energy.csv"), 'w') as file:
                dfs.to_csv(file, index=False)
            EMax.append(x_max)
            print(f"\n{type} angle energies saved in {self.simulation_path}")
        for i in range(self.TimeSteps.size):
            if len(Species) == 1:
                fig, ax = self.plt.subplots(num=1,clear=True, subplot_kw={'projection': 'polar'})
                for type in Species:
                    SaveFile=TempFile if File is not None else f"{type}_" + TempFile
                    angles = self.np.swapaxes(angle_to_plot[type][i], 0, 1)
                    try: cax = ax.pcolormesh(axis[type]['user_function0'],axis[type]['ekin'][i], angles, cmap=self.cmaps.batlow_r, norm=self.cm.LogNorm(vmin=1e4 if CBMin is None else CBMin, vmax=1e10 if CBMax is None else CBMax))
                    except ValueError: 
                        if type == Species[0]: 
                            InitalFile+=1
                        continue
                    cbar = fig.colorbar(cax, aspect=50)
                    cbar.set_label('dNdE [1/MeV $\\mu$m$^3$]')
                    if LasAngle is not None:
                        ax.vlines(self.np.radians(LasAngle), 0, EMax[Species.index(type)], colors='r', linestyles='dashed')
                    ax.set_xlim(-self.np.pi/3 if YMin is None else YMin,self.np.pi/3 if YMax is None else YMax)
                    ax.set_ylim(0,EMax[0] if XMax is None else XMax[0])
                    ax.set_title(f"{label[type]}")
                    if YMax is not None and YMax > self.np.pi/2:
                        ax.set_rlabel_position(90)
            else:
                fig, ax = self.plt.subplots(ncols=len(Species), num=1,clear=True, subplot_kw={'projection': 'polar'})
                for type in Species:
                    SaveFile=TempFile if File is not None else f"{type}_" + TempFile
                    angles = self.np.swapaxes(angle_to_plot[type][i], 0, 1)
                    try: cax = ax[Species.index(type)].pcolormesh(axis[type]['user_function0'],axis[type]['ekin'][i], angles, cmap=self.cmaps.batlow_r, norm=self.cm.LogNorm(vmin=1e4 if CBMin is None else CBMin, vmax=1e10 if CBMax is None else CBMax))
                    except ValueError:
                        if type == Species[0]: 
                            InitalFile+=1
                        continue
                    cbar = fig.colorbar(cax, aspect=50)
                    cbar.set_label('dNdE [1/MeV $\\mu$m$^3$]')
                    if LasAngle is not None:
                        ax[Species.index(type)].vlines(self.np.radians(LasAngle), 0, EMax[Species.index(type)], colors='r', linestyles='dashed')
                    ax[Species.index(type)].set_title(f"{label[type]}", fontsize=16)
                    ax[Species.index(type)].set_xlim(-self.np.pi/3 if YMin is None else YMin,self.np.pi/3 if YMax is None else YMax)
                    ax[Species.index(type)].set_ylim(0,EMax[Species.index(type)] if XMax is None else (XMax[0] if len(XMax) ==1 else XMax[Species.index(type)]))
            
            fig.suptitle(f"{axis[type]['Time'][i]}fs")
            fig.tight_layout()
            self.plt.savefig(self.raw_path + '/' + SaveFile + '_' + str(i) + '.png',dpi=200)
            if self.Log: 
                PrintPercentage(i, self.TimeSteps.size -1 )
        print(f"\nAngles saved in {self.raw_path}")
        if self.Movie:
            MakeMovie(self.raw_path, self.pros_path, InitalFile, self.TimeSteps.size, SaveFile)
            print(f"\nMovies saved in {self.pros_path}")

    def AngleEnergyPlot(self, Species=[], YMin=None, YMax=None, Angles=[], AngleOffset=0, File=None, ProsData=True, DataOnly=False):
        if not Species:
            raise ValueError("No species were provided")
        if not isinstance(Species, list):
            Species = [Species]
        if not isinstance(Angles, list):
            Angles = [Angles]
        if len(Angles) == 0:
            Angles = [0,'FWHM']
        angle_to_plot={}
        axis={}
        label={}
        TempFile=File if File is not None else f"angles_energy"
        for type in Species:
            Diag=type + ' angle'
            if Diag not in self.Simulation.getDiags("ParticleBinning")[1]:
                raise ValueError(f"Diagnostic '{type}' is not a valid angle diagnostic")
            angle_to_plot[type], axis[type] = self.GetData("ParticleBinning", Diag, units=self.Units, x_offset=10)
            label[type] = type
        
        EMax=[]
        InitialFile=0
        for type in Species:
            x_max=0
            for i in range(self.TimeSteps.size):
                if ProsData:
                    angle_to_plot[type][i] = MovingAverage(angle_to_plot[type][i], 3)
                if self.np.max(axis[type]['ekin'][i]) > x_max:
                    x_max = self.np.max(axis[type]['ekin'][i][~self.np.isnan(axis[type]['ekin'][i])])
            EMax.append(x_max)
        for type in Species:
            if DataOnly:
                if len(Species) > 1 or len(Angles) > 1:
                    raise ValueError("DataOnly is only available for a single species and angle")
                for j in (Angles):
                    if j=='FWHM':
                        FWHM_rad, FWHM_deg = getFWHM(axis[type]['user_function0'], angle_to_plot[type][i], axis[type]["ekin"][i])
                        A_arg = self.np.argwhere(abs(axis[type]['user_function0'])<=FWHM_rad/2)
                        A_energies = self.np.sum(Eng_Den[:,A2_arg],axis=1)
                    elif j == 0:
                        A_arg = self.np.argwhere(axis[type]['user_function0']-self.np.radians(AngleOffset)==abs(axis[type]['user_function0']-self.np.radians(AngleOffset)).min())[0]
                        A_energies = Eng_Den[:,A0_arg]
                    else:
                        A_arg = self.np.argwhere(abs(axis[type]['user_function0']-self.np.radians(AngleOffset))<=self.np.radians(j))
                        A_energies = self.np.reshape(self.np.sum(Eng_Den[:,A_arg],axis=1),Eng_Den.shape[0])
                    return axis[type]["ekin"][i], A_energies
            elif not DataOnly:
                print(f"\nPlotting {type} angle energies")
                for i in range(self.TimeSteps.size):
                    fig, ax = self.plt.subplots(num=1,clear=True)
                    SaveFile= TempFile + f"_{type}" 
                    if self.np.max(axis[type]['ekin'][i]) <= EMax[Species.index(type)]/10:
                        InitialFile=i+1
                        continue
                    Eng_Den = self.np.swapaxes(angle_to_plot[type][i], 0, 1)
                    for j in (Angles):
                        if j=='FWHM':
                            FWHM_rad, FWHM_deg = getFWHM(axis[type]['user_function0'], angle_to_plot[type][i], axis[type]["ekin"][i])
                            A2_arg = self.np.argwhere(abs(axis[type]['user_function0'])<=FWHM_rad/2)
                            A4_arg = self.np.argwhere(abs(axis[type]['user_function0'])<=FWHM_rad/4)
                            A2_energies = self.np.sum(Eng_Den[:,A2_arg],axis=1)
                            A4_energies = self.np.sum(Eng_Den[:,A4_arg],axis=1)
                            ax.plot(axis[type]["ekin"][i],A4_energies,label=fr'$\theta$ $\equal$ $\pm${FWHM_deg/4}$\degree$')
                            ax.plot(axis[type]["ekin"][i],A2_energies, label=fr'$\theta$ $\equal$ $\pm${FWHM_deg/2}$\degree$')
                        elif j == 0:
                            A0_arg = self.np.argwhere(axis[type]['user_function0']-self.np.radians(AngleOffset)==abs(axis[type]['user_function0']-self.np.radians(AngleOffset)).min())[0]
                            ax.plot(axis[type]["ekin"][i],Eng_Den[:,A0_arg], label=r'$\theta$ $\equal$ 0$\degree$')
                        else:
                            A_arg = self.np.argwhere(abs(axis[type]['user_function0']-self.np.radians(AngleOffset))<=self.np.radians(j))
                            A_energies = self.np.reshape(self.np.sum(Eng_Den[:,A_arg],axis=1),Eng_Den.shape[0])
                            ax.plot(axis[type]["ekin"][i][1:-1], A_energies, label=f"$\\theta$ $\\equal$ $\\pm${j}$\\degree$" if AngleOffset==0 else f"$\\theta$ $\\equal$ {AngleOffset} $\\pm${j}$\\degree$")
                    ax.set_yscale('log')
                    ax.set_xlim(0,EMax[Species.index(type)])
                    ax.set_ylim(1e4 if YMin is None else YMin, 1e10 if YMax is None else YMax)
                    ax.set_xlabel('Energy [MeV/u]')
                    ax.set_ylabel('dnde [1/MeV $\\mu$m$^3$]')
                    ax.grid(True)
                    ax.legend()
                    ax.set_title(f"{axis[type]['Time'][i]}fs")
                    fig.tight_layout()
                    self.plt.savefig(self.raw_path + '/' + SaveFile + '_' + str(i) + '.png',dpi=200)
                    if self.Log: 
                        PrintPercentage(i, self.TimeSteps.size -1 )
                print(f"\nAngle energies saved in {self.raw_path}")
                if self.Movie:
                    MakeMovie(self.raw_path, self.pros_path, InitialFile, self.TimeSteps.size, SaveFile)
                    print(f"\nMovies saved in {self.pros_path}")
            
    def HiResPlot(self, Species=[], CBMin=None, CBMax=None, File=None):
        if not Species:
            raise ValueError("No species were provided")
        if not isinstance(Species, list):
            Species = [Species]
        den_to_plot={}
        axis={}
        label={}
        TempFile=File if File is not None else "hi_res_densities"
        for type in Species:
            Diag=type + ' density hi res'
            if Diag not in self.Simulation.getDiags("ParticleBinning")[1] and type != "rel electron":
                raise ValueError(f"Diagnostic '{Diag}' is not a valid hi-res density diagnostic")
            if type == "rel electron":
                elec_den, axis[type] = self.GetData("ParticleBinning", "electron density hi res", units=self.Units, x_offset=self.x_spot)
                en_den = self.GetData("ParticleBinning", "electron energy density hi res", Axis=False, x_offset=self.x_spot)
                den_to_plot[type] = self.np.array(elec_den) / ((self.np.array(en_den) / self.np.array(elec_den)) + 1)
                label[type] = type
                continue
            den_to_plot[type], axis[type] = self.GetData("ParticleBinning", Diag, units=self.Units, x_offset=self.x_spot)
            label[type] = type
        
        print(f"\nPlotting {Species} hi-res densities")
        for i in range(self.TimeSteps.size):
            if len(Species) == 1:
                fig, ax = self.plt.subplots(num=1,clear=True)
                for type in Species:
                    SaveFile=TempFile if File is not None else f"{type}_" + TempFile
                    den = self.np.swapaxes(den_to_plot[type][i], 0, 1)
                    cax=ax.pcolormesh(axis[type]['x'], axis[type]['y'], den, cmap='jet', norm=self.cm.LogNorm(vmin=1e-2 if CBMin is None else CBMin, vmax=1e3 if CBMax is None else CBMax))
                    cbar = fig.colorbar(cax, aspect=50)
                    cbar.set_label('N [$N_c$]')
                    ax.set_xlabel(r'x [$\mu$m]')
                    ax.set_ylabel(r'y [$\mu$m]')
                    ax.set_title(f"{axis[type]['Time'][i]}")
            
            else:
                fig, ax = self.plt.subplots(nrows=len(Species), num=1,clear=True, sharex=True)
                for type in Species:
                    SaveFile=TempFile if File is not None else f"{type}_" + TempFile
                    den = self.np.swapaxes(den_to_plot[type][i], 0, 1)
                    cax=ax[Species.index(type)].pcolormesh(axis[type]['x'], axis[type]['y'], den, cmap='jet', norm=self.cm.LogNorm(vmin=1e-2 if CBMin is None else CBMin, vmax=1e3 if CBMax is None else CBMax))
                    cbar = fig.colorbar(cax, aspect=50)
                    cbar.set_label('N [$N_c$]')
                    ax[Species.index(type)].set_ylabel(r'y [$\mu$m]')
                    ax[Species.index(type)].set_title(f"{label[type]}")
                ax[-1].set_xlabel(r'x [$\mu$m]')
            fig.suptitle(f"{axis[type]['Time'][i]}fs")
            fig.tight_layout()
            self.plt.savefig(self.raw_path + '/' + SaveFile + '_' + str(i) + '.png',dpi=200)
            if self.Log: 
                PrintPercentage(i, self.TimeSteps.size -1 )
        print(f"\nHi-res densities saved in {self.raw_path}")
        if self.Movie:
            MakeMovie(self.raw_path, self.pros_path, 0, self.TimeSteps.size, SaveFile)
            print(f"\nMovies saved in {self.pros_path}")

    def CDSurfacePlot(self, F_Spot=0, CBMin=None, CBMax=None, XMin=None, XMax=None, tMax=None, HiRes=False, File=None):
        if F_Spot == 0:
            raise ValueError("No focal spot was provided")
        elif F_Spot < 1:
            F_Spot = F_Spot/self.micro
        if tMax is not None and tMax < 1:
            tMax = tMax*1e15
        if HiRes:
            elec_den, axis = self.GetData("ParticleBinning", "electron density hi res", units=self.Units, x_offset=self.x_spot)
            en_den = self.GetData("ParticleBinning", "electron energy density hi res", Axis=False, x_offset=self.x_spot)
        else:
            elec_den, axis = self.GetData("ParticleBinning", "electron density", units=self.Units, x_offset=self.x_spot)
            en_den = self.GetData("ParticleBinning", "electron energy density", Axis=False, x_offset=self.x_spot)
        den_to_plot = self.np.array(elec_den) / ((self.np.array(en_den) / self.np.array(elec_den)) + 1)

        if XMin is not None and XMin < axis['x'].min():
            print(f"XMin is less than the minimum x value, setting XMin to {axis['x'].min()}")
            XMin = axis['x'].min()
        if XMax is not None and XMax > axis['x'].max():
            print(f"XMax is greater than the maximum x value, setting XMax to {axis['x'].max()}")
            XMax = axis['x'].max()
        CD_Surf, DenTime = getCDSurf(axis['x'], axis['y'], den_to_plot, F_Spot, self.TimeSteps.size)
        cp=(self.Tau*1e15)/(2*self.np.sqrt(2*self.np.log(2)))
        test=Gau(axis["Time"], 1.0, 0.0, cp)
        Trans, TTrans = GoTrans(CD_Surf, self.Tau, axis["Time"])
        SaveFile=File if File is not None else "rel_cd_surface"

        fig =self.plt.figure(figsize=(8,5),num=1,clear=True)
        gs = self.gs.GridSpec(1,2,width_ratios=[1,4])
        ax1 = self.plt.subplot(gs[0])
        ax2 = self.plt.subplot(gs[1], sharey=ax1)

        print(f"\nPlotting relativistic critical density surface")
        den = self.np.swapaxes(DenTime, 0, 1)
        cax=ax2.pcolormesh(axis["x"],axis["Time"],den, cmap=self.cmaps.batlow_r, norm=self.cm.LogNorm(vmin=1e-2 if CBMin is None else CBMin, vmax=1e3 if CBMax is None else CBMax))
        ax2.plot(CD_Surf,axis["Time"], 'k--', label=r'$\gamma$ N$_c$')
        if Trans:
            ax2.arrow(-1. if XMin is None else XMin, TTrans, 0.5 if XMin is None else abs(XMin)/2, 0, head_width=4, head_length=0.1 if XMin is None else abs(XMin)/10, ec='r', ls='--', label=f"Trans @ {TTrans}fs")
        ax2.legend()
        ax1.plot(test,axis["Time"],'r-')
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.set_xticks([])
        ax1.tick_params(axis='both', which='both', bottom=False)
        cbar=fig.colorbar(cax)
        cbar.set_label(r'$\gamma$N$_e$ [$N_c$]')
        ax2.set_xlabel(r'x [$\mu$m]')
        ax2.set_ylabel(r't [$fs$]')
        ax2.set_xlim(-1. if XMin is None else XMin, 1. if XMax is None else XMax)
        ax1.set_ylim(top=50 if tMax is None else tMax)
        self.plt.subplots_adjust(wspace=0.25)
        ax2.set_title('Electron Density and\nRelativistic Critical Density')
        fig.tight_layout()
        self.plt.savefig(self.pros_path + '/' + SaveFile + '.png',dpi=200)
        print(f"\nCritical density surface saved in {self.raw_path}")

    def EngTimePlot(self, Species=[], tMin=None, File=None):
        if not Species:
            raise ValueError("No species were provided")
        if not isinstance(Species, list):
            Species = [Species]
        if tMin is not None and tMin < 1:
            tMin = tMin*1e15
        spectra_to_plot={}
        axis={}
        label={}
        for type in Species:
            Diag=type + ' spectra'
            if Diag not in self.Simulation.getDiags("ParticleBinning")[1]:
                raise ValueError(f"Diagnostic '{Diag}' is not a valid density diagnostic")
            spectra_to_plot[type], axis[type] = self.GetData("ParticleBinning", Diag, units=self.Units)
            label[type] = type
        
        print(f"\nPlotting {Species} densities over time")
        SaveFile=f"{File}" if File is not None else f"energy_time"
        
        fig, ax = self.plt.subplots(num=1,clear=True)
        Max_Energy = {}
        E_derv = {}
        for type in Species:
            EMax = []
            EDerv = []
            for i in range(self.TimeSteps.size):
                EMax.append(axis[type]["ekin"][i].max())
            Max_Energy[type] = EMax
            for i in range(1,len(Max_Energy[type])):
                EDerv.append((Max_Energy[type][i]-Max_Energy[type][i-1])/(axis[type]["Time"][i]-axis[type]["Time"][i-1]))
            E_derv[type] = EDerv
            ax.plot(axis[type]['Time'], Max_Energy[type], label=f"{label[type]}")
            ax.set_xlabel('t [$fs$]')
            ax.set_ylabel('E [$MeV/u$]')
            ax.set_xlim(left=-2*self.Tau*1e15 if tMin is None else tMin)
            ax.legend()
            ax.grid()
            ax.set_title('Max Energy')
            fig.tight_layout()
        self.plt.savefig(self.pros_path + '/' + SaveFile + '.png',dpi=200)
        for type in Species:
            Derv_SaveFile=f"{type}_" + SaveFile + "_derv"
            fig, ax = self.plt.subplots(num=1,clear=True) 
            ax3 = ax.twinx()
            lns2 = ax3.plot(axis[type]['Time'][1:], E_derv[type], label=f"dE/dt", color='r')
            lns1 = ax.plot(axis[type]['Time'], Max_Energy[type], label=f"Max Energy", color='b')
            ax.set_xlabel('t [$fs$]')
            ax.set_ylabel('E [$MeV$]')
            ax3.set_ylabel('dE/dt [$MeV/fs$]')
            lns = lns1+lns2
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs)
            ax.set_title('Energy Derivative')
            fig.tight_layout()
            self.plt.savefig(self.pros_path + '/' + Derv_SaveFile + '.png',dpi=200)
        print(f"\nDensities over time saved in {self.raw_path}")

    def Y0(self, Species=None, E=None, Field=None, FSpot=0, FMax=None, YMin=None, YMax=None, XMin=None, XMax=None, File=None):
        if Species is None and E is None:
            raise ValueError("No species or E-fields were provided")
        if E:
            if Field is None:
                raise ValueError("No field was provided")
            if len(E) != len(Field):
                raise ValueError("E and Field must have the same length")
        if FSpot == 0:
            print("No focal spot was provided, defaulting to y-axis center")
        elif FSpot < 1:
            FSpot = FSpot/self.micro
        data_to_plot={}
        axis={}
        label={}
        if Species:
            for type in Species:
                Diag=type + ' density'
                if Diag not in self.Simulation.getDiags("ParticleBinning")[1] and type != "rel electron":
                    raise ValueError(f"Diagnostic '{Diag}' is not a valid density diagnostic")
                if type == "rel electron":
                    elec_den, axis[type] = self.GetData("ParticleBinning", "electron density", units=self.Units, x_offset=self.x_spot)
                    en_den = self.GetData("ParticleBinning", "electron energy density", Axis=False)
                    data_to_plot[type] = self.np.array(elec_den) / ((self.np.array(en_den) / self.np.array(elec_den)) + 1)
                    label[type] = type
                    continue
                data_to_plot[type], axis[type] = self.GetData("ParticleBinning", Diag, units=self.Units, x_offset=self.x_spot)
                label[type] = type
        if E:
            for type in E:
                if type not in self.Simulation.getDiags("Fields")[1]:
                    raise ValueError(f"Diagnostic '{type}' is not a valid spectra diagnostic")
                data_to_plot[type], axis[type] = self.GetData("Fields", type, Field=Field[E.index(type)], units=self.Units, x_offset=self.x_spot)
                label[type] = f"{type.split(' ')[0][0:2]} {Field[E.index(type)]}"
                
        print(f"\nPlotting line-out averaged over y±{FSpot/2}")
        SaveFile=File if File is not None else "y0"
        colours=['b','g','y','m','c']
        for i in range(self.TimeSteps.size):
            lnf=None
            fig, ax = self.plt.subplots(num=1,clear=True, sharex=True)
            if Species:
                ax.set_ylabel('N [$N_c$]')
                ax.set_yscale('log')
                ax.set_ylim(1e-2 if YMin is None else YMin, 1e3 if YMax is None else YMax)
                for type in Species:
                    if FSpot != 0 : args=self.np.argwhere(abs(axis[type]["y"])<=(FSpot/2))
                    else: args=self.np.argwhere(abs(axis[type]["y"])==self.np.min(abs(axis[type]["y"])))
                    lns=ax.plot(axis[type]['x'], self.np.mean(data_to_plot[type][i][:,args],axis=1), colours[Species.index(type)], label=f"{label[type]}")
                    lnf=lns if lnf is None else lnf+lns
                ax.hlines(1, axis[type]['x'][0], axis[type]['x'][-1], 'k')
                ax.text(-5 if XMin is None else XMin, 1, 'Critical Density', fontsize=8)
                if E:
                    ax2=ax.twinx()
                    ax2.set_ylabel('E [V/m]')
                    ax2.set_ylim(-self.max_number if FMax is None else -FMax, self.max_number if FMax is None else FMax)
                    for type in E:
                        args=self.np.argwhere(abs(axis[type]["y"])<=(FSpot/2))
                        if type == "average fields":
                            lns=ax2.plot(axis[type]['x'], self.np.mean(data_to_plot[type][i][:,args],axis=1), 'r', label=f"{label[type]}")
                        elif type == "instant fields":
                            lns=ax2.plot(axis[type]['x'], self.np.mean(data_to_plot[type][i][:,args],axis=1), 'k--', label=f"{label[type]}")
                        lnf=lns if lnf is None else lnf+lns
            elif E:
                ax.set_ylabel('E [V/m]')
                ax.set_ylim(-self.max_number if FMax is None else -FMax, self.max_number if FMax is None else FMax)

                for type in E:
                    args=self.np.where(abs(axis[type]["y"])<=(FSpot/2))[0]
                    if type == "average fields":
                        lns=ax.plot(axis[type]['x'], self.np.mean(data_to_plot[type][i][:,args],axis=1), 'r', label=f"{label[type]}")
                    elif type == "instant fields":
                        lns=ax.plot(axis[type]['x'], self.np.mean(data_to_plot[type][i][:,args],axis=1), 'k--', label=f"{label[type]}")
                    lnf=lns if lnf is None else lnf+lns
            ax.set_xlim(-5 if XMin is None else XMin, 5 if XMax is None else XMax)
            ax.set_title(f"{axis[type]['Time'][i]}fs")
            ax.set_xlabel(r'x [$\mu$m]')
            labs= [l.get_label() for l in lnf]
            ax.legend(lnf, labs)
            fig.tight_layout()
            self.plt.savefig(self.raw_path + '/' + SaveFile + '_' + str(i) + '.png',dpi=200)
            if self.Log: 
                PrintPercentage(i, self.TimeSteps.size -1 )
        print(f"\nLine-outs saved in {self.raw_path}")
        if self.Movie:
            MakeMovie(self.raw_path, self.pros_path, 0, self.TimeSteps.size, SaveFile)
            print(f"\nMovies saved in {self.pros_path}")
    
    def TempPlot(self, Species=None, Test=False, XMin=None, XMax=None, File=None):
        if Species is None:
            raise ValueError("No species were provided")
        if not isinstance(Species, list):
            Species = [Species]
        
        data_to_plot={}
        axis={}
        Tempfile=File if File is not None else "temp"
        for type in Species:
            Diag=type + ' spectra'
            if Diag not in self.Simulation.getDiags("ParticleBinning")[1]:
                raise ValueError(f"Diagnostic '{Diag}' is not a valid density diagnostic")
            data_to_plot[type], axis[type] = self.GetData("ParticleBinning", Diag, units=self.Units)

        print(f"\nPlotting {Species} temperatures")
        for type in Species:
            SaveFile=f"{type}_" + Tempfile
            for t in range(self.TimeSteps.size):
                dnde = self.np.log(data_to_plot[type][t])
                eng = axis[type]['ekin'][t]
                eng = eng[dnde>-12.5]
                dnde = dnde[dnde>-12.5]
                eng = eng[~self.np.isnan(dnde)]
                dnde = dnde[~self.np.isnan(dnde)]
                eng = eng[~self.np.isinf(dnde)]
                dnde = dnde[~self.np.isinf(dnde)]
                try: poly = self.np.polyfit(eng,dnde, 1)
                except TypeError: 
                    if t == 0:
                        Temp = self.np.nan
                    else: Temp = self.np.vstack((Temp, self.np.nan))
                else:
                    exponential = self.np.poly1d(poly)
                    if t == 0:
                        Temp = -1/(poly[0])
                    else: Temp = self.np.vstack((Temp, -1/(poly[0])))
                    if Test:
                        fig, ax = self.plt.subplots()
                        ax.plot(axis[type]['ekin'][t], self.np.log(data_to_plot[type][t]), 'k,', label='Data')
                        ax.plot(eng, exponential(eng), 'r-', label=f'Fit- {-1/(poly[0]):.2e} MeV')
                        ax.set(xlabel='Energy (MeV)', ylabel='dnde', title=f'{axis[type]["Time"][t]}fs')
                        ax.legend()
                        fig.tight_layout()
                        self.plt.savefig(self.raw_path + '/' + SaveFile + 'Tmp_' + str(t) + '.png',dpi=200)

            fig, ax = self.plt.subplots()
            ax.plot(axis[type]['Time'], Temp, 'k-')
            ax.set(xlabel='Time (fs)', xlim=(axis[type]['Time'][0] if XMin == None else XMin, axis[type]['Time'][-1] if XMax == None else XMax), ylabel='Temperature (MeV)', title='Temperature')
            fig.tight_layout()
            self.plt.savefig(self.pros_path + '/' + SaveFile + '.png',dpi=200)

    def Help(self):
        print(f"""\n
Available functions:
        - _init__(SimName=".", x_spot=0, Tau=0, Ped=None, Dim=2, Log=True, Movie=True)
              
        - AnglePlot(Species=[], CBMin=None, CBMax=None, XMax=None, LasAngle=None, File=None)
              
        - AngleEnergyPlot(Species=[], YMin=None, YMax=None, Angles=[], AngleOffset=0, File=None)
              
        - CDSurfacePlot(F_Spot=0, CBMin=None, CBMax=None, XMin=None, XMax=None, tMax=None, HiRes=False, File=None)
              
        - DensityPlot(Species=[], E_las=False, E_avg=False, Field=None, EMax=None, Colours=None, CBMin=None, CBMax=None, File=None)
              
        - EngTimePlot(Species=[], tMin=None, File=None)
              
        - GetData(Diag, Name, Field=None, units=None, Data=True, Axis=True, ProsData=True, x_offset=None, y_offset=None)

        - HiResPlot(Species=[], CBMin=None, CBMax=None, File=None)

        - PhaseSpacePlot(Species=[], Phase=None, CBMin=None, CBMax=None, YMin=None, YMax=None, XMin=None, XMax=None, File=None)
              
        - SpectraPlot(Species=[], XMax=None, YMin=None, YMax=None, File=None, ProsData=True)
              
        - TempPlot(Species=None, Test=False, XMin=None, XMax=None, File=None)
              
        - Y0(Species=None, E=None, Field=None, FSpot=0, FMax=None, YMin=None, YMax=None, XMin=None, XMax=None, File=None)

Current Simulation:
        - {self.Simulation}

Saving Raw Images:
        - {self.raw_path}

Saving Processed Images:
        - {self.pros_path}
""")
        
    def TestingPlot(self, Species=['proton'], PPC=None, Den=None, File=None):
        ratio = Den/PPC
        spect_to_plot={}
        axis={}
        label={}
        SaveFile='testing'
        x_max = 0
        type='proton'
        Diag=type + ' spectra'
        spect_to_plot[type], axis[type] = self.GetData("ParticleBinning", Diag, units=self.Units)
        label[type] = type
        for i in range(self.TimeSteps.size):
            if self.np.max(axis[type]['ekin'][i]) > x_max:
                x_max = self.np.max(axis[type]['ekin'][i])
        for i in range(self.TimeSteps.size):
            fig, ax = self.plt.subplots(num=1,clear=True)
            ax.plot(axis[type]['ekin'][i], spect_to_plot[type][i], label='N$_c$')
            ax.plot(axis[type]['ekin'][i], spect_to_plot[type][i]*ratio, label=f"*{ratio}")
            ax.set_xlabel('E [$MeV$]')
            ax.set_xlim(0,x_max)
            ax.set_ylim(1e-5, 1e1)
            ax.set_yscale('log')
            ax.yaxis.grid(True)
            ax2=ax.twinx()
            ylims = ax.get_ylim()
            yticks = ax.get_yticks()
            energy_lims = self.np.array(ylims)*ratio
            energy_ticks = self.np.array(yticks)*ratio
            ax2.set_yscale('log')
            ax2.set_ylim(0.15e-5, 0.15e1)
            # ax2.set_yticks(energy_ticks)
            # print(energy_ticks)
            # ax2.set_yticklabels([f"{tick}' for tick in energy_ticks])
            ax.set_ylabel('dNdE [$N_c$]')
            
            ax.legend()
            ax.set_title(f"{axis[type]['Time'][i]}fs")
            fig.tight_layout()
            self.plt.savefig(self.raw_path + '/' + SaveFile + '_' + str(i) + '.png',dpi=200)
            if self.Log: 
                PrintPercentage(i, self.TimeSteps.size -1 )