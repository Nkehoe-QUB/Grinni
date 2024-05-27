from ._Utils import Gau, getFWHM, getCDSurf, GoTrans, PrintPercentage, MakeMovie

class Process():
    def __init__(self, SimName=".", x_spot=0, Tau=0, x_sim=0, y_sim=0, Ped=None, Dim=2, Log=True, Movie=True):
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
        import happi
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        import matplotlib.gridspec as gridspec
        import os
        import pandas as pd
        self.pd = pd
        self.os=os
        self.happi = happi
        self.np = np
        self.plt = plt
        self.cm = colors
        self.gs = gridspec
        self.SimName = SimName
        self.Dim = Dim
        self.Log = Log
        self.Movie = Movie
        self.Units = ["um", "fs", "MeV", "V/m", "kg*m/s", 'um^-3*MeV^-1', 'um^-3*kg^-1*(m/s)^-1']
        self.Simulation = self.happi.Open(self.SimName, verbose=False)
        if self.Simulation == "Invalid Smilei simulation":
            raise ValueError(f"Simulation {self.SimName} does not exist")
        else: print(f"\nSimulation {self.SimName} loaded")
        if x_sim == 0 or y_sim == 0 or x_spot == 0 or Tau == 0:
            raise ValueError("No spot size or simulation size or Tau value was provided")
        self.x_sim = x_sim
        if self.x_sim < 1:
            print("\nx_sim is in meters, converting to micrometers")
            self.x_sim = self.x_sim/self.micro
        self.y_sim = y_sim
        if self.y_sim < 1:
            print("\ny_sim is in meters, converting to micrometers")
            self.y_sim = self.y_sim/self.micro
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
        folder_name = "graphs"
        self.simulation_path = self.os.path.abspath(self.SimName)
        self.folder_path = self.os.path.join(self.simulation_path, folder_name)
        if not(self.os.path.exists(self.folder_path) and self.os.path.isdir(self.folder_path)):
            self.os.mkdir(self.folder_path)
        print(f"\nGraphs will be saved in {self.folder_path}")
        self.video_path = self.os.path.join(self.simulation_path, "videos")
        if not(self.os.path.exists(self.video_path) and self.os.path.isdir(self.video_path)):
            self.os.mkdir(self.video_path)
        print(f"\nVideos will be saved in {self.video_path}")
    
    def moving_average(self, x, w):
        return self.np.convolve(x, self.np.ones(w), 'valid') / w

    def GetData(self, Diag, Name, Field=None, units=None, Data=True, Axis=True, get_new_tsteps=False, x_offset=None, y_offset=None):
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
                gridA=self.Simulation.namelist.Main.grid_length[0]*self.Simulation.namelist.Main.grid_length[1] 
                axis_names=['x', 'y', 'user_function0', 'ekin', 'px', 'py']
            elif self.Dim == 3:
                gridA = self.Simulation.namelist.Main.grid_length[0]*self.Simulation.namelist.Main.grid_length[1]*self.Simulation.namelist.Main.grid_length[-1]
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
                bin_size = ((self.x_sim*self.y_sim) if self.Dim==2 else (self.x_sim*self.y_sim*self.y_sim) ) if bin_size is None else bin_size*((self.x_sim*self.y_sim) if self.Dim==2 else (self.x_sim*self.y_sim*self.y_sim))
                axis_data = self.np.array(MetaData.getAxis('ekin', timestep=self.TimeSteps[0])/Z)
                for t in self.TimeSteps[1:]:
                    axis_data=self.np.vstack((axis_data, MetaData.getAxis('ekin', timestep=t)/Z))
            elif axis_name == "px":
                if "x-px" in Name:
                    bin_size = ((self.x_sim*self.y_sim*(axis['x'][1]-axis['x'][0])) if self.Dim==2 else (self.x_sim*self.y_sim*self.y_sim*(axis['x'][1]-axis['x'][0])) ) if bin_size is None else bin_size*((self.x_sim*self.y_sim*(axis['x'][1]-axis['x'][0])) if self.Dim==2 else (self.x_sim*self.y_sim*self.y_sim*(axis['x'][1]-axis['x'][0])))
                axis_data = self.np.array(MetaData.getAxis('px', timestep=self.TimeSteps[0]),ndmin=2)
                for t in self.TimeSteps[1:]:
                    axis_data = self.np.vstack(axis_data,self.np.array(MetaData.getAxis('px', timestep=t)))
            elif axis_name == "py":
                axis_data = self.np.array(MetaData.getAxis('py', timestep=self.TimeSteps[0]),ndmin=2)
                tmp_bin=[(axis_data[0][1]-axis_data[0][0])]
                for t in self.TimeSteps[1:]:
                    tmp=self.np.array(MetaData.getAxis('py', timestep=t),ndmin=2)
                    axis_data = self.np.append(axis_data,tmp, axis=0)
                    tmp_bin.append((tmp[0][1]-tmp[0][0]))
                if bin_size is None:
                    bin_size=self.np.array(tmp_bin*gridA)
                else:
                    bin_size = self.np.array(bin_size * tmp_bin)
            axis[axis_name] = axis_data
        if Data: 
            if Axis: return Values * bin_size if bin_size is not None else Values, axis
            else: return Values * bin_size if bin_size is not None else Values
        elif Axis: return axis
        else: raise ValueError("No data or axis requested")
        
    def DensityPlot(self, Species=[], E_las=False, E_avg=False, Field=None, Colours=None, Min=None, Max=None, x_offset=None, y_offset=None, File=None):
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
            Ey, E_axis = self.GetData("Fields", "instant fields", Field=Field, units=self.Units, x_offset=10 if x_offset is None else x_offset, y_offset=10 if y_offset is None else y_offset)
        elif E_avg:
            Ey, E_axis = self.GetData("Fields", "average fields", Field=Field, units=self.Units, x_offset=10 if x_offset is None else x_offset, y_offset=10 if y_offset is None else y_offset)
        
        den_to_plot={}
        axis={}
        TempFile=File if File is not None else "density"
        if Species:
            for type in Species:
                Diag=type + ' density'
                if Diag not in self.Simulation.getDiags("ParticleBinning")[1] and type != "rel electron":
                    raise ValueError(f"Diagnostic '{Diag}' is not a valid density diagnostic")
                if type == "rel electron":
                    elec_den, axis[type] = self.GetData("ParticleBinning", "electron density", units=self.Units, x_offset=10 if x_offset is None else x_offset)
                    en_den = self.GetData("ParticleBinning", "electron energy density", Axis=False, x_offset=10 if x_offset is None else x_offset)
                    den_to_plot[type] = self.np.array(elec_den) / ((self.np.array(en_den) / self.np.array(elec_den)) + 1)
                    continue
                den_to_plot[type], axis[type] = self.GetData("ParticleBinning", Diag, units=self.Units, x_offset=10 if x_offset is None else x_offset)
        
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
                    cax1=ax.pcolormesh(E_axis['x'], E_axis['y'], field, cmap='bwr', norm=self.cm.CenteredNorm(halfrange=self.max_number))
                    cbar1 = fig.colorbar(cax1, aspect=50)
                    cbar1.set_label('E$_y$ [V/m]')
            if Species:
                for type in Species:
                    SaveFile=TempFile if File is not None else f"{type}_" + TempFile
                    den = self.np.swapaxes(den_to_plot[type][i], 0, 1)
                    cax=ax.pcolormesh(axis[type]['x'], axis[type]['y'], den, cmap='jet' if Colours is None else Colours[Species.index(type)], norm=self.cm.LogNorm(vmin=1e-2 if Min is None else Min, vmax=1e3 if Max is None else Max))
                    if (Colours is not None) and (len(Colours) > 1) and (not E_las or not E_avg):
                        cbar=fig.colorbar(cax, aspect=50)
                        cbar.set_label(f'N$_{{{type}}}$ [$N_c$]')
                if (Colours is None) or (len(Colours) == 1):
                    cbar=fig.colorbar(cax, aspect=50)
                    cbar.set_label('N [$N_c$]')
            ax.grid(True)
            ax.set_xlabel(r'x [$\mu$m]')
            ax.set_ylabel(r'y [$\mu$m]')
            if Species: ax.set_title(f'{axis[type]['Time'][i]}fs')
            else: ax.set_title(f'{E_axis["Time"][i]}fs')
            fig.tight_layout()
            if not Species: SaveFile=TempFile if File is not None else f"{Field}_" + TempFile
            self.plt.savefig(self.folder_path + '/' + SaveFile + '_' + str(i) + '.png',dpi=200)
            if self.Log: 
                PrintPercentage(i, self.TimeSteps.size)
        print(f"\nDensities saved in {self.folder_path}")
        if self.Movie:
            MakeMovie(self.folder_path, self.video_path, 0, FinalFile, SaveFile)
            print(f"\nMovies saved in {self.video_path}")
            
    def SpectraPlot(self, Species=[], Min=None, Max=None, xMax=None, File=None):
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
                elec_den, axis[type] = self.GetData("ParticleBinning", "electron density", units=self.Units)
                en_den = self.GetData("ParticleBinning", "electron energy density", Axis=False)
                spect_to_plot[type] = self.np.array(elec_den) / ((self.np.array(en_den) / self.np.array(elec_den)) + 1)
                continue
            spect_to_plot[type], axis[type] = self.GetData("ParticleBinning", Diag, units=self.Units)
            label[type] = type
        
        print(f"\nPlotting {Species} spectra")
        for type in Species:
            with open(self.os.path.join(self.simulation_path, f'{type}_MaxE.csv'), 'w') as file:
                file.write('Time,Energy\n')
                for i in range(self.TimeSteps.size):
                    args=axis[type]['ekin'][i]>0
                    file.write(f'{axis[type]["Time"][i]},{axis[type]['ekin'][i][args][-1]}\n')
                    if self.np.max(axis[type]['ekin'][i]) > x_max:
                        x_max = self.np.max(axis[type]['ekin'][i])
            dfs = []
            for i in range(self.TimeSteps.size):
                df =self.pd.DataFrame({
                    'Time':axis[type]['Time'][i],
                    'Energy':axis[type]['ekin'][i],
                    'dNdE':spect_to_plot[type][i]
                    })
                dfs.append(df)
            dfs = self.pd.concat(dfs)
            with open(self.os.path.join(self.simulation_path, f'{type}_energy.csv'), 'w') as file:
                dfs.to_csv(file, index=False)

            print(f"\n{type} energies saved in {self.simulation_path}")
        for i in range(self.TimeSteps.size):
            fig, ax = self.plt.subplots(num=1,clear=True)
            for type in Species:
                SaveFile=TempFile if File is not None else f"{type}_" + TempFile
                ax.plot(axis[type]['ekin'][i], spect_to_plot[type][i], label=f'{label[type]}')
            ax.set_xlabel('E [$MeV$]')
            ax.set_xlim(0,x_max if xMax is None else xMax)
            ax.set_ylim(1e4 if Min is None else Min,1e11 if Max is None else Max)
            ax.set_ylabel('dNdE [MeV$^{-1}$]')
            ax.set_yscale('log')
            ax.legend()
            ax.set_title(f'{axis[type]["Time"][i]}fs')
            fig.tight_layout()
            self.plt.savefig(self.folder_path + '/' + SaveFile + '_' + str(i) + '.png',dpi=200)
            if self.Log: 
                PrintPercentage(i, self.TimeSteps.size)
        print(f"\nSpectra saved in {self.folder_path}")
        if self.Movie:
            MakeMovie(self.folder_path, self.video_path, 0, self.TimeSteps.size, SaveFile)
            print(f"\nMovies saved in {self.video_path}")

    def PhaseSpacePlot(self, Species=[], Phase=None, Min=None, Max=None, x_offset=None, File=None):
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
            phase_to_plot[type], axis[type] = self.GetData("ParticleBinning", Diag, units=self.Units, x_offset=10 if x_offset is None else x_offset)
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
            if "px" in Phase:
                for i in range(self.TimeSteps.size):
                    if Phase[1] == "py" :
                        if self.np.max(axis[type][Phase[1]][i]) > max1:
                            max1 = self.np.max(axis[type][Phase[1]][i])
                        if self.np.min(axis[type][Phase[1]][i]) < min1:
                            min1 = self.np.min(axis[type][Phase[1]][i])
                        
                    if self.np.max(axis[type]["px"][i]) > max0:
                        max0 = self.np.max(axis[type]["px"][i])
                    if self.np.min(axis[type]["px"][i]) < min0:
                        min0 = self.np.min(axis[type]["px"][i])
            for i in range(self.TimeSteps.size):
                fig, ax = self.plt.subplots(num=1,clear=True)
                SaveFile=TempFile if File is not None else f"{type}_" + TempFile
                momentum = self.np.swapaxes(phase_to_plot[type][i], 0, 1)
                X = axis[type][Phase[0]] if Phase[0] not in ["px" or "py"] else axis[type][Phase[0]][i]
                Y = axis[type][Phase[1]][i]
                cax = ax.pcolormesh(X, Y, momentum, cmap='jet', norm=self.cm.LogNorm(vmin=1e2 if Min is None else Min, vmax=1e10 if Max is None else Max))
                cbar = fig.colorbar(cax, aspect=50)
                if Phase[1] == "px":
                    ax.set_xlabel(r'x [$\mu$m]')
                    ax.set_ylabel('px [kgm/s]')
                    ax.set_ylim(min0,max0)
                    cbar.set_label('dndpx [(kgm/s)$^{-1}$]')
                elif Phase[1] == "py":
                    ax.set_ylabel('py [kgms$^{-1}$]')
                    ax.set_xlabel('px [kgms$^{-1}$]')
                    ax.set_ylim(min1,max1)
                    ax.set_xlim(min0,max0)
                elif Phase[1] == "ekin":
                    ax.set_xlabel(r'x [$\mu$m]')
                    ax.set_ylabel('Energy [MeV]')
                    ax.set_ylim(0,1e3)
                cbar.set_label('dn')
                ax.set_title(f'{axis[type]["Time"][i]}fs')
                fig.tight_layout()
                self.plt.savefig(self.folder_path + '/' + SaveFile + '_' + str(i) + '.png',dpi=200)
                if self.Log: 
                    PrintPercentage(i, self.TimeSteps.size)
            print(f"\nPhase spaces saved in {self.folder_path}")
            if self.Movie:
                MakeMovie(self.folder_path, self.video_path, 0, self.TimeSteps.size, SaveFile)
                print(f"\nMovies saved in {self.video_path}")
    
    def AnglePlot(self, Species=[], Min=None, Max=None, File=None):
        if not Species:
            raise ValueError("No species were provided")
        if not isinstance(Species, list):
            Species = [Species]
        angle_to_plot={}
        axis={}
        label={}
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
            for i in range(self.TimeSteps.size):
                if self.np.max(axis[type]['ekin'][i]) > x_max:
                    x_max = self.np.max(axis[type]['ekin'][i])
            EMax.append(x_max)
        for i in range(self.TimeSteps.size):
            if len(Species) == 1:
                fig, ax = self.plt.subplots(num=1,clear=True, subplot_kw={'projection': 'polar'})
                for type in Species:
                    SaveFile=TempFile if File is not None else f"{type}_" + TempFile
                    angles = self.np.swapaxes(angle_to_plot[type][i], 0, 1)
                    try: ax.pcolormesh(axis[type]['user_function0'],axis[type]['ekin'][i], angles, cmap='jet', norm=self.cm.LogNorm(vmin=1e4 if Min is None else Min, vmax=1e10 if Max is None else Max))
                    except ValueError: 
                        continue
                    ax.set_xlim(-self.np.pi/3,self.np.pi/3)
                    ax.set_ylim(0,EMax[0])
                    ax.set_title(f'{label[type]}')
            else:
                fig, ax = self.plt.subplots(ncols=len(Species), num=1,clear=True, subplot_kw={'projection': 'polar'})
                for type in Species:
                    SaveFile=TempFile if File is not None else f"{type}_" + TempFile
                    angles = self.np.swapaxes(angle_to_plot[type][i], 0, 1)
                    ax[Species.index(type)].pcolormesh(axis[type]['user_function0'],axis[type]['ekin'][i], angles, cmap='jet', norm=self.cm.LogNorm(vmin=1e4 if Min is None else Min, vmax=1e10 if Max is None else Max))
                    ax[Species.index(type)].set_title(f'{label[type]}')
                    ax[Species.index(type)].set_xlim(-self.np.pi/3,self.np.pi/3)
                    ax[Species.index(type)].set_ylim(0,EMax[Species.index(type)])
            
            fig.suptitle(f'{axis[type]["Time"][i]}fs')
            fig.tight_layout()
            self.plt.savefig(self.folder_path + '/' + SaveFile + '_' + str(i) + '.png',dpi=200)
            if self.Log: 
                PrintPercentage(i, self.TimeSteps.size)
        print(f"\nAngles saved in {self.folder_path}")
        if self.Movie:
            MakeMovie(self.folder_path, self.video_path, 0, self.TimeSteps.size, SaveFile)
            print(f"\nMovies saved in {self.video_path}")

    def AngleEnergyPlot(self, Species=[], Min=[], Max=[], Angles=[], File=None):
        if not Species:
            raise ValueError("No species were provided")
        if not isinstance(Species, list):
            Species = [Species]
        if not isinstance(Angles, list):
            Angles = [Angles]
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
        
        print(f"\nPlotting {Species} angle energies")
        EMax=[]
        InitialFile=0
        for type in Species:
            x_max=0
            for i in range(self.TimeSteps.size):
                    if self.np.max(axis[type]['ekin'][i]) > x_max:
                        x_max = self.np.max(axis[type]['ekin'][i])
            EMax.append(x_max)

            # dfs = []
            # for i in range(self.TimeSteps.size):
            #     for a in range(len(axis[type]['user_function0'])):
            #         df =self.pd.DataFrame({
            #             'Time':axis[type]['Time'][i],
            #             'Energy':axis[type]['ekin'][i],
            #             'Angle':axis[type]['user_function0'],
            #             'dNdE':angle_to_plot[type][i][a,:]
            #             })
            #     dfs.append(df)
            # dfs = self.pd.concat(dfs)
            # with open(self.os.path.join(self.simulation_path, f'{type}_AngEnergy.csv'), 'w') as file:
            #     dfs.to_csv(file, index=False)

            # print(f"\n{type} energies saved in {self.simulation_path}")
        for type in Species:
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
                        A0_arg = self.np.argwhere(axis[type]['user_function0']==abs(axis[type]['user_function0']).min())[0]
                        ax.plot(axis[type]["ekin"][i],Eng_Den[:,A0_arg], label=r'$\theta$ $\equal$ 0$\degree$')
                    else:
                        A_arg = self.np.argwhere(abs(axis[type]['user_function0'])<=self.np.radians(j))
                        A_energies = self.np.reshape(self.np.sum(Eng_Den[:,A_arg],axis=1),Eng_Den.shape[0])
                        ax.plot(axis[type]["ekin"][i][1:-1], self.moving_average(A_energies, 3),  label=fr'$\theta$ $\equal$ $\pm${j}$\degree$')
                ax.set_yscale('log')
                ax.set_xlim(0,EMax[Species.index(type)])
                ax.set_ylim(1e4 if Min is None else Min, 1e10 if Max is None else Max)
                ax.set_xlabel('Energy [MeV/u]')
                ax.set_ylabel('dnde [$N_c$]')
                ax.legend()
                ax.set_title(f'{axis[type]["Time"][i]}fs')
                fig.tight_layout()
                self.plt.savefig(self.folder_path + '/' + SaveFile + '_' + str(i) + '.png',dpi=200)
                if self.Log: 
                    PrintPercentage(i, self.TimeSteps.size)
            print(f"\nAngle energies saved in {self.folder_path}")
            if self.Movie:
                MakeMovie(self.folder_path, self.video_path, InitialFile, self.TimeSteps.size, SaveFile)
                print(f"\nMovies saved in {self.video_path}")
            
    def HiResPlot(self, Species=[], Min=None, Max=None, x_offset=None, y_offset=None, File=None):
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
                elec_den, axis[type] = self.GetData("ParticleBinning", "electron density hi res", units=self.Units, x_offset=10 if x_offset is None else x_offset, get_new_tsteps=True)
                en_den = self.GetData("ParticleBinning", "electron energy density hi res", Axis=False, x_offset=10 if x_offset is None else x_offset)
                den_to_plot[type] = self.np.array(elec_den) / ((self.np.array(en_den) / self.np.array(elec_den)) + 1)
                label[type] = type
                continue
            den_to_plot[type], axis[type] = self.GetData("ParticleBinning", Diag, units=self.Units, x_offset=10 if x_offset is None else x_offset, get_new_tsteps=True)
            label[type] = type
        
        print(f"\nPlotting {Species} hi-res densities")
        for i in range(self.TimeSteps.size):
            if len(Species) == 1:
                fig, ax = self.plt.subplots(num=1,clear=True)
                for type in Species:
                    SaveFile=TempFile if File is not None else f"{type}_" + TempFile
                    den = self.np.swapaxes(den_to_plot[type][i], 0, 1)
                    cax=ax.pcolormesh(axis[type]['x'], axis[type]['y'], den, cmap='jet', norm=self.cm.LogNorm(vmin=1e-2 if Min is None else Min, vmax=1e3 if Max is None else Max))
                    cbar = fig.colorbar(cax, aspect=50)
                    cbar.set_label('N [$N_c$]')
                    ax.set_xlabel(r'x [$\mu$m]')
                    ax.set_ylabel(r'y [$\mu$m]')
                    ax.set_title(f'{axis[type]["Time"][i]}')
            
            else:
                fig, ax = self.plt.subplots(nrows=len(Species), num=1,clear=True, sharex=True)
                for type in Species:
                    SaveFile=TempFile if File is not None else f"{type}_" + TempFile
                    den = self.np.swapaxes(den_to_plot[type][i], 0, 1)
                    cax=ax[Species.index(type)].pcolormesh(axis[type]['x'], axis[type]['y'], den, cmap='jet', norm=self.cm.LogNorm(vmin=1e-2 if Min is None else Min, vmax=1e3 if Max is None else Max))
                    cbar = fig.colorbar(cax, aspect=50)
                    cbar.set_label('N [$N_c$]')
                    ax[Species.index(type)].set_ylabel(r'y [$\mu$m]')
                    ax[Species.index(type)].set_title(f'{label[type]}')
                ax[-1].set_xlabel(r'x [$\mu$m]')
            fig.suptitle(f'{axis[type]["Time"][i]}fs')
            fig.tight_layout()
            self.plt.savefig(self.folder_path + '/' + SaveFile + '_' + str(i) + '.png',dpi=200)
            if self.Log: 
                PrintPercentage(i, self.TimeSteps.size)
        print(f"\nHi-res densities saved in {self.folder_path}")
        if self.Movie:
            MakeMovie(self.folder_path, self.video_path, 0, self.TimeSteps.size, SaveFile)
            print(f"\nMovies saved in {self.video_path}")

    def CDSurfacePlot(self, F_Spot=0, Min=None, Max=None, tMax=None, x_offset=None, File=None):
        if F_Spot == 0:
            raise ValueError("No focal spot was provided")
        elif F_Spot < 1:
            F_Spot = F_Spot/self.micro
        if tMax is not None and tMax < 1:
            tMax = tMax*1e15
        elec_den, axis = self.GetData("ParticleBinning", "electron density hi res", units=self.Units, x_offset=10 if x_offset is None else x_offset, get_new_tsteps=True)
        en_den = self.GetData("ParticleBinning", "electron energy density hi res", Axis=False, x_offset=10 if x_offset is None else x_offset)
        den_to_plot = self.np.array(elec_den) / ((self.np.array(en_den) / self.np.array(elec_den)) + 1)
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
        cax=ax2.pcolormesh(axis["x"],axis["Time"],den, cmap='jet', norm=self.cm.LogNorm(vmin=1e-2 if Min is None else Min, vmax=1e3 if Max is None else Max))
        ax2.plot(CD_Surf,axis["Time"], 'k--', label=r'$\gamma$ N$_c$')
        if Trans:
            ax2.hlines(TTrans,-1,-0.5, 'r', label=f'Trans @ {TTrans}fs')
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
        ax2.set_xlim(-1.,1.)
        ax1.set_ylim(top=50 if tMax is None else tMax)
        self.plt.subplots_adjust(wspace=0.25)
        ax2.set_title('Electron Density and\nRelativistic Critical Density')
        fig.tight_layout()
        self.plt.savefig(self.folder_path + '/' + SaveFile + '.png',dpi=200)
        print(f"\nCritical density surface saved in {self.folder_path}")

    def DenTimePlot(self, Species=[], tMin=None, File=None):
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
            spectra_to_plot[type], axis[type] = self.GetData("ParticleBinning", Diag, units=self.Units, get_new_tsteps=True)
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
            ax.plot(axis[type]['Time'], Max_Energy[type], label=f'{label[type]}')
            ax.set_xlabel('t [$fs$]')
            ax.set_ylabel('E [$MeV$]')
            ax.set_xlim(left=-2*self.Tau*1e15 if tMin is None else tMin)
            ax.legend()
            ax.set_title('Max Energy')
            fig.tight_layout()
        self.plt.savefig(self.folder_path + '/' + SaveFile + '.png',dpi=200)
        for type in Species:
            Derv_SaveFile=f"{type}_" + SaveFile + "_derv"
            fig, ax = self.plt.subplots(num=1,clear=True) 
            ax3 = ax.twinx()
            lns2 = ax3.plot(axis[type]['Time'][1:], E_derv[type], label=f'dE/dt', color='r')
            lns1 = ax.plot(axis[type]['Time'], Max_Energy[type], label=f'Max Energy', color='b')
            ax.set_xlabel('t [$fs$]')
            ax.set_ylabel('E [$MeV$]')
            ax3.set_ylabel('dE/dt [$MeV/fs$]')
            lns = lns1+lns2
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs)
            ax.set_title('Energy Derivative')
            fig.tight_layout()
            self.plt.savefig(self.folder_path + '/' + Derv_SaveFile + '.png',dpi=200)
        print(f"\nDensities over time saved in {self.folder_path}")

    def Y0(self, Species=None, E=None, Field=None, FSpot=0, yMin=None, yMax=None, xMin=None, xMax=None, x_offset=None, y_offset=None, File=None):
        if Species and E is None:
            raise ValueError("No species or E-fields were provided")
        if Field is None:
            raise ValueError("No field was provided")
        if len(E) != len(Field):
            raise ValueError("E and Field must have the same length")
        if FSpot == 0:
            raise ValueError("No focal spot was provided")
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
                    elec_den, axis[type] = self.GetData("ParticleBinning", "electron density", units=self.Units, x_offset=10 if x_offset is None else x_offset)
                    en_den = self.GetData("ParticleBinning", "electron energy density", Axis=False)
                    data_to_plot[type] = self.np.array(elec_den) / ((self.np.array(en_den) / self.np.array(elec_den)) + 1)
                    label[type] = type
                    continue
                data_to_plot[type], axis[type] = self.GetData("ParticleBinning", Diag, units=self.Units, x_offset=10 if x_offset is None else x_offset)
                label[type] = type
        if E:
            for type in E:
                if type not in self.Simulation.getDiags("Fields")[1]:
                    raise ValueError(f"Diagnostic '{type}' is not a valid spectra diagnostic")
                data_to_plot[type], axis[type] = self.GetData("Fields", type, Field=Field[E.index(type)], units=self.Units, x_offset=10 if x_offset is None else x_offset, y_offset=10 if y_offset is None else y_offset)
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
                ax.set_ylim(1e-2 if yMin is None else yMin, 1e3 if yMax is None else yMax)
                for type in Species:
                    args=self.np.where(abs(axis[type]["y"])<=(FSpot/2))[0]
                    lns=ax.plot(axis[type]['x'], self.np.mean(data_to_plot[type][i][:,args],axis=1), colours[Species.index(type)], label=f'{label[type]}')
                    lnf=lns if lnf is None else lnf+lns
                ax.hlines(1, axis[type]['x'][0], axis[type]['x'][-1], 'k')
                ax.text(-5 if xMin is None else xMin, 1, 'Critical Density', fontsize=8)
                if E:
                    ax2=ax.twinx()
                    ax2.set_ylabel('E [V/m]')
                    ax2.set_ylim(-self.max_number, self.max_number)
                    for type in E:
                        args=self.np.where(abs(axis[type]["y"])<=(FSpot/2))[0]
                        if type == "average fields":
                            lns=ax2.plot(axis[type]['x'], self.np.mean(data_to_plot[type][i][:,args],axis=1), 'r', label=f'{label[type]}')
                        elif type == "instant fields":
                            lns=ax2.plot(axis[type]['x'], self.np.mean(data_to_plot[type][i][:,args],axis=1), 'k--', label=f'{label[type]}')
                        lnf=lns if lnf is None else lnf+lns
            elif E:
                ax.set_ylabel('E [V/m]')
                ax.set_ylim(-self.max_number, self.max_number)

                for type in E:
                    args=self.np.where(abs(axis[type]["y"])<=(FSpot/2))[0]
                    if type == "average fields":
                        lns=ax.plot(axis[type]['x'], self.np.mean(data_to_plot[type][i][:,args],axis=1), 'r', label=f'{label[type]}')
                    elif type == "instant fields":
                        lns=ax.plot(axis[type]['x'], self.np.mean(data_to_plot[type][i][:,args],axis=1), 'k--', label=f'{label[type]}')
                    lnf=lns if lnf is None else lnf+lns
            ax.set_xlim(-5 if xMin is None else xMin, 5 if xMax is None else xMax)
            ax.set_title(f'{axis[type]["Time"][i]}fs')
            ax.set_xlabel(r'x [$\mu$m]')
            labs= [l.get_label() for l in lnf]
            ax.legend(lnf, labs)
            fig.tight_layout()
            self.plt.savefig(self.folder_path + '/' + SaveFile + '_' + str(i) + '.png',dpi=200)
            if self.Log: 
                PrintPercentage(i, self.TimeSteps.size)
        print(f"\nLine-outs saved in {self.folder_path}")
        if self.Movie:
            MakeMovie(self.folder_path, self.video_path, 0, self.TimeSteps.size, SaveFile)
            print(f"\nMovies saved in {self.video_path}")

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
            ax.plot(axis[type]['ekin'][i], spect_to_plot[type][i]*ratio, label=f'*{ratio}')
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
            # ax2.set_yticklabels([f'{tick}' for tick in energy_ticks])
            ax.set_ylabel('dNdE [$N_c$]')
            
            ax.legend()
            ax.set_title(f'{axis[type]["Time"][i]}fs')
            fig.tight_layout()
            self.plt.savefig(self.folder_path + '/' + SaveFile + '_' + str(i) + '.png',dpi=200)
            if self.Log: 
                PrintPercentage(i, self.TimeSteps.size)