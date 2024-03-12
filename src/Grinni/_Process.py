from ._Utils import Gau, getFWHM, getCDSurf, GoTrans, PrintPercentage

class Process():
    def __init__(self, SimName=".",x_spot=0,Tau=0,Log=True):
        ########### Constants ##################################
        self.c = 299792458. 
        self.me = 9.11e-31
        self.epsilon0 = 8.854187e-12
        self.e = 1.602176e-19
        self.amu = 1.673776e-27
        self.massNeutron = 1838. # in units of electron mass
        self.massProton = 1836.

        self.MeV_to_J = 1.6e-13
        self.micro = 1e-6
        self.nano = 1e-9
        self.pico = 1e-12
        self.femto = 1e-15
        ########################################################
        import happi
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        import matplotlib.gridspec as gridspec
        import os
        self.os=os
        self.happi = happi
        self.np = np
        self.plt = plt
        self.cm = colors
        self.gs = gridspec
        self.SimName = SimName
        self.TimeSteps = None
        self.Log = Log
        self.Units = ["um", "fs", "MeV", "V/m", "kg*m/s"]
        self.Simulation = self.happi.Open(self.SimName, verbose=False)
        if self.Simulation == "Invalid Smilei simulation":
            raise ValueError(f"Simulation {self.SimName} does not exist")
        else: print(f"\nSimulation {self.SimName} loaded")
        self.x_spot = x_spot 
        if self.x_spot > 1:
            print("\nx_spot is in meters, converting to micrometers")
            self.x_spot = self.x_spot*self.micro
        self.Tau = Tau
        if self.Tau > 1:
            print("\nTau is in seconds, converting to femtoseconds")
            self.Tau = self.Tau*self.femto
        self.t0=((self.x_spot/self.c)+((2*self.Tau)/(2*self.np.sqrt(self.np.log(2)))))/self.femto

        folder_name = "graphs"
        simulation_path = self.os.path.abspath(self.SimName)
        self.folder_path = self.os.path.join(simulation_path, folder_name)
        if not(self.os.path.exists(self.folder_path) and self.os.path.isdir(self.folder_path)):
            self.os.mkdir(self.folder_path)
        print(f"\nGraphs will be saved in {self.folder_path}")
    
    def GetData(self, Diag, Name, Field=None, units=None, Data=True, Axis=True, get_new_tsteps=False, x_offset=None, y_offset=None):
        # Check if Diag is a valid diagnostic
        if not self.Simulation.getDiags(Diag):
            raise ValueError(f"Diag {Diag} is not a valid diagnostic")
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
            gridA=self.Simulation.namelist.Main.grid_length[0]*self.Simulation.namelist.Main.grid_length[1]
            axis_names=['x', 'y', 'user_function0', 'ekin', 'px']
        elif Diag == "Fields":
            if units is None:
                MetaData = self.Simulation.Field(Name, Field)
            elif units is not None:
                MetaData = self.Simulation.Field(Name, Field, units=units)
            axis_names=['x', 'y']
        else: raise ValueError(f"Diag {Diag} is not a valid diagnostic")

        Values = MetaData.getData()
        if self.TimeSteps is None or get_new_tsteps is True:
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
            axis_data = MetaData.getAxis(axis_name)
            if len(axis_data)==0:
                    continue
            if axis_name == "x":
                axis_data = axis_data - x_offset if x_offset is not None else axis_data  
            if axis_name == "y":
                axis_data = axis_data - y_offset if y_offset is not None else axis_data - ((axis_data[0]+axis_data[-1])/2)
            if axis_name == "user_function0":
                if bin_size is None:
                    bin_size=(axis_data[1]-axis_data[0])*gridA
                else: bin_size = bin_size*(axis_data[1]-axis_data[0])
            elif axis_name == "ekin":
                if "carbon" in Name:
                    Z=6
                elif "proton" in Name:
                    Z=1
                axis_data = self.np.array(MetaData.getAxis('ekin', timestep=self.TimeSteps[0]),ndmin=2)
                for t in self.TimeSteps[1:]:
                    tmp=self.np.array(MetaData.getAxis('ekin', timestep=t)/Z,ndmin=2)
                    axis_data = self.np.append(axis_data,tmp, axis=0)
                if bin_size is None:
                    bin_size=(axis_data[1]-axis_data[0])*gridA
                else: bin_size = bin_size*(axis_data[1]-axis_data[0])
            elif axis_name == "px":
                axis_data = self.np.array(MetaData.getAxis('px', timestep=self.TimeSteps[0]),ndmin=2)
                for t in self.TimeSteps[1:]:
                    tmp=self.np.array(MetaData.getAxis('px', timestep=t),ndmin=2)
                    axis_data = self.np.append(axis_data,tmp, axis=0)
            axis[axis_name] = axis_data
        if Data: 
            if Axis: return Values*bin_size if bin_size is not None else Values, axis
            else: return Values*bin_size if bin_size is not None else Values
        elif Axis: return axis
        else: raise ValueError("No data or axis requested")
        
    def DensityPlot(self, Species=[], E_las=False, E_avg=False, Field=None, Min=None, Max=None, x_offset=None, y_offset=None, File=None):
        if not Species:
            raise ValueError("No species were provided")
        if not isinstance(Species, list):
            Species = [Species]
        if (E_las or E_avg) and Field is None:
            raise ValueError("No field was provided")
        if E_las:
            Ey, E_axis = self.GetData("Fields", "instant fields", Field=Field, units=self.Units, x_offset=10 if x_offset is None else x_offset, y_offset=10 if y_offset is None else y_offset)
        elif E_avg:
            Ey, E_axis = self.GetData("Fields", "average fields", Field=Field, units=self.Units, x_offset=10 if x_offset is None else x_offset, y_offset=10 if y_offset is None else y_offset)
        
        den_to_plot={}
        axis={}
        SaveFile=File if File is not None else "density"
        for type in Species:
            Diag=type + ' density'
            if Diag not in self.Simulation.getDiags("ParticleBinning")[1] and type != "rel electron":
                raise ValueError(f"Diagnostic {type} is not a valid density diagnostic")
            if type == "rel electron":
                elec_den, axis[type] = self.GetData("ParticleBinning", "electron density", units=self.Units, x_offset=10 if x_offset is None else x_offset)
                en_den = self.GetData("ParticleBinning", "electron energy density", Axis=False, x_offset=10 if x_offset is None else x_offset)
                den_to_plot[type] = self.np.array(elec_den) / ((self.np.array(en_den) / self.np.array(elec_den)) + 1)
                continue
            den_to_plot[type], axis[type] = self.GetData("ParticleBinning", Diag, units=self.Units, x_offset=10 if x_offset is None else x_offset)
        
        print(f"\nPlotting {Species} densities")
        for i in range(self.TimeSteps.size):
            fig, ax = self.plt.subplots(num=1,clear=True)
            if E_las or E_avg:
                field = self.np.swapaxes(Ey[i], 0, 1)
                cax1=ax.pcolormesh(E_axis['x'], E_axis['y'], field, cmap='bwr', norm=self.cm.CenteredNorm(halfrange=self.max_number))
                cbar1 = fig.colorbar(cax1, aspect=50)
                cbar1.set_label('E$_y$ [V/m]')
            for type in Species:
                SaveFile=SaveFile if File is not None else f"{type}_" + SaveFile
                den = self.np.swapaxes(den_to_plot[type][i], 0, 1)
                cax=ax.pcolormesh(axis[type]['x'], axis[type]['y'], den, cmap='jet', norm=self.cm.LogNorm(vmin=1e-2 if Min is None else Min, vmax=1e3 if Max is None else Max))
            cbar=fig.colorbar(cax, aspect=50)
            cbar.set_label('N [$N_c$]')
            ax.set_xlabel(r'x [$\mu$m]')
            ax.set_ylabel(r'y [$\mu$m]')
            ax.set_title(f'{axis[type]['Time'][i]}fs')
            fig.tight_layout()
            self.plt.savefig(self.folder_path + '/' + SaveFile + '_' + str(i) + '.png',dpi=200)
            if self.Log: 
                PrintPercentage(i, self.TimeSteps.size)
        print(f"\nDensities saved in {self.folder_path}")
            
    def SpectraPlot(self, Species=[], Min=None, Max=None, File=None):
        if not Species:
            raise ValueError("No species were provided")
        if not isinstance(Species, list):
            Species = [Species]
        spect_to_plot={}
        axis={}
        label={}
        SaveFile=File if File is not None else "energies"
        x_max = 0
        for type in Species:
            Diag=type + ' spectra'
            if Diag not in self.Simulation.getDiags("ParticleBinning")[1] and type != "rel electron":
                raise ValueError(f"Diagnostic {type} is not a valid spectra diagnostic")
            if type == "rel electron":
                elec_den, axis[type] = self.GetData("ParticleBinning", "electron density", units=self.Units)
                en_den = self.GetData("ParticleBinning", "electron energy density", Axis=False)
                spect_to_plot[type] = self.np.array(elec_den) / ((self.np.array(en_den) / self.np.array(elec_den)) + 1)
                continue
            spect_to_plot[type], axis[type] = self.GetData("ParticleBinning", Diag, units=self.Units)
            label[type] = type
        
        print(f"\nPlotting {Species} spectra")
        for i in range(self.TimeSteps.size):
            fig, ax = self.plt.subplots(num=1,clear=True)
            for type in Species:
                SaveFile=SaveFile if File is not None else f"{type}_" + SaveFile
                ax.plot(axis[type]['Ekin'][i], spect_to_plot[type][i], label=f'{label[type]}fs')
                if axis[type]['Ekin'][-1] > x_max:
                    x_max = axis[type]['Ekin'][-1]
            ax.set_xlabel('E [$MeV$]')
            ax.set_xlim(0,x_max)
            ax.set_ylim(1e-7 if Min is None else Min,1e1 if Max is None else Max)
            ax.set_ylabel('dNdE')
            ax.set_yscale('log')
            ax.legend()
            ax.set_title(f'{axis[type]["Time"][i]}fs')
            self.plt.savefig(self.folder_path + '/' + SaveFile + '_' + str(i) + '.png',dpi=200)
            if self.Log: 
                PrintPercentage(i, self.TimeSteps.size)
        print(f"\nSpectra saved in {self.folder_path}")

    def PhaseSpacePlot(self, Species=[], Phase=None, Min=None, Max=None, File=None):
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
        SaveFile=File if File is not None else f"{Phase}_phase"
        for type in Species:
            Diag=type + ' ' + Phase + '  phase space'
            if Diag not in self.Simulation.getDiags("ParticleBinning")[1]:
                raise ValueError(f"Diagnostic {type} is not a valid phase space diagnostic")
            phase_to_plot[type], axis[type] = self.GetData("ParticleBinning", Species, units=self.Units, x_offset=10)
            label[type] = type
        
        print(f"\nPlotting {Species} phase spaces")
        for i in range(self.TimeSteps.size):
            fig, ax = self.plt.subplots(num=1,clear=True)
            for type in Species:
                SaveFile=SaveFile if File is not None else f"{type}_" + SaveFile
                momentum = self.np.swapaxes(phase_to_plot[type][i], 0, 1)
                ax.pcolormesh(axis[type]['x'], axis[type]['Ekin'], momentum, cmap='jet', norm=self.cm.LogNorm(vmin=0.01 if Min is None else Min, vmax=1e2 if Max is None else Max))
                ax.set_ylim(axis[type]['Ekin'][-1].min(),axis[type]['Ekin'][-1].max())
            ax.set_xlabel(r'x [$\mu$m]')
            ax.set_ylabel('px [kgms$^{-1}$]')
            ax.set_title(f'{axis[type]["Time"][i]}fs')
            self.plt.savefig(self.folder_path + '/' + SaveFile + '_' + str(i) + '.png',dpi=200)
            if self.Log: 
                PrintPercentage(i, self.TimeSteps.size)
        print(f"\nPhase spaces saved in {self.folder_path}")
    
    def AnglePlot(self, Species=[], Min=None, Max=None, File=None):
        if not Species:
            raise ValueError("No species were provided")
        if not isinstance(Species, list):
            Species = [Species]
        angle_to_plot={}
        axis={}
        label={}
        SaveFile=File if File is not None else "angles"
        for type in Species:
            Diag=type + ' angle'
            if Diag not in self.Simulation.getDiags("ParticleBinning")[1]:
                raise ValueError(f"Diagnostic {type} is not a valid angle diagnostic")
            angle_to_plot[type], axis[type] = self.GetData("ParticleBinning", Diag, units=self.Units, x_offset=10)
            label[type] = type
        
        print(f"\nPlotting {Species} angles")
        for i in range(self.TimeSteps.size):
            if len(Species) == 1:
                fig, ax = self.plt.subplots(num=1,clear=True, subplot_kw={'projection': 'polar'})
                for type in Species:
                    SaveFile=SaveFile if File is not None else f"{type}_" + SaveFile
                    angles = self.np.swapaxes(angle_to_plot[type][i], 0, 1)
                    ax.pcolormesh(axis[type]['user_function0'],axis[type]['ekin'][i], angles, cmap='jet', norm=self.cm.LogNorm(vmin=1e-8 if Min is None else Min, vmax=1e-4 if Max is None else Max))
                    ax.set_xlim(-self.np.pi/3,self.np.pi/3)
                    ax.set_ylim(0,axis[type]['ekin'][-1].max())
                    ax.set_title(f'{label[type]}')
            else:
                fig, ax = self.plt.subplots(ncols=len(Species), num=1,clear=True, subplot_kw={'projection': 'polar'})
                for type in Species:
                    SaveFile=SaveFile if File is not None else f"{type}_" + SaveFile
                    angles = self.np.swapaxes(angle_to_plot[type][i], 0, 1)
                    ax[Species.index(type)].pcolormesh(axis[type]['user_function0'],axis[type]['ekin'][i], angles, cmap='jet', norm=self.cm.LogNorm(vmin=1e-8 if Min is None else Min, vmax=1e-4 if Max is None else Max))
                    ax[Species.index(type)].set_title(f'{label[type]}')
                    ax[Species.index(type)].set_xlim(-self.np.pi/3,self.np.pi/3)
                    ax[Species.index(type)].set_ylim(0,axis[type]['ekin'][-1].max())
            
            fig.suptitle(f'{axis[type]["Time"][i]}fs')
            self.plt.savefig(self.folder_path + '/' + SaveFile + '_' + str(i) + '.png',dpi=200)
            if self.Log: 
                PrintPercentage(i, self.TimeSteps.size)
        print(f"\nAngles saved in {self.folder_path}")

    def AngleEnergyPlot(self, Species=[], p0=[], File=None):
        if not Species:
            raise ValueError("No species were provided")
        if not isinstance(Species, list):
            Species = [Species]
        angle_to_plot={}
        axis={}
        label={}
        SaveFile=File if File is not None else f"angles_energy"
        for type in Species:
            Diag=type + ' angle'
            if Diag not in self.Simulation.getDiags("ParticleBinning")[1]:
                raise ValueError(f"Diagnostic {type} is not a valid angle diagnostic")
            angle_to_plot[type], axis[type] = self.GetData("ParticleBinning", Diag, units=self.Units, x_offset=10)
            label[type] = type
        
        print(f"\nPlotting {Species} angle energies")
        for i in range(self.TimeSteps.size):
            for type in Species:
                fig, ax = self.plt.subplots(num=1,clear=True)
                SaveFile=SaveFile if File is not None else f"{type}_" + SaveFile
                FWHM_rad, FWHM_deg = getFWHM(axis[type]['user_function0'], axis[type]["Ekin"], p0)
                Eng_Den = self.np.swapaxes(angle_to_plot[type][i], 0, 1)
                A0_arg = self.np.argwhere(axis[type]['user_function0']==abs(axis[type]['user_function0']).min())[0]
                A2_arg = self.np.argwhere(abs(axis[type]['user_function0'])<=FWHM_rad/2)
                A4_arg = self.np.argwhere(abs(axis[type]['user_function0'])<=FWHM_rad/4)
                A2_energies = self.np.sum(Eng_Den[:,A2_arg],axis=1)
                A4_energies = self.np.sum(Eng_Den[:,A4_arg],axis=1)
                ax.plot(axis[type]["Ekin"][i],Eng_Den[:,A0_arg], label=r'$\theta$ $\equal$ 0$\degree$')
                ax.plot(axis[type]["Ekin"][i],A4_energies,label=fr'$\theta$ $\equal$ {FWHM_deg/4}$\degree$')
                ax.plot(axis[type]["Ekin"][i],A2_energies,label=fr'$\theta$ $\equal$ {FWHM_deg/2}$\degree$')
                ax.set_yscale('log')
                ax.set_xlim(0,axis[type]["Ekin"][-1].max())
                ax.set_xlabel('Energy [MeV/u]')
                ax.set_ylabel('dnde [N_c dn]')
                ax.legend()
                ax.set_title(f'{axis["Time"][i]}fs')
                fig.tight_layout()
                self.plt.savefig(self.folder_path + '/' + SaveFile + '_' + str(i) + '.png',dpi=200)
            if self.Log: 
                PrintPercentage(i, self.TimeSteps.size)
        print(f"\nAngle energies saved in {self.folder_path}")
            
    def HiResPlot(self, Species=[], Min=None, Max=None, x_offset=None, y_offset=None, File=None):
        if not Species:
            raise ValueError("No species were provided")
        if not isinstance(Species, list):
            Species = [Species]
        den_to_plot={}
        axis={}
        label={}
        SaveFile=File if File is not None else "hi_res_densities"
        for type in Species:
            Species=type + ' density hi res'
            if Species not in self.Simulation.getDiags("ParticleBinning")[1] and type != "rel electron":
                raise ValueError(f"Diagnostic {type} is not a valid hi-res density diagnostic")
            if type == "rel electron":
                elec_den, axis[type] = self.GetData("ParticleBinning", "electron density hi res", units=self.Units, x_offset=10 if x_offset is None else x_offset, get_new_tsteps=True)
                en_den = self.GetData("ParticleBinning", "electron energy density hi res", Axis=False, x_offset=10 if x_offset is None else x_offset)
                den_to_plot[type] = self.np.array(elec_den) / ((self.np.array(en_den) / self.np.array(elec_den)) + 1)
                continue
            den_to_plot[type], axis[type] = self.GetData("ParticleBinning", Species, units=self.Units, x_offset=10 if x_offset is None else x_offset, get_new_tsteps=True)
            label[type] = type
        
        print(f"\nPlotting {Species} hi-res densities")
        for i in range(self.TimeSteps.size):
            if len(Species) == 1:
                fig, ax = self.plt.subplots(num=1,clear=True)
                for type in Species:
                    SaveFile=SaveFile if File is not None else f"{type}_" + SaveFile
                    den = self.np.swapaxes(den_to_plot[type][i], 0, 1)
                    cax=ax.pcolormesh(axis[type]['x'], axis[type]['y'], den, cmap='jet', norm=self.cm.LogNorm(vmin=1e-2 if Min is None else Min, vmax=1e3 if Max is None else Max))
                    cbar = fig.colorbar(cax, aspect=50)
                    cbar.set_label('N [$N_c$]')
                    ax.set_xlabel(r'x [$\mu$m]')
                    ax.set_ylabel(r'y [$\mu$m]')
                    ax.set_title(f'{axis[type]["Time"][i]}fs')
            
            else:
                fig, ax = self.plt.subplots(nrows=len(Species), num=1,clear=True)
                for type in Species:
                    SaveFile=SaveFile if File is not None else f"{type}_" + SaveFile
                    den = self.np.swapaxes(den_to_plot[type][i], 0, 1)
                    cax=ax[Species.index(type)].pcolormesh(axis[type]['x'], axis[type]['y'], den, cmap='jet', norm=self.cm.LogNorm(vmin=1e-2 if Min is None else Min, vmax=1e3 if Max is None else Max))
                    cbar = fig.colorbar(cax, aspect=50)
                    cbar.set_label('N [$N_c$]')
                    ax[Species.index(type)].set_xlabel(r'x [$\mu$m]')
                    ax[Species.index(type)].set_ylabel(r'y [$\mu$m]')
                    ax[Species.index(type)].set_title(f'{label[type]}fs')
            fig.suptitle(f'{axis[type]["Time"][i]}fs')
            fig.tight_layout()
            self.plt.savefig(self.folder_path + '/' + SaveFile + '_' + str(i) + '.png',dpi=200)
            if self.Log: 
                PrintPercentage(i, self.TimeSteps.size)
        print(f"\nHi-res densities saved in {self.folder_path}")

    def CDSurfacePlot(self, Min=None, Max=None, x_offset=None, y_offset=None, File=None):
        elec_den, axis = self.GetData("ParticleBinning", "electron density hi res", units=self.Units, x_offset=10 if x_offset is None else x_offset, get_new_tsteps=True)
        en_den = self.GetData("ParticleBinning", "electron energy density hi res", Axis=False, x_offset=10 if x_offset is None else x_offset)
        den_to_plot = self.np.array(elec_den) / ((self.np.array(en_den) / self.np.array(elec_den)) + 1)
        CD_Surf = getCDSurf(axis['x'], axis['Time'], den_to_plot, self.x_spot, self.TimeSteps.size)
        cp=(self.Tau*1e15)/(2*self.np.sqrt(2*self.np.log(2)))
        test=Gau(axis["Time"], 1.0, 0.0, cp)
        Trans, TTrans = GoTrans(CD_Surf, self.Tau, axis["Time"])
        SaveFile=File if File is not None else "rel_cd_surface"

        fig =self.plt.figure(figsize=(8,5),num=1,clear=True)
        gs = self.gs.GridSpec(1,2,width_ratios=[1,4])
        ax1 = self.plt.subplot(gs[0])
        ax2 = self.plt.subplot(gs[1], sharey=ax1)

        print(f"\nPlotting relativistic critical density surface")
        den = self.np.swapaxes(den_to_plot, 0, 1)
        cax=ax2.pcolormesh(axis["x"],axis["Time"],den, cmap='jet', norm=self.cm.LogNorm(vmin=1e-2 if Min is None else Min, vmax=1e3 if Max is None else Max))
        ax2.plot(CD_Surf,axis["Time"], 'k--', label=r'$\gamma$ N$_c$')
        if Trans:
            ax2.hlines(TTrans,-1,-0.5, 'r', label=f'Trans @ {TTrans}fs')
        ax2.legend()
        ax1.plot(test,axis["Time"],'r-')
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.tick_params(axis='both', which='both', bottom=False)
        cbar=fig.colorbar(cax)
        cbar.set_label(r'$\gamma$N$_e$ [$N_c$]')
        ax2.set_xlabel(r'x [$\mu$m]')
        ax2.set_ylabel(r't [$fs$]')
        ax2.set_xlim(-1.,1.)
        ax1.set_ylim(top=50)
        self.plt.subplots_adjust(wspace=0.25)
        ax2.set_title('Electron Density and\nRelativistic Critical Density')
        fig.tight_layout()
        self.plt.savefig(self.folder_path + '/' + SaveFile + '.png',dpi=200)
        print(f"\nCritical density surface saved in {self.folder_path}")

    def DenTimePlot(self, Species=[], File=None):
        if not Species:
            raise ValueError("No species were provided")
        if not isinstance(Species, list):
            Species = [Species]
        spectra_to_plot={}
        axis={}
        label={}
        for type in Species:
            Species=type + ' spectra'
            if Species not in self.Simulation.getDiags("ParticleBinning")[1] and type != "rel electron":
                raise ValueError(f"Diagnostic {type} is not a valid density diagnostic")
            if type == "rel electron":
                elec_den, axis[type] = self.GetData("ParticleBinning", "electron density", units=self.Units, get_new_tsteps=True)
                en_den = self.GetData("ParticleBinning", "electron energy density", Axis=False)
                spectra_to_plot[type] = self.np.array(elec_den) / ((self.np.array(en_den) / self.np.array(elec_den)) + 1)
                continue
            spectra_to_plot[type], axis[type] = self.GetData("ParticleBinning", Species, units=self.Units, get_new_tsteps=True)
            label[type] = type
        
        print(f"\nPlotting {Species} densities over time")
        SaveFile=f"{File}" if File is not None else f"energy_time"
        
        for type in Species:
            Derv_SaveFile=f"{type}_" + SaveFile + "_derv"
            Max_Energy = []
            E_derv = []
            for i in range(self.TimeSteps.size):
                Max_Energy.append(spectra_to_plot[type][i].max())
            for i in range(1,len(Max_Energy)):
                E_derv.append((Max_Energy[i]-Max_Energy[i-1])/(axis[type]["Time"][i]-axis[type]["Time"][i-1]))
            fig, ax = self.plt.subplots(num=1,clear=True)
            ax.plot(axis[type]['Time'], Max_Energy, label=f'{label[type]}')
            ax.set_xlabel('t [$fs$]')
            ax.set_ylabel('E [$MeV$]')
            ax.legend()
            ax.set_title('Max Energy')
            fig.tight_layout()

            fig2, ax2 = self.plt.subplots(num=1,clear=True)  
            ax2.plot(axis[type]['Time'][1:], E_derv, label=f'{label[type]}')
            ax2.plot(axis[type]['Time'], Max_Energy, label=f'{label[type]}')
            ax2.set_xlabel('t [$fs$]')
            ax2.set_ylabel('dE/dt [$MeV/fs$]')
            ax2.legend()
            ax2.set_title('Energy Derivative')
            fig2.tight_layout()
            fig2.savefig(self.folder_path + '/' + Derv_SaveFile + '.png',dpi=200)
        fig.savefig(self.folder_path + '/' + SaveFile + '.png',dpi=200)
        print(f"\nDensities over time saved in {self.folder_path}")