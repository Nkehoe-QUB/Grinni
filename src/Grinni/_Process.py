class Process():
    def __init__(self, SimName=".",x_spot=0,Tau=0):
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
        import os
        self.os=os
        self.happi = happi
        self.np = np
        self.plt = plt
        self.cm = colors
        self.SimName = SimName
        self.TimeSteps = None
        self.Simulation = self.happi.Open(self.SimName)
        if self.Simulation == "Invalid Smilei simulation":
            raise ValueError(f"Simulation {self.SimName} does not exist")
        self.x_spot = x_spot * self.micro
        self.Tau = Tau * self.femto
        self.t0=((self.x_spot/self.c)+((2*self.Tau)/(2*self.np.sqrt(self.np.log(2)))))/self.femto

        folder_name = "graphs"
        simulation_path = self.os.path.abspath(self.SimName)
        self.folder_path = self.os.path.join(simulation_path, folder_name)
        if not(self.os.path.exists(self.folder_path) and self.os.path.isdir(self.folder_path)):
            self.os.mkdir(self.folder_path)
    
    def GetData(self, Diag, Name, Field=None, units=None, Data=True, Axis=True, get_new_tsteps=False, x_offset=0, y_offset=0):
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
        if Diag == "ParticleBinning":
            if units is None:
                MetaData = self.Simulation.ParticleBinning(Name)
            elif units is not None:
                MetaData = self.Simulation.ParticleBinning(Name, units=units)
            gridA=self.Simulation.namelist.Main.grid_length[0]*self.Simulation.namelist.Main.grid_length[1]
            axis_names=['x', 'y', 'user_function0', 'Ekin', 'px']
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
            axis_data = axis_data - x_offset if axis_name == 'x' else axis_data - ((axis_data[0]+axis_data[-1])/2) if axis_name == 'y' else axis_data
            if axis_name == "user_function0":
                if bin_size is None:
                    bin_size=(axis_data[1]-axis_data[0])*gridA
                else: bin_size = bin_size*(axis_data[1]-axis_data[0])
            if axis_name == "Ekin":
                axis_data = self.np.array(MetaData.getAxis('Ekin', timestep=self.TimeSteps[0]),ndmin=2)
                for t in self.TimeSteps[1:]:
                    tmp=self.np.array(MetaData.getAxis('Ekin', timestep=t),ndmin=2)
                    axis_data = self.np.append(axis_data,tmp, axis=0)
                if bin_size is None:
                    bin_size=(axis_data[1]-axis_data[0])*gridA
                else: bin_size = bin_size*(axis_data[1]-axis_data[0])
            if axis_name == "px":
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
        
    def ProtonDensity(self, E_las=False, E_avg=False, Min=None, Max=None, Other_Densities=None):
        ProtonDensity, axis = self.GetData("ParticleBinning", "proton density", units=['um','fs'], x_offset=10)
        if E_las:
            Ey, E_axis = self.GetData("Fields", "instant fields", Field="Ey", units=['um','fs','V/m'], x_offset=10, y_offset=10)
        elif E_avg:
            Ey, E_axis = self.GetData("Fields", "average fields", Field="Ex", units=['um','fs', 'V/m'], x_offset=10, y_offset=10)
        
        if Other_Densities is not None:
            Other_den={}
            Other_axis={}
            for type in Other_Densities:
                if 'density' not in type:
                    raise ValueError(f"Diagnostic {type} is not a valid diagnostic")
                Other_den[type], Other_axis[type] = self.GetData("ParticleBinning", type, units=['um','fs'], x_offset=10)
        
        for i in range(self.TimeSteps.size):
            fig, ax = self.plt.subplots(num=1,clear=True)
            if E_las or E_avg:
                field = self.np.swapaxes(Ey[i], 0, 1)
                cax1=ax.pcolormesh(E_axis['x'],E_axis['y'], field, cmap='bwr', norm=self.cm.CenteredNorm(halfrange=self.max_number))
                cbar1 = fig.colorbar(cax1, aspect=50)
                cbar1.set_label('E$_y$ [V/m]')
            den = self.np.swapaxes(ProtonDensity[i], 0, 1)
            cax=ax.pcolormesh(axis['x'],axis['y'],den, cmap='jet', norm=self.cm.LogNorm(vmin=1e-2 if Min is None else Min, vmax=1e2 if Max is None else Max))
            cbar=fig.colorbar(cax, aspect=50)
            cbar.set_label('N$_p$ [$N_c$]')
            if Other_Densities is not None:
                for type in Other_Densities:
                    den = self.np.swapaxes(Other_den[type][i], 0, 1)
                    cax=ax.pcolormesh(Other_axis[type]['x'],Other_axis[type]['y'],den, cmap='jet', norm=self.cm.LogNorm(vmin=1e-2 if Min is None else Min, vmax=1e2 if Max is None else Max))
            ax.set_xlabel(r'x [$\mu$m]')
            ax.set_ylabel(r'y [$\mu$m]')
            ax.set_title(f'{axis['Time'][i]}fs')
            fig.tight_layout()
            self.plt.savefig(self.folder_path + '/proton_density_' + str(i) + '.png',dpi=200)