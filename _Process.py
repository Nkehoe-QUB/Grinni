class Process():
    def __init__(self, SimName="."):
        import happi
        import numpy as np
        import matplotlib.pyplot as plt
        self.happi = happi
        self.np = np
        self.plt = plt
        self.SimName = SimName
        self.Simulation = self.happi.Open(SimName)
        self.TimeSteps = None
        self.Diags = {
            "ParticleBinning": self.Simulation.getDiags("ParticleBinning")[1],
            "Field": self.Simulation.getDiags("Fields")[1],
            "Probe": self.Simulation.getDiags("Probes")[1],
            "Screen": self.Simulation.getDiags("Screen")[1],
            "Scalar": self.Simulation.getDiags("Scalar")[1]
        }
    
    def GetData(self, Diag, Name, units=None, Axis=True, x_offset=0, y_offset=0):
        # Check if Diag is a valid diagnostic
        if Diag not in self.Diags.keys():
            raise ValueError(f"Diag {Diag} is not a valid diagnostic")
        # Check if units is a list of strings
        if units is not None:
            if not (isinstance(units, list) and all(isinstance(unit, str) for unit in units)):
                raise TypeError("units must be a list of strings")
        
        # Get the data
        if Diag == "ParticleBinning":
            if units is None:
                MetaData = self.Simulation.ParticleBinning(Name)
            elif units is not None:
                MetaData = self.Simulation.ParticleBinning(Name, units=self.happi.Units(units))
            Data = MetaData.getData()
            gridA=self.Simulation.namelist.Main.grid_length[0]*self.Simulation.namelist.Main.grid_length[1]
            if self.TimeSteps is None:
                self.TimeSteps = self.np.array(MetaData.getTimesteps())
            if Axis:
                axis =[]
                bin_size = None
                for axis_name in ['x', 'y', 'user_function0', 'Ekin', 'px']:
                    axis_data = MetaData.getAxis(axis_name)
                    if not axis_data:
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
                    axis.append(axis_data)
                return Data*bin_size if bin_size is not None else Data, axis
            else: return Data
        
        elif Diag == "Field":
            if units is None:
                Data = self.Simulation.Field(Name)
            elif units is not None:
                Data = self.Simulation.Field(Name, units=self.happi.Units(units))
            if self.TimeSteps is None:
                self.TimeSteps = self.np.array(Data.getTimesteps())
            if Axis:
                x = Data.getAxis('x')
                x = x - x_offset
                y = Data.getAxis('y')
                y = y - y_offset
                return Data.getData(), x, y
            else: return Data.getData()
    
    def ProtonDensity(self):
        pass