import Grinni
import sys
import os

'''
- Open(SimName=".", x_spot=0, Tau=0, Log=True)
    - SimName: Path to the simulation folder
    - x_spot: x position of the focal spot
    - Tau: Laser pulse duration
    - Log: If True, the graphng progress will be printed in the terminal

- GetData(Diag, Name, Field=None, units=None, Data=True, Axis=True, get_new_tsteps=False, x_offset=None, y_offset=None)
    - Diag: Type of the diagnostic
    - Name: Name of the diagnostic
    - Field: Field to be plotted
    - units: Units to be converted to
    - Data: If True, the data will be returned
    - Axis: If True, the axis will be returned
    - get_new_tsteps: If True, new time steps will be generated
    - x_offset: x zero point of the plot
    - y_offset: y zero point of the plot

- DensityPlot(Species=[], E_las=False, E_avg=False, Field=None, Min=None, Max=None, x_offset=None, y_offset=None, File=None)
    - Species: Species to be plotted
    - E_las: If True, the laser field will be plotted
    - E_avg: If True, the average E field over laser cycle will be plotted
    - Field: Field to be plotted
    - Min: Minimum value of the colour bar
    - Max: Maximum value of the colour bar
    - x_offset: x zero point of the plot
    - y_offset: y zero point of the plot
    - File: Name of the file to be saved

- SpectraPlot(Species=[], Min=None, Max=None, File=None)
    - Species: Species to be plotted
    - Min: Minimum value of the colour bar
    - Max: Maximum value of the colour bar
    - File: Name of the file to be saved

- PhaseSpacePlot(Species=[], Phase=None, Min=None, Max=None, File=None)
    - Species: Species to be plotted
    - Phase: Phase space to be plotted
    - Min: Minimum value of the colour bar
    - Max: Maximum value of the colour bar
    - File: Name of the file to be saved

- AnglePlot(Species=[], Min=None, Max=None, File=None)
    - Species: Species to be plotted
    - Min: Minimum value of the colour bar
    - Max: Maximum value of the colour bar
    - File: Name of the file to be saved

- AngleEnergyPlot(Species=[], p0=[], File=None)
    - Species: Species to be plotted
    - p0: Initial parameters for Gaussian fit
    - File: Name of the file to be saved

- HiResPlot(Species=[], Min=None, Max=None, x_offset=None, y_offset=None, File=None)
    - Species: Species to be plotted
    - Min: Minimum value of the colour bar
    - Max: Maximum value of the colour bar
    - x_offset: x zero point of the plot
    - y_offset: y zero point of the plot
    - File: Name of the file to be saved

- CDSurfacePlot(Min=None, Max=None, x_offset=None, y_offset=None, File=None)
    - Min: Minimum value of the colour bar
    - Max: Maximum value of the colour bar
    - x_offset: x zero point of the plot
    - y_offset: y zero point of the plot
    - File: Name of the file to be saved

- DenTimePlot(Species=[], File=None)
    - Species: Species to be plotted
    - File: Name of the file to be saved
'''
########### Constants ##################################
MeV_to_J = 1.6e-13
micro = 1e-6
nano = 1e-9
pico = 1e-12
femto = 1e-15

current_directory = sys.argv[1]
SimulationName = sys.argv[2]

SimulationPath = os.path.join(current_directory, SimulationName)

x_spot=10 * micro
Tau = 25 *femto
x_offset = 10
########################################################

Sim = Grinni.Open(SimulationPath, x_spot, Tau)

Species = ["electron", "proton", "carbon"]

Sim.DensityPlot(Species[1],E_las=True, Field='Ey', x_offset=x_offset, File='proton_density')