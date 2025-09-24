# Grinni
Python package to automate plotting of SMILEI data.

Some assumptions are made:
- Laser wavelength is defined as: `lambda_las = ... * <nano or micro>`
- Distance from x_min boundary to target front is defined as: `x_vac = ... * <nano or micro>`
- Pulse duration in intensity space of the laser is defined as: `Tau_I = ... * <femto or pico>`
- Electric fields are names: `instant fields` and `average fields`
- Particle Binning diagnostics are names: `<species> density`, `<species> spectra`, <species> <phase axis i.e. x-px, px-py> phase space`, `<species> angle`.

## Get Grinni
Within the directory you want Grinni to live, e.g `~/software`, run:
```
git clone https://github.com/Nkehoe-QUB/Grinni
```
This will create a Grinni directory.

## Create the conda environemnt and install dependencies
```
cd Grinni
conda env create -f requiremnts.yml
```

## Install Grinni into your environment
Within the Grinni directory (`~/software/Grinni`) run:
```
pip install .
```
