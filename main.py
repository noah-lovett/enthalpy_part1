import torch
from mace.calculators import mace_mp
from ase import build
from ase.optimize import LBFGS #was bfgs
from ase.io import read
from ase.visualize import view

device = "cuda" if torch.cuda.is_available() else "cpu"
calculator = mace_mp(
model='mh-1',
device=device,
head='omol',
dispersion=False,
default_dtype="float32"
)

ethylene = build.molecule("C2H4")
ethylene.set_pbc(False)
ethylene.calc = calculator
opt = LBFGS(ethylene, trajectory='C2H4.traj')
opt.run(fmax=0.01)
print("U (C2H4): " + str(ethylene.get_potential_energy()))
ethylene_optimization = read('C2H4.traj', ':')

hydrogen = build.molecule("H2")
hydrogen.set_pbc(False)
hydrogen.calc = calculator
opt = LBFGS(hydrogen, trajectory='H2.traj')
opt.run(fmax=0.01)
print("U (H2): " + str(hydrogen.get_potential_energy()))
hydrogen_optimization = read('H2.traj', ':')

ethane = build.molecule("C2H6")
ethane.set_pbc(False)
ethane.calc = calculator
opt = LBFGS(ethane, trajectory='C2H6.traj')
opt.run(fmax=0.01)
print("U (C2H6): " + str(ethane.get_potential_energy()))
ethane_optimization = read('C2H6.traj', ':')

enthalpy_T0 = ethane.get_potential_energy() - ethylene.get_potential_energy() - hydrogen.get_potential_energy()
print("Enthalpy at 0K: " + str(enthalpy_T0) + " eV")
enthalpy_T0_kcal = 23.0609*enthalpy_T0
print("Enthalpy at 0K: " + str(enthalpy_T0_kcal) + " kcal/mol")