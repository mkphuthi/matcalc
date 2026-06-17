"""Calculator for Monte Carlo sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from tqdm import tqdm

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from ._base import PropCalc
from ._relaxation import RelaxCalc
from .utils import to_ase_atoms, to_pmg_structure

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from pymatgen.core import Structure

KB = 8.617330337217213e-05  # eV/K


def _atoms_from_results(results: dict[str, Any]) -> Atoms:
    atoms = to_ase_atoms(results["final_structure"])
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=results["energy"],
        forces=results["forces"],
        stress=results["stress"],
    )
    return atoms


class MCCalc(PropCalc):
    def __init__(
        self,
        calculator: Calculator | str,
        transformation: Any,
        transform_initial: bool,
        ensemble: Literal["canonical","grand_canonical"] = "canonical",
        *,
        nsteps: int = 100,
        temperature: float = 300,
        save_freq: int = 10,
        trajfile: str = "mc.traj",
        relax: bool = True,
        seed: int | None = None,
        relax_calc_kwargs: dict | None = None,
    ) -> None:
        self.calculator = calculator
        self.transformation = transformation
        self.nsteps = nsteps
        self.temperature = temperature
        self.ensemble = ensemble
        self.save_freq = save_freq
        self.trajfile = trajfile
        self.relax = relax
        self.relax_calc_kwargs = relax_calc_kwargs or {}
        self._rng = np.random.default_rng(seed)
        self.transform_initial = transform_initial
        self.trajectory: list[Atoms] = []
        self.acceptance_ratio = 0.0
        self._n_accepted = 0
        self._structure: Structure | None = None
        self._initial_structure: Structure | None = None
        self._results: dict[str, Any] | None = None

    def _score(self, structure: Structure) -> dict[str, Any]:
        kwargs = dict(self.relax_calc_kwargs)
        relaxer = RelaxCalc(
            self.calculator,
            optimizer=kwargs.pop("optimizer", "FIRE"),
            fmax=kwargs.pop("fmax", 0.02),
            max_steps=kwargs.pop("max_steps", 200) if self.relax else 0,
            **kwargs,
        )
        return relaxer.calc(structure)

    def step_canonical(self) -> bool:
        if self.transform_initial:
            proposal = self.transformation.apply_transformation(self._initial_structure)
        else:
            proposal = self.transformation.apply_transformation(self._structure)
        results = self._score(proposal)
        delta = results["energy"] - self._results["energy"]
        accept = delta < 0 or self._rng.random() < np.exp(-delta / (KB * self.temperature))
        if accept:
            self._structure = results["final_structure"]
            self._results = results
            self._n_accepted += 1
        return accept

    def step_grand_canonical(self) -> bool:
        raise NotImplementedError

    def calc(self, structure: Structure | Atoms) -> dict[str, Any]:
        self._initial_structure = to_pmg_structure(structure)
        if self.transform_initial:
            self._structure = self.transformation.apply_transformation(self._initial_structure)
        else:
            self._structure = self._initial_structure
        self._results = self._score(self._structure)
        self.trajectory = [_atoms_from_results(self._results)]
        if self.ensemble == "canonical":
            step = self.step_canonical
        elif self.ensemble == "grand_canonical":
            step = self.step_grand_canonical
        else:
            raise ValueError("Only canonical Ensemble accepted")
        for i in tqdm(range(self.nsteps), desc="Step"):
            step()
            if i % self.save_freq == 0:
                self.trajectory.append(_atoms_from_results(self._results))
                write(self.trajfile, self.trajectory)
        self.acceptance_ratio = self._n_accepted / self.nsteps
        return self._results | {"acceptance_ratio": self.acceptance_ratio}
