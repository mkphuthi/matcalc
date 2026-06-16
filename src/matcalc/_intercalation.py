"""Intercalation and voltage-profile calculations via Monte Carlo."""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from monty.serialization import dumpfn
from pymatgen.transformations.site_transformations import RemoveSitesTransformation
from tqdm import tqdm

from ._base import PropCalc
from ._relaxation import RelaxCalc
from .utils import to_ase_atoms, to_pmg_structure

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from pymatgen.core import Structure

# scipy.linalg.logm (called by ASE's FrechetCellFilter on every relaxation step) emits a
# benign RuntimeWarning whenever its error estimate is non-zero, even at ~1e-13. Silence it.
warnings.filterwarnings("ignore", message="logm result may be inaccurate")

# Boltzmann constant in eV/K.
_KB = 8.617330337217213e-05


def _boltzmann_weight(e1: float, e0: float, temperature: float) -> float:
    """
    Compute the Metropolis-Hastings acceptance weight ``exp(-(e1 - e0) / (kB * T))``.

    :param e1: Energy of the proposed configuration in eV.
    :type e1: float
    :param e0: Energy of the current configuration in eV.
    :type e0: float
    :param temperature: Temperature in Kelvin.
    :type temperature: float
    :return: The Boltzmann acceptance weight.
    :rtype: float
    """
    return float(np.exp(-(e1 - e0) / (_KB * temperature)))


def _atoms_from_results(results: dict[str, Any]) -> Atoms:
    """
    Build an ASE ``Atoms`` carrying a ``SinglePointCalculator`` from a results dict.

    :param results: A results dictionary containing ``final_structure``, ``energy``,
        ``forces`` and ``stress`` keys (as returned by :class:`RelaxCalc`).
    :type results: dict[str, Any]
    :return: An ``Atoms`` object with energy/forces/stress attached, suitable for writing
        to an ASE trajectory.
    :rtype: Atoms
    """
    atoms = to_ase_atoms(results["final_structure"])
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=results["energy"],
        forces=results["forces"],
        stress=results["stress"],
    )
    return atoms


class IntercalationCalc(PropCalc):
    """
    Calculator for intercalation and voltage-profile predictions via Monte Carlo.

    For each target concentration derived from ``concentration_range``, a number ``k`` of
    intercalating ions is removed from the (optionally supercelled) host structure and the
    resulting configuration is scored by a single-point energy or a relaxation. Configurations
    are sampled with a Metropolis-Hastings acceptance criterion, and the accepted energies are
    used downstream to predict voltage profiles. A trajectory is written per concentration level
    and the per-level results are serialized to ``results.json.gz``.

    :param calculator: An ASE calculator object used to perform energy and force
        calculations. If a string is provided, the corresponding universal calculator is loaded.
    :type calculator: Calculator | str
    """

    def __init__(
        self,
        calculator: Calculator | str,
        nsteps: int,
        temperature: float,
        concentration_range: Sequence[float],
        *,
        save_freq: int = 100,
        trajfile: str | None = None,
        supercell: Sequence[int] | None = None,
        relax: bool = True,
        species: str | None = None,
        indices: Sequence[int] | None = None,
        seed: int | None = None,
        relax_calc_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the IntercalationCalc.

        :param calculator: An ASE calculator object used to perform energy and force
            calculations. If a string is provided, the corresponding universal calculator is loaded.
        :type calculator: Calculator | str
        :param nsteps: Number of Monte Carlo steps to run at each concentration level.
        :type nsteps: int
        :param temperature: Temperature in Kelvin used in the Metropolis-Hastings acceptance test.
        :type temperature: float
        :param concentration_range: Arguments forwarded to :func:`numpy.arange` (start, stop, step)
            defining the fractional concentrations of removed ions to sample.
        :type concentration_range: Sequence[float]
        :param save_freq: Append a frame to the trajectory every ``save_freq`` Monte Carlo steps.
            Default is 100.
        :type save_freq: int, optional
        :param trajfile: Base trajectory filename. A ``_k{k}`` suffix is inserted before the
            extension for each concentration level. Defaults to a timestamped name.
        :type trajfile: str | None, optional
        :param supercell: Optional supercell scaling applied to the input structure before site
            indices are resolved. Only valid together with ``species``. Default is None.
        :type supercell: Sequence[int] | None, optional
        :param relax: Whether to relax each configuration before scoring it. If False a single
            point energy is used. Default is True.
        :type relax: bool, optional
        :param species: Symbol of the intercalating species to remove. Mutually exclusive with
            ``indices``. Default is None.
        :type species: str | None, optional
        :param indices: Explicit site indices eligible for removal. Mutually exclusive with
            ``species``. Default is None.
        :type indices: Sequence[int] | None, optional
        :param seed: Seed for the random number generator, for reproducible sampling. Default is None.
        :type seed: int | None, optional
        :param relax_calc_kwargs: Additional keyword arguments passed to :class:`RelaxCalc`.
            ``max_steps``, ``optimizer`` and ``fmax`` are honored if present. Default is None.
        :type relax_calc_kwargs: dict[str, Any] | None, optional
        """
        self.calculator = calculator  # type: ignore[assignment]
        self.nsteps = nsteps
        self.temperature = temperature
        self.concentration_range = concentration_range
        self.supercell = supercell
        self.save_freq = save_freq
        self.trajfile = trajfile or f"traj-{datetime.now().strftime('%H%Mhrs_%d-%m-%Y')}.traj"
        self.species = species
        self.indices = indices
        self.relax = relax
        self.relax_calc_kwargs = relax_calc_kwargs or {}
        self.seed = seed
        self._rng = np.random.default_rng(self.seed)

    def _remove_k(self, structure: Structure, indices: Sequence[int], k: int) -> Structure:
        """
        Return a copy of ``structure`` with ``k`` randomly chosen sites removed.

        Sites are drawn without replacement from ``indices`` so that exactly ``k`` distinct
        sites are removed.

        :param structure: The structure to deintercalate.
        :type structure: Structure
        :param indices: Candidate site indices eligible for removal.
        :type indices: Sequence[int]
        :param k: Number of sites to remove.
        :type k: int
        :return: A new structure with the selected sites removed.
        :rtype: Structure
        """
        indices_to_remove = self._rng.choice(indices, k, replace=False)
        transformation = RemoveSitesTransformation(indices_to_remove)
        return transformation.apply_transformation(structure)

    def calc(
        self,
        structure: Structure | Atoms,
    ) -> dict[str, Any]:
        """
        Run the Monte Carlo deintercalation sweep over the requested concentrations.

        :param structure: The fully occupied host structure to deintercalate.
        :type structure: Structure | Atoms
        :return: A dictionary keyed by concentration index, each value being the accepted
            configuration's results augmented with ``Acceptance Ratio`` and ``Num_removed``.
        :rtype: dict[str, Any]
        :raises ValueError: If neither or both of ``species`` and ``indices`` are provided, or if
            ``supercell`` is combined with explicit ``indices``.
        """
        structure = to_pmg_structure(structure)

        if self.indices is None and self.species is None:
            raise ValueError("Provide one of indices or species")
        if self.species is not None and self.indices is None:
            if self.supercell:
                structure = structure.make_supercell(self.supercell)
            self.indices = structure.indices_from_symbol(self.species)
        elif self.indices is not None and self.supercell is not None:
            raise ValueError("Provide supercell as input structure if specifying supercell argument")

        assert self.indices is not None  # noqa: S101  # resolved above; satisfies type checker

        relax_kwargs = dict(self.relax_calc_kwargs)
        max_steps = relax_kwargs.pop("max_steps", 200)
        optimizer = relax_kwargs.pop("optimizer", "FIRE")
        fmax = relax_kwargs.pop("fmax", 0.02)
        if not self.relax:
            max_steps = 0

        relaxer = RelaxCalc(
            self.calculator,
            optimizer=optimizer,
            fmax=fmax,
            max_steps=max_steps,
            **relax_kwargs,
        )

        n_indices = len(self.indices)
        concentrations = np.arange(*self.concentration_range)
        ks = np.unique(np.round(concentrations * n_indices)).astype(int)

        results = {}
        k_iter = tqdm(ks, desc="concentration levels")
        for ik, k in enumerate(k_iter):
            concentration = k / n_indices
            k_iter.set_postfix(k=int(k))
            traj_path = Path(self.trajfile)
            k_trajfile = str(traj_path.with_name(f"{traj_path.stem}_k{int(k)}{traj_path.suffix}"))

            prev_results = relaxer.calc(self._remove_k(structure, self.indices, k))
            trajectory = [_atoms_from_results(prev_results)]
            n_accepted = 0
            for i in tqdm(range(self.nsteps), desc=f"k={int(k)}, c={concentration:.3f}", leave=False):
                relax_results = relaxer.calc(self._remove_k(structure, self.indices, k))
                cur_e = relax_results["energy"]
                prev_e = prev_results["energy"]
                weight = _boltzmann_weight(cur_e, prev_e, self.temperature)
                if (cur_e - prev_e) < 0 or self._rng.random() < weight:
                    prev_results = relax_results
                    n_accepted += 1

                if i % self.save_freq == 0:
                    trajectory.append(_atoms_from_results(relax_results))
                    write(k_trajfile, trajectory)

            results[f"{ik}"] = prev_results | {
                "Acceptance Ratio": n_accepted / self.nsteps,
                "Num_removed": int(k),
                "concentration": concentration,
            }

        dumpfn(results, "results.json.gz")
        return results
