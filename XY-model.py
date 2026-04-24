# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 12:39:56 2026

@author: user
"""

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SimulationResult:
    temperature: float
    magnetization_per_spin: np.ndarray
    energy_per_spin: np.ndarray
    final_angles: np.ndarray
    accepted_moves_per_sweep: np.ndarray
    initial_condition: str


class XYModel2D:
    """
    2D XY model on an N x N square lattice with periodic boundary conditions.

    Spins are represented by angles theta in [-pi, pi), with spin vectors
    s = (cos(theta), sin(theta)).

    Hamiltonian (with J = 1, h = 0):
        H = -sum_<i,j> s_i . s_j

    The Metropolis update proposes a random angle increment dtheta in [-delta, delta]
    for a single randomly chosen spin.
    """

    def __init__(
        self,
        N: int,
        temperature: float,
        delta: float = 0.7,
        seed: Optional[int] = None,
        initial_condition: str = "random",
    ) -> None:
        if temperature <= 0:
            raise ValueError("Temperature must be > 0.")
        if initial_condition not in {"random", "aligned"}:
            raise ValueError("initial_condition must be 'random' or 'aligned'.")

        self.N = N
        self.T = temperature
        self.beta = 1.0 / temperature
        self.delta = delta
        self.rng = np.random.default_rng(seed)
        self.initial_condition = initial_condition

        if initial_condition == "random":
            self.angles = self.rng.uniform(-math.pi, math.pi, size=(N, N))
        else:
            aligned_angle = 0.0
            self.angles = np.full((N, N), aligned_angle, dtype=float)

        self.energy = self.total_energy()
        self.mx, self.my = self.total_magnetization_components()

    def total_energy(self) -> float:
        """Compute the full energy, counting each bond once."""
        theta = self.angles
        right = np.roll(theta, shift=-1, axis=1)
        down = np.roll(theta, shift=-1, axis=0)
        return -np.sum(np.cos(theta - right) + np.cos(theta - down))

    def total_magnetization_components(self) -> Tuple[float, float]:
        mx = np.sum(np.cos(self.angles))
        my = np.sum(np.sin(self.angles))
        return float(mx), float(my)

    def magnetization_magnitude_per_spin(self) -> float:
        return math.hypot(self.mx, self.my) / (self.N * self.N)

    def local_energy_contribution(self, i: int, j: int, angle: float) -> float:
        """
        Interaction energy of site (i,j) with its four neighbors.
        This local sum is suitable for computing energy differences.
        """
        N = self.N
        neighbor_angles = (
            self.angles[(i - 1) % N, j],
            self.angles[(i + 1) % N, j],
            self.angles[i, (j - 1) % N],
            self.angles[i, (j + 1) % N],
        )
        return -sum(math.cos(angle - nbr) for nbr in neighbor_angles)

    def metropolis_step(self) -> bool:
        """Attempt one single-spin update. Returns True if accepted."""
        i = self.rng.integers(0, self.N)
        j = self.rng.integers(0, self.N)

        old_angle = self.angles[i, j]
        proposal = old_angle + self.rng.uniform(-self.delta, self.delta)
        proposal = (proposal + math.pi) % (2 * math.pi) - math.pi

        old_local = self.local_energy_contribution(i, j, old_angle)
        new_local = self.local_energy_contribution(i, j, proposal)
        dE = new_local - old_local

        if dE <= 0 or self.rng.random() < math.exp(-self.beta * dE):
            old_cos = math.cos(old_angle)
            old_sin = math.sin(old_angle)
            new_cos = math.cos(proposal)
            new_sin = math.sin(proposal)

            self.angles[i, j] = proposal
            self.energy += dE
            self.mx += new_cos - old_cos
            self.my += new_sin - old_sin
            return True
        return False

    def sweep(self) -> Tuple[float, float, float]:
        """
        Perform N^2 attempted updates (one Monte Carlo sweep).
        Returns: (magnetization per spin, energy per spin, acceptance fraction)
        """
        accepted = 0
        n_attempts = self.N * self.N
        for _ in range(n_attempts):
            accepted += int(self.metropolis_step())

        m = self.magnetization_magnitude_per_spin()
        e = self.energy / n_attempts
        acceptance = accepted / n_attempts
        return m, e, acceptance

    def run(self, n_sweeps: int) -> SimulationResult:
        magnetization = np.empty(n_sweeps, dtype=float)
        energy = np.empty(n_sweeps, dtype=float)
        acceptance = np.empty(n_sweeps, dtype=float)

        for sweep_idx in range(n_sweeps):
            m, e, a = self.sweep()
            magnetization[sweep_idx] = m
            energy[sweep_idx] = e
            acceptance[sweep_idx] = a

        return SimulationResult(
            temperature=self.T,
            magnetization_per_spin=magnetization,
            energy_per_spin=energy,
            final_angles=self.angles.copy(),
            accepted_moves_per_sweep=acceptance,
            initial_condition=self.initial_condition,
        )


def plot_spin_configuration(angles: np.ndarray, title: str, output_path: Optional[str] = None) -> None:
    N = angles.shape[0]
    x, y = np.meshgrid(np.arange(N), np.arange(N))
    u = np.cos(angles)
    v = np.sin(angles)

    plt.figure(figsize=(6, 6))
    plt.quiver(x, y, u, v, angles, pivot="mid")
    plt.title(title)
    plt.xlim(-0.5, N - 0.5)
    plt.ylim(-0.5, N - 0.5)
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.show()


def plot_two_magnetization_traces(
    result_a: SimulationResult,
    result_b: SimulationResult,
    output_path: Optional[str] = None,
) -> None:
    sweeps = np.arange(1, len(result_a.magnetization_per_spin) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(sweeps, result_a.magnetization_per_spin, label=f"{result_a.initial_condition} start")
    plt.plot(sweeps, result_b.magnetization_per_spin, label=f"{result_b.initial_condition} start")
    plt.xlabel("Monte Carlo sweeps")
    plt.ylabel("Magnetization per spin")
    plt.title(f"XY model magnetization traces at T = {result_a.temperature:.2f}")
    plt.legend()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.show()


def estimate_equilibration_sweep(
    trace_a: np.ndarray,
    trace_b: np.ndarray,
    tolerance: float = 0.03,
    window: int = 20,
) -> Optional[int]:
    """
    Rough heuristic: first sweep where moving averages differ by less than tolerance.
    Useful as a first milestone estimate, not as a precise production metric.
    """
    if len(trace_a) != len(trace_b):
        raise ValueError("Traces must have the same length.")
    if window <= 0 or window > len(trace_a):
        raise ValueError("Invalid averaging window.")

    kernel = np.ones(window) / window
    avg_a = np.convolve(trace_a, kernel, mode="valid")
    avg_b = np.convolve(trace_b, kernel, mode="valid")
    diff = np.abs(avg_a - avg_b)

    for idx, value in enumerate(diff):
        if value < tolerance:
            return idx + window
    return None


def run_pair_at_temperature(
    N: int,
    temperature: float,
    n_sweeps: int,
    delta: float,
    seed: int,
) -> Tuple[SimulationResult, SimulationResult, Optional[int]]:
    model_random = XYModel2D(
        N=N,
        temperature=temperature,
        delta=delta,
        seed=seed,
        initial_condition="random",
    )
    model_aligned = XYModel2D(
        N=N,
        temperature=temperature,
        delta=delta,
        seed=seed + 1,
        initial_condition="aligned",
    )

    result_random = model_random.run(n_sweeps)
    result_aligned = model_aligned.run(n_sweeps)
    equilibration = estimate_equilibration_sweep(
        result_random.magnetization_per_spin,
        result_aligned.magnetization_per_spin,
    )
    return result_random, result_aligned, equilibration


def temperature_sweep(
    N: int = 10,
    temperatures: Optional[np.ndarray] = None,
    n_sweeps: int = 300,
    delta: float = 0.7,
    seed: int = 1234,
    output_dir: str = "xy_outputs",
    save_figures: bool = True,
) -> List[dict]:
    """
    Run the first milestone sweep over temperatures.

    Recommended by the project: T from 0.5 to 2.5 in steps of 0.2.
    """
    if temperatures is None:
        temperatures = np.round(np.arange(0.5, 2.5 + 1e-9, 0.2), 2)

    os.makedirs(output_dir, exist_ok=True)
    summary = []

    for idx, T in enumerate(temperatures):
        result_random, result_aligned, equilibration = run_pair_at_temperature(
            N=N,
            temperature=float(T),
            n_sweeps=n_sweeps,
            delta=delta,
            seed=seed + 10 * idx,
        )

        fig_path = os.path.join(output_dir, f"magnetization_T_{T:.2f}.png") if save_figures else None
        plot_two_magnetization_traces(result_random, result_aligned, output_path=fig_path)

        snap_path_random = os.path.join(output_dir, f"snapshot_random_T_{T:.2f}.png") if save_figures else None
        snap_path_aligned = os.path.join(output_dir, f"snapshot_aligned_T_{T:.2f}.png") if save_figures else None
        plot_spin_configuration(
            result_random.final_angles,
            title=f"Final spin configuration (random start), T = {T:.2f}",
            output_path=snap_path_random,
        )
        plot_spin_configuration(
            result_aligned.final_angles,
            title=f"Final spin configuration (aligned start), T = {T:.2f}",
            output_path=snap_path_aligned,
        )

        summary.append(
            {
                "T": float(T),
                "m_final_random": float(result_random.magnetization_per_spin[-1]),
                "m_final_aligned": float(result_aligned.magnetization_per_spin[-1]),
                "e_final_random": float(result_random.energy_per_spin[-1]),
                "e_final_aligned": float(result_aligned.energy_per_spin[-1]),
                "equilibration_sweep_estimate": equilibration,
                "acceptance_random": float(np.mean(result_random.accepted_moves_per_sweep)),
                "acceptance_aligned": float(np.mean(result_aligned.accepted_moves_per_sweep)),
            }
        )

    return summary


def quick_demo() -> None:
    """
    Good first test:
      1. Start with N = 10.
      2. Check a few temperatures.
      3. Then move to N = 20 and N = 50.
    """
    summary = temperature_sweep(
        N=10,
        temperatures=np.array([0.5, 1.1, 1.9, 2.5]),
        n_sweeps=250,
        delta=0.7,
        seed=42,
        output_dir="xy_demo_outputs",
        save_figures=True,
    )

    print("Temperature sweep summary:")
    for row in summary:
        print(row)


if __name__ == "__main__":
    quick_demo()
