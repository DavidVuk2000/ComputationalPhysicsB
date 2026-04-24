# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 11:59:21 2026

@author: David
"""

#CompPhysB 
import numpy as np
import matplotlib.pyplot as plt


class XYModel2D:
    def __init__(self, N, T, J=1.0, seed=None):
        """
        2D XY model on an NxN lattice with periodic boundary conditions.

        Spins are angles theta in [0, 2pi).
        Hamiltonian:
            H = -J * sum_<ij> cos(theta_i - theta_j)
        """
        self.N = N
        self.T = T
        self.beta = 1.0 / T
        self.J = J
        self.rng = np.random.default_rng(seed)

        # Random initial configuration
        self.theta = self.rng.uniform(-np.pi, np.pi, size=(N, N))

    def delta_energy(self, i, j, new_theta):
        """
        Energy change if spin at (i,j) changes from old theta to new_theta.
        Only nearest neighbors matter.
        """
        N = self.N
        old_theta = self.theta[i, j]

        # Periodic boundary conditions
        neighbors = [
            self.theta[(i + 1) % N, j],
            self.theta[(i - 1) % N, j],
            self.theta[i, (j + 1) % N],
            self.theta[i, (j - 1) % N],
        ]

        old_E = -self.J * sum(np.cos(old_theta - nn) for nn in neighbors)
        new_E = -self.J * sum(np.cos(new_theta - nn) for nn in neighbors)

        return new_E - old_E


    def sweep(self, proposal_width=np.pi / 2):
        """
        One Monte Carlo sweep = N^2 attempted updates.
        """
        for _ in range(self.N * self.N):
            i = self.rng.integers(0, self.N)
            j = self.rng.integers(0, self.N)

            old_theta = self.theta[i, j]
            new_theta = ((old_theta + self.rng.uniform(-proposal_width, proposal_width) + np.pi) % (2 * np.pi)) - np.pi

            dE = self.delta_energy(i, j, new_theta)

            # Metropolis acceptance
            if dE <= 0 or self.rng.random() < np.exp(-self.beta * dE):
                self.theta[i, j] = new_theta

    def magnetization_vector(self):
        """
        Total magnetization vector per spin.
        """
        mx = np.mean(np.cos(self.theta))
        my = np.mean(np.sin(self.theta))
        return mx, my

    def magnetization(self):
        """
        Magnitude of magnetization per spin.
        """
        mx, my = self.magnetization_vector()
        return np.sqrt(mx**2 + my**2)

    def simulate(self, n_thermal=500, n_steps=2000, proposal_width=np.pi / 2):
        """
        Thermalize, then record magnetization over time.
        """
        # Thermalization
        for _ in range(n_thermal):
            self.sweep(proposal_width=proposal_width)

        mags = []
        for _ in range(n_steps):
            self.sweep(proposal_width=proposal_width)
            mags.append(self.magnetization())

        return np.array(mags)


def run_sizes(sizes, T=0.7, n_thermal=500, n_steps=2000, seed=42):
    results = {}

    for N in sizes:
        print(f"Running N = {N}")
        model = XYModel2D(N=N, T=T, seed=seed)
        mags = model.simulate(n_thermal=n_thermal, n_steps=n_steps)
        results[N] = mags
        print(f"  <|M|> = {np.mean(mags):.4f}")

    return results


def plot_magnetization_time_series(results):
    plt.figure(figsize=(10, 6))
    for N, mags in results.items():
        plt.plot(mags, label=f"N = {N}", alpha=0.9)

    plt.xlabel("Monte Carlo sweep")
    plt.ylabel("Magnetization per spin |M|")
    plt.title("2D XY model: magnetization vs Monte Carlo sweep")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_average_magnetization(results):
    sizes = sorted(results.keys())
    avg_mags = [np.mean(results[N]) for N in sizes]
    std_mags = [np.std(results[N]) for N in sizes]

    plt.figure(figsize=(7, 5))
    plt.errorbar(sizes, avg_mags, yerr=std_mags, fmt='o-', capsize=5)
    plt.xlabel("System size N")
    plt.ylabel("Average magnetization per spin <|M|>")
    plt.title("Average magnetization vs system size")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sizes = [10, 20, 50]

    # Try changing T to explore the model
    T = 0.7

    results = run_sizes(
        sizes=sizes,
        T=T,
        n_thermal=500,
        n_steps=2000,
        seed=42
    )

    plot_magnetization_time_series(results)
    plot_average_magnetization(results)