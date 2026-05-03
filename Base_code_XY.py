# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 11:59:21 2026

@author: David
"""

#CompPhysB 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#%% Simulation parameters
n_thermal = 0            # Number of steps to equilibrate, is thrown away
n_steps = 1000           # Number of steps done after equilibration
proposal_width = np.pi/2 # Theta is updated with steps of [-proposal_width, proposal_width]
T = 0.7                  # Basis temperature for simulations
lattice_size = 20        # Simulations without specified N are run with this number
#%% Class definition 

class XYModel2D:
    def __init__(self, N, T, J=1.0, seed=None):
        """
        2D XY model on an NxN lattice with periodic boundary conditions.

        Spins are angles theta in [-pi, pi).
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


    def sweep(self, proposal_width=proposal_width):
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

    def simulate(self, n_thermal=n_thermal, n_steps=n_steps, proposal_width=proposal_width):
        """
        Thermalize, then record magnetization over time.
        """
        # Thermalization
        for _ in range(n_thermal):
            self.sweep(proposal_width=proposal_width)

        magnetizations = []
        for _ in range(n_steps):
            self.sweep(proposal_width=proposal_width)
            magnetizations.append(self.magnetization())
            
        return np.array(magnetizations)
        
    def set_initial_condition(self, initial_condition):
        """
        Set the initial spin configuration.

        Parameters
        ----------
        initial_condition : str
            Either "random" or "aligned".
        """
        if initial_condition == "random":
            self.theta = self.rng.uniform(-np.pi, np.pi, size=(self.N, self.N))

        elif initial_condition == "aligned":
            self.theta = np.zeros((self.N, self.N))


#%% Seperate functions for milestones and plotting
def run_sizes(sizes, T=T, n_thermal=n_thermal, n_steps=n_steps, seed=42):
    results = {}

    for N in sizes:
        print(f"Running N = {N}")
        model = XYModel2D(N=N, T=T, seed=seed)
        magnetizations = model.simulate(n_thermal=n_thermal, n_steps=n_steps)
        results[N] = magnetizations
        print(f"  <|M|> = {np.mean(magnetizations):.4f}")

    return results

def run_temperatures(temperatures,lattice_size=lattice_size ,n_steps=n_steps,proposal_width=proposal_width,
   seed=42):
    """
    Run one simulation for each temperature.

    Returns
    -------
    dict
        Dictionary with temperatures as keys and magnetization arrays as values.
    """
    results = {}

    for temperature in temperatures:
        print(f"Running T = {temperature:.1f}")

        model = XYModel2D(
            N=lattice_size,
            T=temperature,
            J=1.0,
            seed=seed,
        )

        magnetizations = model.simulate(n_thermal=n_thermal, n_steps=n_steps)

        results[temperature] = magnetizations

    return results

        
def compare_initial_conditions(lattice_size=lattice_size,temperature=T,n_steps=n_steps,proposal_width=proposal_width,seed=42,):
    """
    Run two simulations: one random start and one aligned start.
    """
    random_model = XYModel2D(N=lattice_size, T=temperature, seed=seed)
    random_model.set_initial_condition("random")

    aligned_model = XYModel2D(N=lattice_size, T=temperature, seed=seed)
    aligned_model.set_initial_condition("aligned")

    random_magnetizations = random_model.simulate()

    aligned_magnetizations = aligned_model.simulate()

    return random_magnetizations, aligned_magnetizations


def plot_magnetization_time_series(results):
    plt.figure(figsize=(10, 6))
    for N, magnetizations in results.items():
        plt.plot(magnetizations, label=f"N = {N}", alpha=0.9)

    plt.xlabel("Monte Carlo sweep")
    plt.ylabel("Magnetization per spin |M|")
    plt.title("2D XY model: magnetization vs Monte Carlo sweep")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def plot_temperature_series(results):
    """
    Plot magnetization as a function of Monte Carlo sweep for each temperature.
    """
    plt.figure(figsize=(11, 7))

    for temperature, magnetizations in results.items():
        plt.plot(
            magnetizations,
            label=f"T = {temperature:.1f}",
            alpha=0.8,
        )

    plt.xlabel("Monte Carlo sweep")
    plt.ylabel("Magnetization per spin |m|")
    plt.title("Equilibration of the 2D XY model at different temperatures")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def plot_initial_condition_comparison(
    random_magnetizations,
    aligned_magnetizations,
    temperature,
):
    """
    Plot magnetization curves for random and aligned initial conditions.
    """
    plt.figure(figsize=(10, 6))

    plt.plot(
        random_magnetizations,
        label="Random initial condition",
        alpha=0.9,
    )

    plt.plot(
        aligned_magnetizations,
        label="Aligned initial condition",
        alpha=0.9,
    )

    plt.xlabel("Monte Carlo sweep")
    plt.ylabel("Magnetization per spin |m|")
    plt.title(f"Equilibration from different initial conditions, T = {temperature}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_average_magnetization(results): #Not neccesary
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
    


def animate_spin_configuration(
    lattice_size=lattice_size,
    temperature=T,
    n_frames=n_steps,
    sweeps_per_frame=1,
    proposal_width=proposal_width,
    initial_condition="random",
    seed=42,
):
    """
    Animate the 2D XY model spin configuration.

    The angle theta is shown as a color from -pi to pi.
    Each animation frame advances the simulation by sweeps_per_frame sweeps.
    """
    model = XYModel2D(N=lattice_size, T=temperature, seed=seed)
    model.set_initial_condition(initial_condition)

    figure, axis = plt.subplots(figsize=(6, 6))

    image = axis.imshow(
        model.theta,
        vmin=-np.pi,
        vmax=np.pi,
        cmap="twilight",
        origin="lower",
    )

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Spin angle θ")

    axis.set_title(f"T = {temperature}, sweep = 0")
    axis.set_xlabel("x")
    axis.set_ylabel("y")

    def update(frame_index):
        """
        Advance the simulation and update the image.
        """
        for _ in range(sweeps_per_frame):
            model.sweep(proposal_width=proposal_width)

        image.set_data(model.theta)

        current_sweep = (frame_index + 1) * sweeps_per_frame
        magnetization = model.magnetization()

        axis.set_title(
            f"T = {temperature}, sweep = {current_sweep}, |m| = {magnetization:.3f}"
        )

        return [image]

    animation = FuncAnimation(
        figure,
        update,
        frames=n_frames,
        interval=80,
        blit=False,
    )

    plt.show()

    return animation    


def animate_spin_arrows(
    lattice_size=lattice_size,
    temperature=T,
    n_frames=n_steps,
    sweeps_per_frame=1,
    proposal_width=proposal_width,
    initial_condition="random",
    seed=42,
):
    """
    Animate the 2D XY model using arrows for spin directions.
    """
    model = XYModel2D(N=lattice_size, T=temperature, seed=seed)
    model.set_initial_condition(initial_condition)

    x_positions, y_positions = np.meshgrid(
        np.arange(lattice_size),
        np.arange(lattice_size),
    )

    u_components = np.cos(model.theta)
    v_components = np.sin(model.theta)

    figure, axis = plt.subplots(figsize=(7, 7))

    arrows = axis.quiver(
        x_positions,
        y_positions,
        u_components,
        v_components,
        pivot="middle",
        scale=25,
    )

    axis.set_aspect("equal")
    axis.set_xlim(-1, lattice_size)
    axis.set_ylim(-1, lattice_size)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(f"T = {temperature}, sweep = 0")

    def update(frame_index):
        """
        Advance the simulation and update the arrows.
        """
        for _ in range(sweeps_per_frame):
            model.sweep(proposal_width=proposal_width)

        u_components = np.cos(model.theta)
        v_components = np.sin(model.theta)

        arrows.set_UVC(u_components, v_components)

        current_sweep = (frame_index + 1) * sweeps_per_frame
        magnetization = model.magnetization()

        axis.set_title(
            f"T = {temperature}, sweep = {current_sweep}, |m| = {magnetization:.3f}"
        )

        return arrows,

    animation = FuncAnimation(
        figure,
        update,
        frames=n_frames,
        interval=80,
        blit=False,
    )

    plt.show()

    return animation
#%% Milestone 1.1: Vary system size

sizes = [10, 20, 50]
    
results = run_sizes(sizes)
    
plot_magnetization_time_series(results)
#plot_average_magnetization(results)

#%% Milestone 1.2: Vary temperature
temperatures = np.arange(0.5, 2.51, 0.2)

results = run_temperatures(temperatures)
plot_temperature_series(results)

#%% Milestone 1.3: random and aligned initial configurations
random_magnetizations, aligned_magnetizations = compare_initial_conditions()

plot_initial_condition_comparison(random_magnetizations,aligned_magnetizations,temperature=T)

#%% Milestone 1.4: animate spin configuration as colored blocks
%matplotlib qt

animation = animate_spin_configuration()

#%% animate spin configuration as arrows
%matplotlib qt

animation = animate_spin_arrows()
