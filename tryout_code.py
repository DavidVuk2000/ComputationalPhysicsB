# -*- coding: utf-8 -*-
"""
Created on Mon May 11 21:49:08 2026

@author: user
"""

#CompPhysB 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
import os

#%% Simulation parameters
n_thermal = 0        # Number of steps to equilibrate
n_steps =  1000         # Number of steps done after equilibration
proposal_width = np.pi/2 # Theta is updated with steps of [-proposal_width, proposal_width]
T = 0.7                  # Basis temperature for simulations
temperatures = np.arange(0.5, 2.51, 0.2)
n_thermals = [200,400,2000,1000,400,200,200,200,200,200,200]
lattice_size = 50        # Simulations without specified N are run with this number
seed = 0                #
vortex_interval = 20     # Count number of vortices every ... steps
critical_temperature = 0.881
n_blocks = 5
pilot_sweeps = 1000
block_factor = 16
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
            
    def total_energy(self):
        """
        Calculate total energy of the lattice.
    
        Each nearest-neighbor interaction is counted once.
        """
        energy = 0.0
    
        for row in range(self.N):
            for col in range(self.N):
                theta = self.theta[row, col]
    
                right = self.theta[row, (col + 1) % self.N]
                up = self.theta[(row + 1) % self.N, col]
    
                energy -= self.J * np.cos(theta - right)
                energy -= self.J * np.cos(theta - up)
    
        return energy
    
    
#%%Functions for vortices and correlation time

def wrapped_angle_difference(angle_difference):
    """
    Wrap angle difference to [-pi, pi).
    """
    return (angle_difference + np.pi) % (2 * np.pi) - np.pi


def vortex_charges(theta):
    """
    Compute vortex charge for each plaquette.
    """
    lattice_size = theta.shape[0]
    charges = np.zeros((lattice_size, lattice_size), dtype=int)

    for row in range(lattice_size):
        for col in range(lattice_size):
            lower_left = theta[row, col]
            lower_right = theta[row, (col + 1) % lattice_size]
            upper_right = theta[(row + 1) % lattice_size, (col + 1) % lattice_size]
            upper_left = theta[(row + 1) % lattice_size, col]

            winding = (
                wrapped_angle_difference(lower_right - lower_left)
                + wrapped_angle_difference(upper_right - lower_right)
                + wrapped_angle_difference(upper_left - upper_right)
                + wrapped_angle_difference(lower_left - upper_left)
            )

            charges[row, col] = int(np.rint(winding / (2 * np.pi)))

    return charges


def count_vortices(theta):
    """
    Count vortices and anti-vortices.
    """
    charges = vortex_charges(theta)

    n_vortices = np.sum(charges == 1)
    n_antivortices = np.sum(charges == -1)

    return n_vortices, n_antivortices

def autocorrelation(values):
    """
    Compute normalized autocorrelation function.
    """
    values = np.asarray(values)
    n_values = len(values)

    correlations = np.empty(n_values)

    for lag in range(n_values):
        first_values = values[: n_values - lag]
        shifted_values = values[lag:]

        correlations[lag] = (
            np.mean(first_values * shifted_values)
            - np.mean(first_values) * np.mean(shifted_values)
        )

    return correlations / correlations[0]


def correlation_time(values):
    """
    Estimate tau by summing normalized autocorrelation until it reaches zero    .
    """
    normalized_correlation = autocorrelation(values)

    positive_values = []

    for value in normalized_correlation:
        if value < 0:
            break

        positive_values.append(value)

    tau = np.sum(positive_values)

    return tau, normalized_correlation

def independent_mean_and_error(values, tau):
    """
    Compute mean and standard error using samples spaced by approximately 2*tau.
    """
    independent_spacing = max(1, int(round(2 * tau)))

    independent_values = values[::independent_spacing]

    mean_value = np.mean(independent_values)

    standard_error = (
        np.std(independent_values, ddof=1)
        / np.sqrt(len(independent_values))
    )

    return mean_value, standard_error, independent_values

def magnetic_susceptibility(magnetizations, temperature, number_of_spins):
    """
    Compute magnetic susceptibility per spin from magnetization fluctuations.
    """
    beta = 1.0 / temperature

    return beta * number_of_spins * (
        np.mean(magnetizations**2) - np.mean(magnetizations)**2
    )

def specific_heat(energies, temperature, number_of_spins):
    """
    Compute specific heat per spin from energy fluctuations.
    """
    beta = 1.0 / temperature

    return beta**2 / number_of_spins * (
        np.mean(energies**2) - np.mean(energies)**2
    )

def block_errors(magnetizations, energies, temperature, number_of_spins, tau):
    """
    Estimate susceptibility and specific heat with blocking.

    Each block should be much longer than tau, 16 * tau is used.
    """
    block_size = round(16 * tau)

    n_blocks = len(magnetizations) // block_size

    chi_blocks = []
    heat_blocks = []

    for block_index in range(n_blocks):
        start = block_index * block_size
        end = start + block_size

        block_magnetizations = magnetizations[start:end]
        block_energies = energies[start:end]

        chi_blocks.append(
            magnetic_susceptibility(
                block_magnetizations,
                temperature,
                number_of_spins,
            )
        )

        heat_blocks.append(
            specific_heat(
                block_energies,
                temperature,
                number_of_spins,
            )
        )

    chi_blocks = np.array(chi_blocks)
    heat_blocks = np.array(heat_blocks)

    mean_chi = np.mean(chi_blocks)
    error_chi = np.std(chi_blocks, ddof=1) / np.sqrt(n_blocks)

    mean_heat = np.mean(heat_blocks)
    error_heat = np.std(heat_blocks, ddof=1) / np.sqrt(n_blocks)

    return mean_chi, error_chi, mean_heat, error_heat

def required_measurement_steps(tau, n_blocks=20, block_factor=16):
    """
    Determine the number of measurement sweeps needed for blocking.
    """
    block_size = max(1, int(round(block_factor * tau)))

    return n_blocks * block_size


#%%Running functions
def run_sizes(
    sizes, 
    T=T, 
    n_thermal=n_thermal, 
    n_steps=n_steps, 
    seed=seed
):
    """
    Run one simulation for each temperature.
    """
    results = {}

    for N in sizes:
        print(f"Running N = {N}")
        model = XYModel2D(N=N, T=T, seed=seed)
        magnetizations = model.simulate(n_thermal=n_thermal, n_steps=n_steps)
        results[N] = magnetizations
        print(f"  <|M|> = {np.mean(magnetizations):.4f}")

    return results

def run_temperatures(
    temperatures,
    lattice_size = lattice_size,
    n_steps = n_steps,
    proposal_width = proposal_width,
    seed = seed
):
    """
    Runs one simulation for each temperature.
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

        magnetizations = model.simulate()

        results[temperature] = magnetizations

    return results

        
def compare_initial_conditions(
    lattice_size = lattice_size,
    temperature = T,
    n_steps = n_steps,
    proposal_width = proposal_width,
    seed = seed
):
    """
    Runs two simulations: one random start and one aligned start.
    """
    
    random_model = XYModel2D(N=lattice_size, T=temperature, seed=seed)
    
    aligned_model = XYModel2D(N=lattice_size, T=temperature, seed=seed)
    aligned_model.set_initial_condition("aligned")
    
    random_magnetizations = random_model.simulate()
    
    aligned_magnetizations = aligned_model.simulate()

    return random_magnetizations, aligned_magnetizations

def run_correlation_time_analysis(
    temperatures,
    lattice_size = lattice_size,
    n_thermals = n_thermals,
    n_steps = n_steps,
    proposal_width = proposal_width,
    seed = seed
):
    results = {}
    
    for temperature, n_thermal in zip(temperatures, n_thermals):

        temperature = round(float(temperature), 2)

        print(
            f"Running T = {temperature:.2f} "
            f"with n_thermal = {n_thermal} "
            f"with n_steps = {n_steps}"
        )
        model = XYModel2D(
            N=lattice_size,
            T=temperature,
            J=1.0,
            seed=seed,
        )
        
        if temperature < critical_temperature:
            model.set_initial_condition("aligned")
        else:
            model.set_initial_condition("random")
            
        # Equilibration
        thermal_magnetizations = np.empty(n_thermal)
        #thermal_energies = np.empty(n_thermal)

        for step in range(n_thermal):
            model.sweep(proposal_width=proposal_width)
            thermal_magnetizations[step] = model.magnetization()
            #thermal_energies[step] = model.total_energy() / lattice_size**2
            
        # Compute correlation function
        magnetizations = np.empty(n_steps)
        #energies = np.empty(n_steps)
        
        for step in range(n_steps):
            model.sweep(proposal_width=proposal_width)
            magnetizations[step] = model.magnetization()
            #mx, my = model.magnetization_vector()
            #magnetizations[step] = mx
            
            #energies[step] = model.total_energy() / lattice_size**2
        
        tau, normalized_correlation = correlation_time(magnetizations)
        
        results[temperature] = {
            "tau": tau,
            "autocorrelation": normalized_correlation,
            "thermal_magnetizations": thermal_magnetizations,
            "magnetizations": magnetizations
        }
    
    return results

def run_single_simulation(
    temperature,
    lattice_size = lattice_size,
    n_thermal = n_thermals,
    n_steps = n_steps,
    proposal_width = proposal_width,
    vortex_interval = vortex_interval,
    seed = seed
):
    results = {}
    print(
        f"Running T = {temperature:.2f} "
        f"with n_thermal = {n_thermal} "
        f"with n_steps = {n_steps}"
    )
    model = XYModel2D(
        N=lattice_size,
        T=temperature,
        J=1.0,
        seed=seed,
    )
    
    if temperature < critical_temperature:
        model.set_initial_condition("aligned")
    else:
        model.set_initial_condition("random")
    
    # Equilibration
    thermal_magnetizations = np.empty(n_thermal)
    thermal_energies = np.empty(n_thermal)

    for step in range(n_thermal):
        model.sweep(proposal_width=proposal_width)
        thermal_magnetizations[step] = model.magnetization()
        thermal_energies[step] = model.total_energy() / lattice_size**2
        
    # Compute correlation function
    magnetizations = np.empty(n_steps)
    energies = np.empty(n_steps)
    
    for step in range(n_steps):
        model.sweep(proposal_width=proposal_width)
        magnetizations[step] = model.magnetization()
        energies[step] = model.total_energy() / lattice_size**2
    
    results[temperature] = {
        "thermal_magnetizations": thermal_magnetizations,
        "thermal_energies": thermal_energies,
        "magnetizations": magnetizations,
        "energies": energies,
        "final_theta": model.theta
    }
    
    return results

def run_full_temperature_analysis(
    temperatures,
    lattice_size=lattice_size,
    n_thermals=n_thermals,
    pilot_sweeps=pilot_sweeps,
    n_blocks=n_blocks,
    block_factor=block_factor,
    proposal_width=proposal_width,
    vortex_interval=vortex_interval,
    seed=seed,
):

    results = {}

    for temperature, n_thermal in zip(temperatures, n_thermals):

        temperature = round(float(temperature), 2)

        print(
            f"Running T = {temperature:.2f} "
            f"with n_thermal = {n_thermal}"
        )

        model = XYModel2D(
            N=lattice_size,
            T=temperature,
            J=1.0,
            seed=seed,
        )

        if temperature < critical_temperature:
            model.set_initial_condition("aligned")
        else:
            model.set_initial_condition("random")

        # Equilibration
        thermal_magnetizations = np.empty(n_thermal)
        thermal_energies = np.empty(n_thermal)

        for step in range(n_thermal):
            model.sweep(proposal_width=proposal_width)
            thermal_magnetizations[step] = model.magnetization()
            thermal_energies[step] = model.total_energy() / lattice_size**2

        # Pilot measurement to estimate tau
        pilot_magnetizations = np.empty(pilot_sweeps)

        for step in range(pilot_sweeps):
            model.sweep(proposal_width=proposal_width)
            pilot_magnetizations[step] = model.magnetization()

        tau, normalized_correlation = correlation_time(pilot_magnetizations)

        n_steps_auto = required_measurement_steps(
            tau=tau,
            n_blocks=n_blocks,
            block_factor=block_factor,
        )

        print(
            f"  tau = {tau:.2f}, "
            f"using n_steps = {n_steps_auto}"
        )

        # Final measurement run
        magnetizations = np.empty(n_steps_auto)
        energies = np.empty(n_steps_auto)

        vortex_counts = []
        antivortex_counts = []

        for step in range(n_steps_auto):
            model.sweep(proposal_width=proposal_width)

            magnetizations[step] = model.magnetization()
            energies[step] = model.total_energy()

            if step % vortex_interval == 0:
                n_vortices, n_antivortices = count_vortices(model.theta)
                vortex_counts.append(n_vortices)
                antivortex_counts.append(n_antivortices)

        number_of_spins = lattice_size**2
        energies_per_spin = energies / number_of_spins

        #tau, normalized_correlation = correlation_time(magnetizations)

        mean_m, error_m, independent_magnetizations = independent_mean_and_error(
            magnetizations,
            tau,
        )

        mean_e, error_e, independent_energies = independent_mean_and_error(
            energies_per_spin,
            tau,
        )

        chi_m, error_chi_m, specific_heat_value, error_specific_heat = block_errors(
            magnetizations,
            energies,
            temperature,
            number_of_spins,
            tau,
        )

        results[temperature] = {
            "tau": tau,
            #"pilot_tau": pilot_tau,
            "n_steps": n_steps_auto,
            "autocorrelation": normalized_correlation,

            "mean_m": mean_m,
            "error_m": error_m,

            "mean_e": mean_e,
            "error_e": error_e,

            "chi_m": chi_m,
            "error_chi_m": error_chi_m,

            "specific_heat": specific_heat_value,
            "error_specific_heat": error_specific_heat,

            "mean_vortices": np.mean(vortex_counts),
            "std_vortices": np.std(vortex_counts, ddof=1),

            "mean_antivortices": np.mean(antivortex_counts),
            "std_antivortices": np.std(antivortex_counts, ddof=1),

            "thermal_magnetizations": thermal_magnetizations,
            "thermal_energies": thermal_energies,
        }

        print(
            f"  final tau = {tau:.2f}, "
            f"n_steps = {n_steps_auto}, "
            f"<m> = {mean_m:.3f}, "
            f"<e> = {mean_e:.3f}, "
            f"vortices = {np.mean(vortex_counts):.1f}"
        )

    return results

#%%Animation functions

def animate_spin_configuration(lattice_size=lattice_size,temperature=T,n_frames=n_steps,
    sweeps_per_frame=1,proposal_width=proposal_width,initial_condition="random",seed=seed):
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

#%%Plotting functions

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
    
def plot_correlation_fit(results, temperature):
    temperature = round(float(temperature), 2)
    
    corr = results[temperature]["autocorrelation"]
    tau = results[temperature]["tau"]
    
    lags = np.arange(len(corr))
    
    fitted_decay = np.exp(-lags / tau)
    
    plt.figure(figsize=(7, 5))
    plt.plot(lags, corr, label="Autocorrelation")
    plt.plot(lags, fitted_decay, "--", label=f"exp(-t/τ), τ = {tau:.2f}")
    plt.axhline(0, linestyle=":", color="black")
    
    plt.xlabel("Lag time [sweeps]")
    plt.ylabel("Normalized autocorrelation")
    plt.title(f"Autocorrelation decay at T = {temperature}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_full_results(results, selected_temperatures=None):
    """
    Plot all main results from the full temperature analysis.

    This includes:
    - correlation time versus temperature,
    - magnetization,
    - energy,
    - magnetic susceptibility,
    - specific heat,
    - vortex counts,
    - equilibration curves,
    - autocorrelation functions used to estimate tau.
    """
    temperatures = np.array(sorted(results.keys()))

    tau = np.array([results[T]["tau"] for T in temperatures])

    mean_m = np.array([results[T]["mean_m"] for T in temperatures])
    error_m = np.array([results[T]["error_m"] for T in temperatures])

    mean_e = np.array([results[T]["mean_e"] for T in temperatures])
    error_e = np.array([results[T]["error_e"] for T in temperatures])

    chi_m = np.array([results[T]["chi_m"] for T in temperatures])
    error_chi_m = np.array([results[T]["error_chi_m"] for T in temperatures])

    specific_heat = np.array([results[T]["specific_heat"] for T in temperatures])
    error_specific_heat = np.array(
        [results[T]["error_specific_heat"] for T in temperatures]
    )

    vortices = np.array([results[T]["mean_vortices"] for T in temperatures])
    antivortices = np.array([results[T]["mean_antivortices"] for T in temperatures])

    if selected_temperatures is None:
        selected_temperatures = temperatures

    selected_temperatures = [
        round(float(temperature), 2)
        for temperature in selected_temperatures
    ]

    # Correlation time
    plt.figure(figsize=(7, 5))
    plt.plot(temperatures, tau, "o-")
    plt.axvline(critical_temperature, linestyle="--", label=r"$T_c \approx 0.881$")
    plt.xlabel("Temperature T")
    plt.ylabel(r"Correlation time $\tau$ [sweeps]")
    plt.title("Correlation time vs temperature")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Magnetization
    plt.figure(figsize=(7, 5))
    plt.errorbar(temperatures, mean_m, yerr=error_m, fmt="o-", capsize=4)
    plt.axvline(critical_temperature, linestyle="--", label=r"$T_c \approx 0.881$")
    plt.xlabel("Temperature T")
    plt.ylabel(r"Magnetization per spin $\langle |m| \rangle$")
    plt.title("Magnetization vs temperature")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Energy
    plt.figure(figsize=(7, 5))
    plt.errorbar(temperatures, mean_e, yerr=error_e, fmt="o-", capsize=4)
    plt.axvline(critical_temperature, linestyle="--", label=r"$T_c \approx 0.881$")
    plt.xlabel("Temperature T")
    plt.ylabel(r"Energy per spin $\langle e \rangle$")
    plt.title("Energy vs temperature")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Magnetic susceptibility
    plt.figure(figsize=(7, 5))
    plt.errorbar(
        temperatures,
        chi_m,
        yerr=error_chi_m,
        fmt="o-",
        capsize=4,
    )
    plt.axvline(critical_temperature, linestyle="--", label=r"$T_c \approx 0.881$")
    plt.xlabel("Temperature T")
    plt.ylabel(r"Magnetic susceptibility $\chi_m$")
    plt.title("Magnetic susceptibility vs temperature")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Specific heat
    plt.figure(figsize=(7, 5))
    plt.errorbar(
        temperatures,
        specific_heat,
        yerr=error_specific_heat,
        fmt="o-",
        capsize=4,
    )
    plt.axvline(critical_temperature, linestyle="--", label=r"$T_c \approx 0.881$")
    plt.xlabel("Temperature T")
    plt.ylabel("Specific heat C")
    plt.title("Specific heat vs temperature")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Vortices
    plt.figure(figsize=(7, 5))
    plt.plot(temperatures, vortices, "o-", label="Vortices")
    plt.plot(temperatures, antivortices, "s-", label="Antivortices")
    plt.axvline(critical_temperature, linestyle="--", label=r"$T_c \approx 0.881$")
    plt.xlabel("Temperature T")
    plt.ylabel("Average count")
    plt.title("Vortex and antivortex count vs temperature")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Thermalization magnetization
    plt.figure(figsize=(8, 5))
    for temperature in selected_temperatures:
        plt.plot(
            results[temperature]["thermal_magnetizations"],
            label=f"T = {temperature:.1f}",
        )

    plt.xlabel("Thermalization sweep")
    plt.ylabel(r"Magnetization per spin $|m|$")
    plt.title("Magnetization during equilibration")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Thermalization energy
    plt.figure(figsize=(8, 5))
    for temperature in selected_temperatures:
        plt.plot(
            results[temperature]["thermal_energies"],
            label=f"T = {temperature:.1f}",
        )

    plt.xlabel("Thermalization sweep")
    plt.ylabel("Energy per spin e")
    plt.title("Energy during equilibration")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Autocorrelation functions used to compute tau
    plt.figure(figsize=(8, 5))
    for temperature in selected_temperatures:
        autocorrelation = results[temperature]["autocorrelation"]

        plt.plot(
            autocorrelation,
            label=fr"T = {temperature:.1f}, $\tau$ = {results[temperature]['tau']:.1f}",
        )

    plt.axhline(0, linestyle="--")
    plt.xlabel("Lag time [sweeps]")
    plt.ylabel(r"Normalized autocorrelation $\chi(t)/\chi(0)$")
    plt.title("Autocorrelation functions used to estimate correlation time")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


#%%
# Simulate long and save data
for temperature in [1.1, 1.3, 1.5]:

    n_thermal = 2000
    n_steps = 10000
    seed = 0
    
    results = run_single_simulation(temperature = temperature, lattice_size = 50, n_thermal = n_thermal, n_steps = n_steps, seed = 0)
    
    
    # Save data
    file_name = "T" + str(temperature) + ".n_t" + str(n_thermal) + ".n_s" + str(n_steps) + ".s" + str(seed) + ".pkl"
    location_name = "saved_data/" + file_name
    
    # print(os.getcwd()) # to get where python is saving to
    os.chdir(r"C:\Users\David\Documents\Git\ComputationalPhysicsB")
    os.makedirs("saved_data", exist_ok=True)
    
    with open(location_name, "wb") as file:
        pickle.dump(results, file)
    
    print("Saved!")

#%%
# Load data
file_name = "T0.5.n_t20.n_s100.s0.pkl"
location_name = "saved_data/" + file_name

with open(location_name, "rb") as file:
    results = pickle.load(file)

#%%
# Plot magnetizations
magnetizations = results[0.5]["magnetizations"]
plt.plot(magnetizations, label=f"Temp = {0.5}", alpha=0.9)

plt.xlabel("Monte Carlo sweep")
plt.ylabel("Magnetization per spin |M|") 
plt.title("2D XY model: magnetization vs Monte Carlo sweep")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


#%%
# - Eerst correlatie tijd 5 keer doen temperatuur. (Check waarom die afhankelijk is van hoe lang je hem laat runnen.)
# %matplotlib qt
# %matplotlib inline

finding_correlation_temperatures = np.arange(0.5, 2.51, 0.2) #np.arange(0.5, 2.51, 0.2)
finding_correlation_n_thermals = [200,400,2000,1000,400,200,200,200,200,200,200] #[200,400,2000,1000,400,200,200,200,200,200,200]
finding_correlation_n_steps = 1000

results = run_correlation_time_analysis(finding_correlation_temperatures, n_steps = finding_correlation_n_steps, n_thermals = finding_correlation_n_thermals)

#%%
for temp in finding_correlation_temperatures:
    plot_correlation_fit(results, temperature=temp)
    print(f'T = {temp}, tau = ' + str(results[temp]["tau"]))


#%%
multiple_seeds_results = []
for seed in range(5):
    results = run_correlation_time_analysis([0.5], n_thermals = [200], n_steps = 3000, seed = seed)
    multiple_seeds_results.append(results)
#%%
plot_correlation_fit(multiple_seeds_results[0], temperature=0.5)
#%%

# show correlations
colors = ['b', 'g','r','y','m']
plt.figure(figsize=(7, 5))
for seed in range(len(multiple_seeds_results)):
    results = multiple_seeds_results[seed]
    
    corr = results[0.5]["autocorrelation"]
    tau = results[0.5]["tau"]
    
    lags = np.arange(len(corr))
    fitted_decay = np.exp(-lags / tau)

    plt.plot(lags, corr, label=f"$\chi$ Seed = {seed}", color = colors[seed], alpha = 0.9)
    plt.plot(lags, fitted_decay, "--", label=f"exp(-t/τ), τ = {tau:.2f}", color = colors[seed])
    
plt.axhline(0, linestyle=":", color="black")

plt.xlabel("Lag time [sweeps]")
plt.ylabel("Normalized autocorrelation")
plt.title(f"Autocorrelation decay at T = {0.5}")
plt.legend(loc = 'upper right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#%%

seed_0_results = multiple_seeds_results[0]
magnetizations = seed_0_results[0.5]["magnetizations"]
plt.plot(magnetizations, label=f"Temp = {0.5}", alpha=0.9)

plt.xlabel("Monte Carlo sweep")
plt.ylabel("Magnetization per spin |M|")
plt.title("2D XY model: magnetization vs Monte Carlo sweep")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


#%%
results = multiple_seeds_results[4]
plt.plot(results[0.5]["magnetizations"], label=f"Temp = {0.5}", alpha=0.9)

plt.xlabel("Monte Carlo sweep")
plt.ylabel("Magnetization per spin |M|")
plt.title("2D XY model: magnetization vs Monte Carlo sweep")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%%
# Show magentizations

for seed in range(len(multiple_seeds_results)):
    results = multiple_seeds_results[seed]
    plt.plot(results[0.5]["magnetizations"], label=f"Temp = {0.5}", alpha=0.9)

plt.xlabel("Monte Carlo sweep")
plt.ylabel("Magnetization per spin |M|")
plt.title("2D XY model: magnetization vs Monte Carlo sweep")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%% Total run of all observables and correlation time
results = run_full_temperature_analysis(temperatures, seed = 0)

plot_full_results(results)   

