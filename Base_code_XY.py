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
n_thermal = 3000         # Number of steps to equilibrate
n_steps =  5000          # Number of steps done after equilibration
proposal_width = np.pi/2 # Theta is updated with steps of [-proposal_width, proposal_width]
T = 0.7                  # Basis temperature for simulations
lattice_size = 80        # Simulations without specified N are run with this number
seed = 43                #
vortex_interval = 20     # Count number of vortices every ... steps
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


#%% Seperate functions for milestones and plotting
def run_sizes(sizes, T=T, n_thermal=n_thermal, n_steps=n_steps, seed=seed):
    results = {}

    for N in sizes:
        print(f"Running N = {N}")
        model = XYModel2D(N=N, T=T, seed=seed)
        magnetizations = model.simulate(n_thermal=n_thermal, n_steps=n_steps)
        results[N] = magnetizations
        print(f"  <|M|> = {np.mean(magnetizations):.4f}")

    return results

def run_temperatures(temperatures,lattice_size=lattice_size ,n_steps=n_steps,proposal_width=proposal_width,
   seed=seed):
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

        magnetizations = model.simulate()

        results[temperature] = magnetizations

    return results

        
def compare_initial_conditions(lattice_size=lattice_size,temperature=T,n_steps=n_steps,proposal_width=proposal_width,seed=seed):
    """
    Run two simulations: one random start and one aligned start.
    """
    random_model = XYModel2D(N=lattice_size, T=temperature, seed=seed)

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
    Estimate tau by summing normalized autocorrelation until it becomes negative.
    """
    normalized_correlation = autocorrelation(values)

    positive_values = []

    for value in normalized_correlation:
        if value < 0:
            break

        positive_values.append(value)

    tau = np.sum(positive_values)

    return tau, normalized_correlation

def run_full_temperature_analysis(
    temperatures,
    lattice_size=lattice_size,
    n_thermal=n_thermal,
    n_steps=n_steps,
    proposal_width=proposal_width,
    vortex_interval=vortex_interval,
    seed=seed,
):
    """
    For each temperature:
    1. Equilibrate once.
    2. Run one measurement simulation.
    3. Measure magnetization, energy, vortices.
    4. Compute tau, m, e, chi_M, and C.
    """
    results = {}

    for temperature in temperatures:
        temperature = round(float(temperature), 2)
        print(f"Running T = {temperature:.2f}")

        model = XYModel2D(
            N=lattice_size,
            T=temperature,
            J=1.0,
            seed=seed,
        )

        # Equilibration
        thermal_magnetizations = np.empty(n_thermal)
        thermal_energies = np.empty(n_thermal)
        
        for step in range(n_thermal):
            model.sweep(proposal_width=proposal_width)
        
            thermal_magnetizations[step] = model.magnetization()
            thermal_energies[step] = model.total_energy() / lattice_size**2

        magnetizations = np.empty(n_steps)
        energies = np.empty(n_steps)

        vortex_counts = []
        antivortex_counts = []

        # Measurement run
        for step in range(n_steps):
            model.sweep(proposal_width=proposal_width)

            magnetizations[step] = model.magnetization()
            energies[step] = model.total_energy()

            if step % vortex_interval == 0:
                n_vortices, n_antivortices = count_vortices(model.theta)
                vortex_counts.append(n_vortices)
                antivortex_counts.append(n_antivortices)

        number_of_spins = lattice_size**2
        beta = 1.0 / temperature

        energies_per_spin = energies / number_of_spins

        tau, normalized_correlation = correlation_time(magnetizations)

        mean_m = np.mean(magnetizations)
        std_m = np.std(magnetizations, ddof=1)

        mean_e = np.mean(energies_per_spin)
        std_e = np.std(energies_per_spin, ddof=1)

        chi_m = beta * number_of_spins * (
            np.mean(magnetizations**2) - np.mean(magnetizations) ** 2
        )

        specific_heat = beta**2 / number_of_spins * (
            np.mean(energies**2) - np.mean(energies) ** 2
        )

        results[temperature] = {
            "tau": tau,
            "autocorrelation": normalized_correlation,
            "mean_m": mean_m,
            "std_m": std_m,
            "mean_e": mean_e,
            "std_e": std_e,
            "chi_m": chi_m,
            "specific_heat": specific_heat,
            "mean_vortices": np.mean(vortex_counts),
            "std_vortices": np.std(vortex_counts, ddof=1),
            "mean_antivortices": np.mean(antivortex_counts),
            "std_antivortices": np.std(antivortex_counts, ddof=1),
            "thermal_magnetizations": thermal_magnetizations,
            "thermal_energies": thermal_energies,
        }

        print(
            f"  tau = {tau:.2f}, "
            f"<m> = {mean_m:.3f}, "
            f"<e> = {mean_e:.3f}, "
            f"vortices = {np.mean(vortex_counts):.1f}"
        )

    return results

def plot_full_results(results):
    """
    Plot tau, m, e, chi_M, C, and vortex counts versus temperature.
    """
    temperatures = np.array(sorted(results.keys()))

    tau = np.array([results[T]["tau"] for T in temperatures])
    mean_m = np.array([results[T]["mean_m"] for T in temperatures])
    std_m = np.array([results[T]["std_m"] for T in temperatures])
    mean_e = np.array([results[T]["mean_e"] for T in temperatures])
    std_e = np.array([results[T]["std_e"] for T in temperatures])
    chi_m = np.array([results[T]["chi_m"] for T in temperatures])
    specific_heat = np.array([results[T]["specific_heat"] for T in temperatures])
    vortices = np.array([results[T]["mean_vortices"] for T in temperatures])
    antivortices = np.array([results[T]["mean_antivortices"] for T in temperatures])

    plt.figure(figsize=(7, 5))
    plt.plot(temperatures, tau, "o-")
    plt.axvline(0.881, linestyle="--", label="Tc ≈ 0.881")
    plt.xlabel("Temperature T")
    plt.ylabel("Correlation time τ [sweeps]")
    plt.title("Correlation time vs temperature")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.errorbar(temperatures, mean_m, yerr=std_m, fmt="o-", capsize=4)
    plt.xlabel("Temperature T")
    plt.ylabel("Magnetization per spin <|m|>")
    plt.title("Magnetization vs temperature")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.errorbar(temperatures, mean_e, yerr=std_e, fmt="o-", capsize=4)
    plt.xlabel("Temperature T")
    plt.ylabel("Energy per spin <e>")
    plt.title("Energy vs temperature")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(temperatures, chi_m, "o-")
    plt.axvline(0.881, linestyle="--", label="Tc ≈ 0.881")
    plt.xlabel("Temperature T")
    plt.ylabel("Magnetic susceptibility χM")
    plt.title("Magnetic susceptibility vs temperature")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(temperatures, specific_heat, "o-")
    plt.axvline(0.881, linestyle="--", label="Tc ≈ 0.881")
    plt.xlabel("Temperature T")
    plt.ylabel("Specific heat C")
    plt.title("Specific heat vs temperature")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(temperatures, vortices, "o-", label="Vortices")
    plt.plot(temperatures, antivortices, "s-", label="Anti-vortices")
    plt.xlabel("Temperature T")
    plt.ylabel("Average count")
    plt.title("Vortex and anti-vortex count vs temperature")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_equilibration_from_full_run(results, selected_temperatures):
    """
    Plot thermalization curves before the measurement phase.
    """
    plt.figure(figsize=(8, 5))

    for temperature in selected_temperatures:
        temperature = round(float(temperature), 2)

        plt.plot(
            results[temperature]["thermal_magnetizations"],
            label=f"T = {temperature:.1f}",
        )

    plt.xlabel("Thermalization sweep")
    plt.ylabel("Magnetization per spin |m|")
    plt.title("Equilibration before measurement")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))

    for temperature in selected_temperatures:
        temperature = round(float(temperature), 2)

        plt.plot(
            results[temperature]["thermal_energies"],
            label=f"T = {temperature:.1f}",
        )

    plt.xlabel("Thermalization sweep")
    plt.ylabel("Energy per spin e")
    plt.title("Energy equilibration before measurement")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
# def autocorrelation(magnetizations):
#     """
#     Compute the normalized autocorrelation function of magnetization.

#     Parameters
#     ----------
#     magnetizations : np.ndarray
#         Magnetization values measured after equilibration.

#     Returns
#     -------
#     np.ndarray
#         Normalized autocorrelation chi(t) / chi(0).
#     """
#     n_measurements = len(magnetizations)
#     correlations = np.empty(n_measurements)

#     for lag_time in range(n_measurements):
#         first_values = magnetizations[: n_measurements - lag_time]
#         shifted_values = magnetizations[lag_time:]

#         correlations[lag_time] = (
#             np.mean(first_values * shifted_values)
#             - np.mean(first_values) * np.mean(shifted_values)
#         )

#     return correlations / correlations[0]


# def correlation_time(magnetizations):
#     """
#     Estimate the correlation time by summing the normalized autocorrelation
#     until it first becomes negative.
#     """
#     normalized_correlation = autocorrelation(magnetizations)

#     positive_values = []

#     for value in normalized_correlation:
#         if value < 0:
#             break

#         positive_values.append(value)

#     return np.sum(positive_values), normalized_correlation


# def run_correlation_times(temperatures,lattice_size=lattice_size,n_thermal=1000,n_steps=5000,
#     proposal_width=proposal_width,seed=seed):
#     """
#     Estimate correlation time tau for each temperature.
#     """
#     tau_values = {}
#     autocorrelations = {}

#     for temperature in temperatures:
#         print(f"Running T = {temperature:.1f}")

#         model = XYModel2D(N=lattice_size,T=temperature,J=1.0,seed=seed)

#         magnetizations = model.simulate(
#             n_thermal=n_thermal,
#             n_steps=n_steps,
#             proposal_width=proposal_width,
#         )

#         tau, normalized_correlation = correlation_time(magnetizations)

#         tau_values[temperature] = tau
#         autocorrelations[temperature] = normalized_correlation

#         print(f"  tau = {tau:.2f} sweeps")

#     return tau_values, autocorrelations


# def plot_correlation_times(tau_values):
#     """
#     Plot correlation time as a function of temperature.
#     """
#     temperatures = np.array(list(tau_values.keys()))
#     taus = np.array(list(tau_values.values()))

#     plt.figure(figsize=(8, 5))
#     plt.plot(temperatures, taus, "o-")
#     plt.axvline(0.881, linestyle="--", label="Tc ≈ 0.881")

#     plt.xlabel("Temperature T")
#     plt.ylabel("Correlation time τ [sweeps]")
#     plt.title("Correlation time of the 2D XY model")
#     plt.legend()
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.show()


# def plot_autocorrelation_examples(autocorrelations, selected_temperatures):
#     """
#     Plot selected autocorrelation functions.
#     """
#     plt.figure(figsize=(8, 5))

#     for temperature in selected_temperatures:
#         plt.plot(
#             autocorrelations[temperature],
#             label=f"T = {temperature:.1f}",
#         )

#     plt.xlabel("Lag time t [sweeps]")
#     plt.ylabel("χ(t) / χ(0)")
#     plt.title("Normalized autocorrelation functions")
#     plt.legend()
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.show()
    
    
# def measure_observables(
#     model,
#     n_thermal=n_thermal,
#     n_steps=n_steps,
#     proposal_width=proposal_width,
# ):
#     """
#     Equilibrate the model, then measure magnetization and energy.
#     """
#     for _ in range(n_thermal):
#         model.sweep(proposal_width=proposal_width)

#     magnetizations = np.empty(n_steps)
#     energies = np.empty(n_steps)

#     for step_index in range(n_steps):
#         model.sweep(proposal_width=proposal_width)

#         magnetizations[step_index] = model.magnetization()
#         energies[step_index] = model.total_energy()

#     return magnetizations, energies

# def thermodynamic_quantities(
#     magnetizations,
#     energies,
#     lattice_size,
#     temperature,
# ):
#     """
#     Compute mean thermodynamic quantities from measured time series.
#     """
#     number_of_spins = lattice_size**2
#     beta = 1.0 / temperature

#     mean_magnetization = np.mean(magnetizations)
#     std_magnetization = np.std(magnetizations, ddof=1)

#     energies_per_spin = energies / number_of_spins
#     mean_energy_per_spin = np.mean(energies_per_spin)
#     std_energy_per_spin = np.std(energies_per_spin, ddof=1)

#     magnetic_susceptibility = (
#         beta
#         * number_of_spins
#         * (np.mean(magnetizations**2) - np.mean(magnetizations) ** 2)
#     )

#     specific_heat = (
#         beta**2
#         / number_of_spins
#         * (np.mean(energies**2) - np.mean(energies) ** 2)
#     )

#     return {
#         "mean_magnetization": mean_magnetization,
#         "std_magnetization": std_magnetization,
#         "mean_energy_per_spin": mean_energy_per_spin,
#         "std_energy_per_spin": std_energy_per_spin,
#         "magnetic_susceptibility": magnetic_susceptibility,
#         "specific_heat": specific_heat,
#     }

# def run_thermodynamic_observables(
#     temperatures,
#     lattice_size=lattice_size,
#     n_thermal=n_thermal,
#     n_steps=n_steps,
#     proposal_width= proposal_width,
#     seed=seed,
# ):
#     """
#     Compute thermodynamic observables as a function of temperature.
#     """
#     results = {}

#     for temperature in temperatures:
#         temperature_key = round(temperature, 2)
#         print(f"Running T = {temperature_key:.2f}")

#         model = XYModel2D(
#             N=lattice_size,
#             T=temperature_key,
#             J=1.0,
#             seed=seed,
#         )

#         magnetizations, energies = measure_observables(
#             model=model,
#             n_thermal=n_thermal,
#             n_steps=n_steps,
#             proposal_width=proposal_width,
#         )

#         results[temperature_key] = thermodynamic_quantities(
#             magnetizations=magnetizations,
#             energies=energies,
#             lattice_size=lattice_size,
#             temperature=temperature_key,
#         )

#     return results

# def plot_thermodynamic_results(results):
#     """
#     Plot m, e, magnetic susceptibility, and specific heat versus temperature.
#     """
#     temperatures = np.array(sorted(results.keys()))

#     magnetizations = np.array(
#         [results[temperature]["mean_magnetization"] for temperature in temperatures]
#     )
#     energies = np.array(
#         [results[temperature]["mean_energy_per_spin"] for temperature in temperatures]
#     )
#     susceptibilities = np.array(
#         [results[temperature]["magnetic_susceptibility"] for temperature in temperatures]
#     )
#     specific_heats = np.array(
#         [results[temperature]["specific_heat"] for temperature in temperatures]
#     )

#     plt.figure(figsize=(7, 5))
#     plt.plot(temperatures, magnetizations, "o-")
#     plt.xlabel("Temperature T")
#     plt.ylabel("Mean magnetization per spin <|m|>")
#     plt.title("Magnetization vs temperature")
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.show()

#     plt.figure(figsize=(7, 5))
#     plt.plot(temperatures, energies, "o-")
#     plt.xlabel("Temperature T")
#     plt.ylabel("Mean energy per spin <e>")
#     plt.title("Energy vs temperature")
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.show()

#     plt.figure(figsize=(7, 5))
#     plt.plot(temperatures, susceptibilities, "o-")
#     plt.xlabel("Temperature T")
#     plt.ylabel("Magnetic susceptibility per spin χM")
#     plt.title("Magnetic susceptibility vs temperature")
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.show()

#     plt.figure(figsize=(7, 5))
#     plt.plot(temperatures, specific_heats, "o-")
#     plt.xlabel("Temperature T")
#     plt.ylabel("Specific heat per spin C")
#     plt.title("Specific heat vs temperature")
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.show()
#%% Milestone 7.1: Vary system size

sizes = [10, 20, 50]
    
results = run_sizes(sizes)
    
plot_magnetization_time_series(results)
#plot_average_magnetization(results)

#%% Milestone 7.2: Vary temperature
temperatures = np.arange(0.5, 2.51, 0.2)

results = run_temperatures(temperatures)
plot_temperature_series(results)

#%% Milestone 7.3: random and aligned initial configurations
random_magnetizations, aligned_magnetizations = compare_initial_conditions()

plot_initial_condition_comparison(random_magnetizations,aligned_magnetizations,temperature=T)

#%% Milestone 7.4: animate spin configuration as colored blocks
%matplotlib qt

animation = animate_spin_configuration()

#%% Animate spin configuration as arrows
%matplotlib qt

animation = animate_spin_arrows()

#%% Milestone 8.1: Correlation times
# temperatures = np.arange(0.5, 2.51, 0.2)

# tau_values, autocorrelations = run_correlation_times(
#     temperatures=temperatures,
#     lattice_size=50,
#     n_thermal=1000,
#     n_steps=2000,
#     proposal_width=proposal_width,
#     seed=seed,
# )

# plot_correlation_times(tau_values)

# plot_autocorrelation_examples(
#     autocorrelations,
#     selected_temperatures = np.arange(0.5, 2.51, 0.2),
# )
# plot_thermodynamic_results(thermo_results)

#%% Milestone 8.2: Measure observable quantities
# temperatures = np.round(np.arange(0.5, 2.51, 0.2), 2)

# thermo_results = run_thermodynamic_observables(temperatures=temperatures)

#%% Total run of all observables and correlation time
temperatures = np.arange(0.5, 2.51, 0.2)

results = run_full_temperature_analysis(temperatures)

plot_full_results(results)

plot_equilibration_from_full_run(results,selected_temperatures=[0.5, 0.9, 1.1, 2.5])