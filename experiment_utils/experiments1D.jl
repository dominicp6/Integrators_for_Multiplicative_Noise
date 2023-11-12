module Experiments
include("../general_utils/calculus.jl")
include("../general_utils/potentials.jl")
include("../general_utils/diffusion_tensors.jl")
include("../general_utils/probability_utils.jl")
include("../general_utils/plotting_utils.jl")
include("../general_utils/misc_utils.jl")
include("../general_utils/transform_utils.jl")
include("../general_utils/experiment_utils.jl")
using HCubature, QuadGK, FHist, JLD2, Statistics, .Threads, ProgressBars, JSON, Random, StatsBase, TimerOutputs, LombScargle, FFTW, Plots, Interpolations
import .Calculus: differentiate1D
import .ProbabilityUtils: compute_1D_mean_L1_error, compute_1D_invariant_distribution
import .PlottingUtils: save_and_plot
import .MiscUtils: init_x0, create_directory_if_not_exists
import .TransformUtils: increment_g_counts, increment_I_counts
import .DiffusionTensors: Dconst1D
import .ExperimentUtils: make_experiment_folders
export run_1D_experiment, master_1D_experiment, run_1D_experiment_until_given_error, run_autocorrelation_experiment, run_ess_autocorrelation_experiment

"""
Run a 1D finite-time experiment using the specified integrator and parameters.

# Arguments
- `integrator`: The integrator function to use for the simulation.
- `num_repeats`: Number of repeats for the experiment.
- `V`: The potential function that describes the energy landscape.
- `D`: The diffusion coefficient function that defines the noise in the system.
- `T`: Total simulation time.
- `sigma`: The noise strength parameter.
- `stepsizes`: An array of step sizes to be used in the simulation.
- `probabilities`: The target probabilities to compute the convergence error.
- `bin_boundaries`: Bin boundaries for constructing histograms.
- `save_dir`: The directory path to save experiment results.
- `chunk_size`: Number of steps to run in each computational chunk to avoid memory issues.
- `q0`: The initial position of the trajectory. If not provided, it will be randomly initialized.

# Returns
- `convergence_errors`: A matrix containing convergence errors for each step size and repeat.

# Details
This function runs a 1D finite-time experiment with the specified integrator and system parameters. 
The experiment is repeated `num_repeats` times, each time with different initial conditions. For each combination of step size and repeat, the weak error w.r.t. the invariant distribution is computed.
"""
function run_1D_experiment(integrator, num_repeats, V, D, T, sigma, stepsizes, probabilities, bin_boundaries, save_dir; chunk_size=10000000, x0=nothing, noise_integrator=nothing, n=nothing)
    
    # Make master directory
    make_experiment_folders(save_dir, integrator, stepsizes, num_repeats, V, D, sigma, bin_boundaries, chunk_size, T)

    # Compute the symbolic derivative of the potential and diffusion functions
    Vprime = differentiate1D(V)
    D_squared = x -> D(x)^2
    D2prime = differentiate1D(D_squared)

    # Initialise empty data array
    convergence_errors = zeros(length(stepsizes), num_repeats)

    Threads.@threads for repeat in ProgressBar(1:num_repeats)
        # set the random seed for reproducibility
        Random.seed!(repeat) 

        # If no initial position is provided, randomly initialise
        x0 = init_x0(x0, dim=1)

        # Run the simulation for each specified step size
        for (stepsize_idx, dt) in enumerate(stepsizes)
            steps_remaining = floor(Int, T / dt)                
            total_samples = Int(steps_remaining)                               
            hist = Hist1D([], bin_boundaries)            

            while steps_remaining > 0
                # Run steps in chunks to minimise memory footprint
                steps_to_run = convert(Int, min(steps_remaining, chunk_size))
                
                # Run a chunk of the simulation
                q_chunk, _ = integrator(x0, Vprime, D, D2prime, sigma, steps_to_run, dt, nothing, noise_integrator, n)
                q0 = copy(q_chunk[end])
                hist += Hist1D(q_chunk, bin_boundaries)
                
                steps_remaining -= steps_to_run
            end

            # Compute the convergence error
            convergence_errors[stepsize_idx, repeat] = compute_1D_mean_L1_error(hist, probabilities, total_samples)
        end
    end

    # Save the error data and plot
    save_and_plot(integrator, convergence_errors, stepsizes, save_dir)

    @info "Mean L1 errors: $(mean(convergence_errors, dims=2))"
    @info "Standard deviation of L1 errors: $(std(convergence_errors, dims=2))"

    return convergence_errors
end


"""
Run a 1D experiment using a given integrator until a specified error level (L1 error w.r.t. the exact invariant measure) is reached.

Parameters:
- `integrator`: The integrator to be used in the simulation.
- `num_repeats`: Number of experiment repeats to perform.
- `V`: Potential function V(x) representing the energy landscape.
- `D`: Diffusion function D(x) representing the diffusion coefficient.
- `sigma`: The noise strength parameter.
- `stepsizes`: An array of time step sizes to be used in the simulation.
- `probabilities`: The target probability distribution to compare against.
- `bin_boundaries`: An array of bin boundaries for histogram computation.
- `save_dir`: The directory where results and checkpoints will be saved.
- `target_uncertainty`: The desired uncertainty level to be achieved.
- `chunk_size`: Number of simulation steps to be run in each chunk. Default is 10000.
- `checkpoint`: A boolean flag indicating whether to save checkpoints. Default is false.
- `q0`: The initial configuration for the simulation. Default is nothing, which generates a random configuration.
- `time_transform`: A boolean flag indicating whether to apply time transformation. Default is false.
- `space_transform`: A boolean flag indicating whether to apply space transformation. Default is false.
- `x_of_y`: A function that maps y-coordinates to corresponding x-coordinates for space-transformed integrators. Default is nothing.
    
Returns:
- `steps_until_uncertainty_data`: A matrix containing the number of steps taken until reaching the target uncertainty level for each step size and repeat.

Note:
- If `time_transform` is true, the potential function V(x) is transformed to ensure constant diffusion.
- If `space_transform` is true, the potential function V(x) is transformed based on the provided mapping x_of_y to ensure constant diffusion.
"""
function run_1D_experiment_until_given_error(integrator, num_repeats, V, D, sigma, stepsizes, probabilities, bin_boundaries, save_dir, target_error; chunk_size=10000)
    # TODO: need to update this function
    
    # Create the experiment folders
    make_experiment_folders(save_dir, integrator, stepsizes, checkpoint, num_repeats, V, D, sigma, bin_boundaries, chunk_size, target_uncertainty=target_error)

    # Compute the symbolic derivatives of the potential and diffusion functions
    Vprime = differentiate1D(V)
    Dprime = differentiate1D(D)

    # Initialise the data array
    steps_until_uncertainty_data = zeros(length(stepsizes), num_repeats)

    Threads.@threads for repeat in ProgressBar(1:num_repeats)
        # set the random seed for reproducibility
        Random.seed!(repeat) 
        q0 = init_x0(q0)

        # Run the simulation for each specified step size
        for (stepsize_idx, dt) in enumerate(stepsizes)

            steps_ran = 0                                    
            chunk_number = 0                                 
            error = Inf                                    
            hist = Hist1D([], bin_boundaries)                 

            while error > target_error
                # Run a chunk of the simulation
                q_chunk, _ = integrator(q0, Vprime, D, Dprime, sigma, steps_to_run, dt)
                q0 = copy(q_chunk[end])
                hist += Hist1D(q_chunk, bin_boundaries)
                steps_ran += chunk_size
                
                error = compute_1D_mean_L1_error(hist, probabilities, steps_ran)
            end

            # Populate the data array
            steps_until_uncertainty_data[stepsize_idx, repeat] = steps_ran
        end
    end

    # Save the data and plot the results
    save_and_plot(integrator, steps_until_uncertainty_data, stepsizes, save_dir, ylabel="Steps until uncertainty < $(target_error)", error_in_mean=true)

    return steps_until_uncertainty_data
end


"""
Run a master 1D experiment with multiple integrators.

Parameters:
- `integrators`: An array of integrators to be used in the experiments.
- `num_repeats`: Number of experiment repeats to perform for each integrator.
- `V`: Potential function V(x) representing the energy landscape.
- `D`: Diffusion function D(x) representing the diffusion coefficient.
- `T`: Total time for the simulation.
- `sigma`: The noise strength parameter.
- `stepsizes`: An array of time step sizes to be used in the simulation.
- `bin_boundaries`: An array of bin boundaries for histogram computation.
- `save_dir`: The directory where results and time convergence data will be saved.
- `chunk_size`: Number of simulation steps to be run in each chunk. Default is 10000000.
- `checkpoint`: A boolean flag indicating whether to save checkpoints. Default is false.
- `q0`: The initial configuration for the simulation. Default is nothing, which generates a random configuration.
- `time_transform`: A boolean flag indicating whether to apply time transformation. Default is false.
- `space_transform`: A boolean flag indicating whether to apply space transformation. Default is false.
- `x_of_y`: A function that maps y-coordinates to corresponding x-coordinates for space-transformed integrators. Default is nothing.

Returns:
- The function saves the results of each experiment in the specified `save_dir` and also saves the time convergence data in a file named "time.json".
"""
function master_1D_experiment(integrators, num_repeats, V, D, T, sigma, stepsizes, bin_boundaries, save_dir; chunk_size=10000000, x0=nothing, noise_integrator=nothing, n=nothing)
    to = TimerOutput()

    @info "Computing the Invariant Distribution"
    exact_invariant_distribution = compute_1D_invariant_distribution(V, sigma, bin_boundaries)

    @info "Running Experiments"
    for integrator in integrators
        @info "Running $(string(nameof(integrator))) experiment"
        @timeit to "Exp$(string(nameof(integrator)))" begin
        # reset the random seed for reproducibility
        Random.seed!(1)
            _ = run_1D_experiment(integrator, num_repeats, V, D, T, sigma, stepsizes, exact_invariant_distribution, bin_boundaries, save_dir, chunk_size=chunk_size, x0=x0, noise_integrator=noise_integrator, n=n)
        end
    end

    # save the time convergence_data
    open("$(save_dir)/time.json", "w") do io
        JSON.print(io, TimerOutputs.todict(to), 4)
    end
end

end # module