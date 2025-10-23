module Experiments2D
include("../general_utils/calculus.jl")
include("../general_utils/potentials.jl")
include("../general_utils/probability_utils.jl")
include("../general_utils/plotting_utils.jl")
include("../general_utils/misc_utils.jl")
include("../general_utils/transform_utils.jl")
include("../general_utils/diffusion_tensors.jl")
include("../general_utils/integrators.jl")
using HCubature, QuadGK, FHist, JLD2, Statistics, .Threads, ProgressBars, JSON, Random, StatsBase, TimerOutputs, LinearAlgebra
using Plots
import .Calculus: differentiate2D, symbolic_matrix_divergence2D
import .ProbabilityUtils: compute_2D_mean_L1_error, compute_2D_invariant_distribution
import .PlottingUtils: save_and_plot, plot_histograms
import .MiscUtils: init_x0, create_directory_if_not_exists, find_row_indices, remove_rows

"""
Creates necessary directories and save experiment parameters for the 2D experiment.
"""
function make_experiment2D_folders(save_dir, integrator, stepsizes, checkpoint, num_repeats, V, D, sigma, x_bins, y_bins, chunk_size; T=nothing, target_uncertainty=nothing, noise_integrator=nothing, n=nothing)

    if !isfile("$(save_dir)/info.json") 
        @info "Saving metadata"
        metadata = Dict("integrator" => string(nameof(integrator)),
                        "num_repeats" => num_repeats, 
                        "V" => string(nameof(V)), 
                        "D" => string(nameof(D)), 
                        "T" => T, 
                        "target_uncertainty" => target_uncertainty,
                        "noise_integrator" => string(nameof(noise_integrator)),
                        "n" => n,
                        "sigma" => sigma, 
                        "stepsizes" => stepsizes, 
                        "x_bins" => x_bins, 
                        "y_bins" => y_bins,
                        "chunk_size" => chunk_size
                        )
        open("$(save_dir)/info.json", "w") do f
            JSON.print(f, metadata, 4)
        end
    end

end

function run_2D_experiment(integrator, num_repeats, V, D, T, sigma, stepsizes, probabilities, x_bins, y_bins, save_dir; chunk_size=10000000, checkpoint=false, x0=nothing, noise_integrator=nothing, n=nothing, max_retries=3)
    make_experiment2D_folders(save_dir, integrator, stepsizes, checkpoint, num_repeats, V, D, sigma, x_bins, y_bins, chunk_size, T=T, target_uncertainty=nothing, noise_integrator=noise_integrator, n=n)

    # Compute symbolic derivatives of the potential and diffusion
    Vprime = differentiate2D(V)
    DDT = (x,y) -> D(x,y) * Base.transpose(D(x,y))
    div_DDT = symbolic_matrix_divergence2D(DDT)

    # Initialise empty data arrays
    convergence_errors = zeros(length(stepsizes), num_repeats)
    histogram_data = Matrix{Hist2D}(undef, length(stepsizes), num_repeats)

    Threads.@threads for repeat in ProgressBar(1:num_repeats)
        # set the random seed for reproducibility
        Random.seed!(repeat) 

        # If no initial position is provided, randomly initialise
        x0 = init_x0(x0, dim=2)

        # Run the simulation for each specified step size
        for (stepsize_idx, dt) in enumerate(reverse(stepsizes))
            stepsize_idx = length(stepsizes) + 1 - stepsize_idx
            steps_remaining = floor(T / dt)
            total_samples = Int(steps_remaining)

            # Create a zeros array of the correct size for the histogram
            hist = Hist2D(; binedges=(x_bins, y_bins))  # histogram of the trajectory

            steps_done = 0
            retry_count = 0
            x_start = deepcopy(x0)
            while retry_count < max_retries
                #try
                    while steps_remaining > 0

                        # Determine the number of steps to run in this chunk
                        steps_to_run = Int(min(steps_remaining, chunk_size))

                        # Save the state at the start of the chunk
                        x_start = deepcopy(x0)

                        x_chunk, _ = integrator(x0, Vprime, D, div_DDT, sigma, steps_to_run, dt, nothing, noise_integrator, nothing)
                        hist += Hist2D((x_chunk[1,:], x_chunk[2,:]); binedges=(x_bins, y_bins))

                        bins_x = bincenters(hist)[1]
                        bins_y = bincenters(hist)[2]
                        freq = bincounts(hist)
                        xlabel = "x"
                        ylabel = "y"
                        title = "$(dt)"
                        
                        # Plot heatmap
                        #h = heatmap(bins_x, bins_y, freq, aspect_ratio=:equal, color=:viridis,
                        #        xlabel=xlabel, ylabel=ylabel, title=title)

                        #savefig(h, "$(save_dir)/h=$(dt).png")

                        # New starting coordinate in the end of the previous chunk
                        x0 = x_chunk[:,end]

                        steps_remaining -= steps_to_run
                        steps_done += steps_to_run

                        # Reset the retry count if successfully completed a chunk
                        retry_count = 0

                        # Keep track of experiment progress in progress.txt
                        if repeat == 1
                            # Read the JSON file
                            file_path = "$(save_dir)/progress.json"
                            json_data = JSON.parsefile(file_path)
                            json_data[string(nameof(integrator))][string(dt)] = steps_done / total_samples

                            # Write the modified data back to the JSON file
                            open(file_path, "w") do io
                                JSON.print(io, json_data, 4)
                            end
                        end
                    end

                    convergence_errors[stepsize_idx, repeat] = compute_2D_mean_L1_error(hist, probabilities, total_samples)
                    delay = repeat / num_repeats
                    sleep(delay)
                    errors_path = "$(save_dir)/partial_results.txt"
                    open(errors_path, "a") do io
                        message = string(string(nameof(integrator)), ", ", dt, ", ", repeat, ", ", convergence_errors[stepsize_idx, repeat], "\n")
                        write(io, message)
                    end


                    break  # Exit retry while loop if successful
                # catch
                #     retry_count += 1
                #     if retry_count >= max_retries
                #         errors_path = "$(save_dir)/errors.txt"
                #         open(errors_path, "a") do io
                #             message = string("Experiment with ", string(nameof(integrator)), " stepsize ", dt, " failed in repeat ", repeat, "\n")
                #             write(io, message)
                #         end
                #         convergence_errors[stepsize_idx, repeat] = -1
                #         break
                #     else
                #         # Reset state to the start of the previous batch
                #         x0 = deepcopy(x_start)
                #     end
                # end
            end
        end
    end

    # Save the error data and plot
    save_and_plot(integrator, convergence_errors, stepsizes, save_dir)
    #plot_histograms(integrator, histogram_data, stepsizes, save_dir)

    # Print the mean and standard deviation of the L1 errors
    @info "Mean L1 errors: $(mean(convergence_errors, dims=2))"
    @info "Standard deviation of L1 errors: $(std(convergence_errors, dims=2))"

    return convergence_errors
end


# """
# Run a 2D finite-time experiment using the specified integrator and parameters.

# # Arguments
# - `integrator`: The integrator function to use for the simulation.
# - `num_repeats`: Number of repeats for the experiment.
# - `V`: The potential function that describes the energy landscape.
# - `D`: The diffusion coefficient function that defines the noise in the system.
# - `R`: The constant-coefficient matrix that is needed for the time-transformation (see paper for details).
# - `T`: Total simulation time.
# - `sigma`: The noise strength parameter.
# - `stepsizes`: An array of step sizes to be used in the simulation.
# - `probabilities`: The target probabilities to compute the convergence error.
# - `x_bins`: Bin boundaries for the x-axis.
# - `y_bins`: Bin boundaries for the y-axis.
# - `save_dir`: The directory path to save experiment results.
# - `chunk_size`: Number of steps to run in each computational chunk to avoid memory issues.
# - `checkpoint`: If true, save intermediate results in checkpoints.
# - `q0`: The initial position of the trajectory. If not provided, it will be randomly initialized.
# - `time_transform`: If true, apply time transformation to the potential and diffusion.

# # Returns
# - `convergence_errors`: A matrix containing convergence errors for each step size and repeat.

# # Details
# This function runs a 1D finite-time experiment with the specified integrator and system parameters. It supports various configurations, including time and space transformations. 
# The experiment is repeated `num_repeats` times, each time with different initial conditions. For each combination of step size and repeat, the weak error w.r.t. the invariant distribution is computed.

# Note: The `V` and `D` functions may be modified internally to implement time or space transformations, based on the provided `time_transform` and `space_transform` arguments.
# """
# function run_2D_experiment(integrator, num_repeats, V, D, T, sigma, stepsizes, probabilities, x_bins, y_bins, save_dir; chunk_size=10000000, checkpoint=false, x0=nothing, noise_integrator=nothing, n=nothing)
    
#     make_experiment2D_folders(save_dir, integrator, stepsizes, checkpoint, num_repeats, V, D, sigma, x_bins, y_bins, chunk_size, T=T, target_uncertainty=nothing, noise_integrator=noise_integrator, n=n)

#     # Compute symbolic derivatives of the potential and diffusion
#     Vprime = differentiate2D(V)
#     DDT = (x,y) -> D(x,y) * Base.transpose(D(x,y))
#     div_DDT = symbolic_matrix_divergence2D(DDT)

#     # Initialise empty data arrays
#     convergence_errors = zeros(length(stepsizes), num_repeats)
#     histogram_data = Matrix{Hist2D}(undef, length(stepsizes), num_repeats)

#     Threads.@threads for repeat in ProgressBar(1:num_repeats)
#         # set the random seed for reproducibility
#         Random.seed!(repeat) 

#         # If no initial position is provided, randomly initialise
#         x0 = init_x0(x0, dim=2)

#         # Run the simulation for each specified step size
#         for (stepsize_idx, dt) in enumerate(stepsizes)
#             try
#                 steps_remaining = floor(T / dt)                 
#                 total_samples = Int(steps_remaining)                                  

#                 # Create a zeros array of the correct size for the histogram
#                 num_x_bins = length(x_bins) - 1
#                 num_y_bins = length(y_bins) - 1
#                 zeros_array = zeros(Int64, num_x_bins, num_y_bins)

#                 hist = Hist2D(zeros_array, (x_bins, y_bins))                 # histogram of the trajectory

#                 while steps_remaining > 0
#                     # Run steps in chunks to minimise memory footprint
#                     steps_to_run = convert(Int, min(steps_remaining, chunk_size))
#                     x_chunk, _ = integrator(x0, Vprime, D, div_DDT, sigma, steps_to_run, dt, nothing, noise_integrator, nothing)
#                     hist += Hist2D((x_chunk[1,:], x_chunk[2,:]), (x_bins, y_bins))
#                     steps_remaining -= steps_to_run
#                 end

#                 convergence_errors[stepsize_idx, repeat] = compute_2D_mean_L1_error(hist, probabilities, total_samples)

#                 histogram_data[stepsize_idx, repeat] = hist

#                 if checkpoint
#                     # Save the histogram
#                     save("$(save_dir)/checkpoints/$(string(nameof(integrator)))/h=$dt/$(repeat).jld2", "data", hist)
#                 end
#             catch
#                 println("Experiment with stepsize ", dt, " failed in repeat ", repeat)
#                 convergence_errors[stepsize_idx, repeat] = -1
#             end
#         end
#     end

#     # Save the error data and plot
#     save_and_plot(integrator, convergence_errors, stepsizes, save_dir)
#     # plot_histograms(integrator, histogram_data, stepsizes, save_dir)

#     # Print the mean and standard deviation of the L1 errors
#     @info "Mean L1 errors: $(mean(convergence_errors, dims=2))"
#     @info "Standard deviation of L1 errors: $(std(convergence_errors, dims=2))"

#     return convergence_errors
# end


"""
Run a master 2D experiment with multiple integrators.

Parameters:
- `integrators`: An array of integrators to be used in the experiments.
- `num_repeats`: Number of experiment repeats to perform for each integrator.
- `V`: Potential function V(x) representing the energy landscape.
- `D`: Diffusion function D(x) representing the diffusion coefficient.
- `R`: The constant-coefficient matrix that is needed for the time-transformation (see paper for details).
- `T`: Total time for the simulation.
- `sigma`: The noise strength parameter.
- `stepsizes`: An array of time step sizes to be used in the simulation.
- `xmin`: The minimum x-coordinate of the domain.
- `ymin`: The minimum y-coordinate of the domain.
- `xmax`: The maximum x-coordinate of the domain.
- `ymax`: The maximum y-coordinate of the domain.
- `n_bins`: The number of bins to be used in the histogram (in each dimension).
- `save_dir`: The directory where results and time convergence data will be saved.
- `chunk_size`: Number of simulation steps to be run in each chunk. Default is 10000000.
- `checkpoint`: A boolean flag indicating whether to save checkpoints. Default is false.
- `q0`: The initial configuration for the simulation. Default is nothing, which generates a random configuration.
- `time_transform`: A boolean flag indicating whether to apply time transformation. Default is false.

Returns:
- The function saves the results of each experiment in the specified `save_dir` and also saves the time convergence data in a file named "time.json".
"""
function master_2D_experiment(integrators, num_repeats, V, D, T, sigma, stepsizes, xmin, ymin, xmax, ymax, n_bins, save_dir; chunk_size=10000000, x0=nothing, noise_integrator=nothing, n=nothing, max_retries=3)
    to = TimerOutput()

    @info "Computing Expected Probabilities"
    probabilities, x_bins, y_bins, n_bins = compute_2D_invariant_distribution(V, sigma, xmin, ymin, xmax, ymax, n_bins)

    create_directory_if_not_exists(save_dir)
    progress = Dict(string(nameof(integrator)) => Dict(string(dt) => 0 for dt in stepsizes) for integrator in integrators)
    open("$(save_dir)/progress.json", "w") do f
        JSON.print(f, progress, 4)
    end
    
    @info "Running Experiments"
    for integrator in integrators
        @info "Running $(string(nameof(integrator))) experiment"
        # reset the random seed for reproducibility
        Random.seed!(1)
        #@timeit to "Exp$(string(nameof(integrator)))" begin 
        _ = run_2D_experiment(integrator, num_repeats, V, D, T, sigma, stepsizes, probabilities, x_bins, y_bins, save_dir, chunk_size=chunk_size, x0=x0, noise_integrator=noise_integrator, n=nothing, max_retries=max_retries)
        #end
    end

    # save the time convergence_data
    open("$(save_dir)/time.json", "w") do io
        JSON.print(io, TimerOutputs.todict(to), 4)
    end
end

end # module