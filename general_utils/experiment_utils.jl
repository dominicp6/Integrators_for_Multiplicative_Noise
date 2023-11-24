module ExperimentUtils
include("../general_utils/misc_utils.jl")
include("../general_utils/transform_utils.jl")
using FHist, JSON
import .MiscUtils: init_x0, create_directory_if_not_exists
import .TransformUtils: increment_g_counts, increment_I_counts
export make_experiment_folders, run_chunk

"""
Creates necessary directories and save experiment parameters for the 1D experiment.
"""
function make_experiment_folders(save_dir, integrator, stepsizes, num_repeats, V, D, sigma, x_bins, chunk_size, time_transform, space_transform; T=nothing)
    # Make master directory
    create_directory_if_not_exists(save_dir)

    if !isfile("$(save_dir)/info.json")
        @info "Saving metadata"
        metadata = Dict("integrator" => string(nameof(integrator)),
                        "num_repeats" => num_repeats,
                        "V" => string(nameof(V)),
                        "D" => string(nameof(D)),
                        "T" => T,
                        "sigma" => sigma,
                        "stepsizes" => stepsizes,
                        "x_bins" => x_bins,
                        "chunk_size" => chunk_size,
                        "time_transform" => string(time_transform),
                        "space_transform" => string(space_transform))
        open("$(save_dir)/info.json", "w") do f
            JSON.print(f, metadata, 4)
        end
    end
end

"""
Creates necessary directories and save experiment parameters for the 1D experiment.
"""
function make_experiment_folders(save_dir, integrator, stepsizes, num_repeats, V, D, sigma, x_bins, chunk_size, T=nothing, target_uncertainty=nothing)
    # Make master directory
    create_directory_if_not_exists(save_dir)

    if !isfile("$(save_dir)/info.json")
        @info "Saving metadata"
        metadata = Dict("integrator" => string(nameof(integrator)),
                        "num_repeats" => num_repeats,
                        "V" => string(nameof(V)),
                        "D" => string(nameof(D)),
                        "T" => T,
                        "target_uncertainty" => target_uncertainty,
                        "sigma" => sigma,
                        "stepsizes" => stepsizes,
                        "x_bins" => x_bins,
                        "chunk_size" => chunk_size,
        )
        open("$(save_dir)/info.json", "w") do f
            JSON.print(f, metadata, 4)
        end
    end
end

"""
The `run_chunk` function runs a chunk of the 1D finite-time simulation using the specified integrator and parameters.
It performs the simulation for `steps_to_run` time steps and updates the histogram with the trajectory data.

Note: The function is typically called within the context of the main simulation loop, and its results are used for further analysis.
"""
function run_chunk(integrator, q0, Vprime, D, Dprime, sigma::Number, dt::Number, steps_to_run::Integer, hist, bin_boundaries, chunk_number::Integer, time_transform::Bool, space_transform:: Bool, ΣgI::Union{Vector, Nothing}, Σg::Union{Float64, Nothing}, ΣI::Union{Vector, Nothing}, original_D, x_of_y)

    # Run a chunk of the simulation
    q_chunk, _ = integrator(q0, Vprime, D, Dprime, sigma, steps_to_run, dt)

    # Get the last position of the chunk
    q0 = copy(q_chunk[end])

    # [For time-transformed integrators] Increment g counts (see paper for details)
    if time_transform
        ΣgI, Σg =  increment_g_counts(q_chunk, original_D, bin_boundaries, ΣgI, Σg)
    end

    # [For space-transformed integrators] Increment I counts (see paper for details)
    if space_transform
        ΣI = increment_I_counts(q_chunk, x_of_y, bin_boundaries, ΣI)
    end

    # Update the number of steps left to run
    hist += Hist1D(q_chunk, bin_boundaries)
    chunk_number += 1

    return q0, hist, chunk_number, ΣgI, Σg, ΣI
end

end