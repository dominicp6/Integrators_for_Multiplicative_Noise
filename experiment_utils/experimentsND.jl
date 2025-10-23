module ExperimentsND
include("../general_utils/calculus.jl")
include("../general_utils/plotting_utils.jl")
include("../general_utils/misc_utils.jl")
include("../general_utils/experiment_utils.jl")
using JLD2, Statistics, .Threads, ProgressBars, JSON, Random, StatsBase, TimerOutputs, Plots
import .Calculus: differentiateND, symbolic_matrix_divergenceND, gradientND
import .PlottingUtils: save_and_plot
import .MiscUtils: init_x0, create_directory_if_not_exists
import .ExperimentUtils: make_experiment_folders2
export run_ND_experiment

function run_ND_experiment(integrator, num_repeats, dim, V, Vprime, D, D_column, div_DDT, F, Da, T, sigma, stepsizes, save_dir, observable, expected_observable; chunk_size=100000, x0=nothing, noise_integrator=nothing)
    
    # Make master directory
    make_experiment_folders2(save_dir, integrator, stepsizes, num_repeats, V, D, sigma, chunk_size, T)

    create_directory_if_not_exists(save_dir)
    progress = Dict(string(nameof(integrator)) => Dict(string(dt) => 0 for dt in stepsizes))
    open("$(save_dir)/progress.json", "w") do f
        JSON.print(f, progress, 4)
    end

    # Initialise empty data array
    observable_errors = zeros(length(stepsizes), num_repeats)

    x_chunk_value = nothing
    x_chunk_returned = false

    Threads.@threads for repeat in ProgressBar(1:num_repeats)
        Random.seed!(repeat) 

        # Loop backwards, starting with the largest step size
        for (stepsize_reverse_idx, dt) in enumerate(stepsizes[end:-1:1])
            x_start = init_x0(x0, dim=dim)
            stepsize_idx = length(stepsizes) - stepsize_reverse_idx + 1
            steps_remaining = floor(Int, T / dt)                
            total_samples = Int(steps_remaining)                               
            obs = 0.0          

            steps_done = 0
            while steps_remaining > 0
                # Run steps in chunks to minimise memory footprint
                steps_to_run = convert(Int, min(steps_remaining, chunk_size))

                println("Steps to run", steps_to_run)
                # Run a chunk of the simulation
                x_chunk = integrator(x_start, Vprime, D, div_DDT, D_column, F, Da, sigma, steps_to_run, dt, noise_integrator)
                
                if !x_chunk_returned
                    x_chunk_value = x_chunk
                    x_chunk_returned = true
                end
                
                x_start = copy(x_chunk[:, end])
                obs = ((total_samples - steps_remaining) * obs + sum(observable(x_chunk[:, i]) for i in 1:size(x_chunk, 2))) / (total_samples - steps_remaining + steps_to_run)

                steps_remaining -= steps_to_run
                steps_done += steps_to_run

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
            
            observable_errors[stepsize_idx, repeat] = abs(obs - expected_observable)
                
            # Save progress of results
            delay = repeat / num_repeats
            sleep(delay)
            errors_path = "$(save_dir)/partial_results.txt"
            open(errors_path, "a") do io
                message = string(string(nameof(integrator)), ", ", dt, ", ", repeat, ", ", observable_errors[stepsize_idx, repeat], "\n")
                write(io, message)
            end
        end
    end

    return x_chunk_value

    # Save the error data and plot
    save_and_plot(integrator, observable_errors, stepsizes, save_dir, suffix="_observable", error_in_mean=true)

    @info "Mean L1 errors: $(mean(observable_errors, dims=2))"
    @info "Standard deviation of L1 errors: $(std(observable_errors, dims=2))"

    return observable_errors
end

end # module