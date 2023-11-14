include("./general_utils/integrators.jl")
include("./general_utils/potentials.jl")
include("./general_utils/diffusion_tensors.jl")
include("./general_utils/probability_utils.jl")
include("./experiment_utils/experiments2D.jl")
include("./general_utils/integrator_utils.jl")
include("./general_utils/calculus.jl")

import .Integrators: euler_maruyama2D, eugen_gilles2D, hummer_leimkuhler_matthews2D
import .Potentials: softQuadrupleWell2D
import .DiffusionTensors: DmoroCardin
import .ProbabilityUtils: compute_1D_invariant_distribution
import .Experiments2D: master_2D_experiment
import .IntegratorUtils: MT2_2D, W2Ito1_2D
import .Calculus: differentiate1D

using LinearAlgebra

# Global params
chunk_size = 500000;
num_step_sizes = 10
stepsizes = 10 .^ range(-1.3,stop=-0.3,length=num_step_sizes);
# Histogram parameters for binning
xmin = -3
xmax = 3
ymin = -3
ymax = 3
n_bins = 30   # number of bins in each dimensio
R = Matrix{Float64}(I, 2, 2)

T = 5000000                           # length of simulation
sigma = 1                            
num_repeats = 10      

# The potential and diffusion coefficents to use
potential = softQuadrupleWell2D
diffusion = DmoroCardin

master_dir = "./simulation_results"     # Directory to save results in

# Experiment 1 ---------------------------------------------------------------
exp_name = "quartic_cos_noise_2D_MT2_5M"     # Name
save_dir = "$(master_dir)/$(exp_name)"
integrators = [eugen_gilles2D] 
noise_integrator = MT2_2D
n = nothing

@info "Running experiment 1"
master_2D_experiment(integrators, num_repeats, potential, diffusion, T, sigma, stepsizes, xmin, ymin, xmax, ymax, n_bins, save_dir; chunk_size=chunk_size, x0=nothing, noise_integrator=noise_integrator, n=n)

# # Experiment 2 ---------------------------------------------------------------
exp_name = "quartic_cos_noise_2D_W2Ito1_5M"     # Name
save_dir = "$(master_dir)/$(exp_name)"
integrators = [eugen_gilles2D]
noise_integrator = W2Ito1_2D
n = nothing

@info "Running experiment 2"
master_2D_experiment(integrators, num_repeats, potential, diffusion, T, sigma, stepsizes, xmin, ymin, xmax, ymax, n_bins, save_dir; chunk_size=chunk_size, x0=nothing, noise_integrator=noise_integrator, n=n)

# # Experiment 3 ---------------------------------------------------------------
exp_name = "quartic_cos_noise_2D_EM_5M"     # Name
save_dir = "$(master_dir)/$(exp_name)"
integrators = [euler_maruyama2D]
n = nothing

@info "Running experiment 3"
master_2D_experiment(integrators, num_repeats, potential, diffusion, T, sigma, stepsizes, xmin, ymin, xmax, ymax, n_bins, save_dir; chunk_size=chunk_size, x0=nothing, noise_integrator=noise_integrator, n=n)

# # Experiment 4 ---------------------------------------------------------------
exp_name = "quartic_cos_noise_2D_HLM_5M"     # Name
save_dir = "$(master_dir)/$(exp_name)"
integrators = [hummer_leimkuhler_matthews2D]
n = nothing

@info "Running experiment 4"
master_2D_experiment(integrators, num_repeats, potential, diffusion, T, sigma, stepsizes, xmin, ymin, xmax, ymax, n_bins, save_dir; chunk_size=chunk_size, x0=nothing, noise_integrator=noise_integrator, n=n)

# # Experiment 5 ---------------------------------------------------------------
# exp_name = "quartic_cos_noise_LMVD_n=1_1M"     # Name
# save_dir = "$(master_dir)/$(exp_name)"
# integrators = [limit_method_with_variable_diffusion1D]
# n = 1

# @info "Running experiment 5"
# master_1D_experiment(integrators, num_repeats, potential, diffusion, T, sigma, stepsizes, bin_boundaries, save_dir; chunk_size=chunk_size, x0=nothing, noise_integrator=nothing, n=n)

# Experiment 6 ---------------------------------------------------------------
# exp_name = "quartic_cos_noise_euler_maruyama_1M"     # Name
# save_dir = "$(master_dir)/$(exp_name)"
# integrators = [euler_maruyama1D]
# n = nothing

# num_step_sizes = 8
# stepsizes = 10 .^ range(-2.0,stop=-1.2,length=num_step_sizes);

# @info "Running experiment 6"
# master_1D_experiment(integrators, num_repeats, potential, diffusion, T, sigma, stepsizes, bin_boundaries, save_dir; chunk_size=chunk_size, x0=nothing, noise_integrator=nothing, n=n)

# exp_name = "quartic_cos_noise_hummer_leimkuhler_matthews_1M"     # Name
# save_dir = "$(master_dir)/$(exp_name)"
# integrators = [hummer_leimkuhler_matthews1D]
# n = nothing

# num_step_sizes = 10
# stepsizes = 10 .^ range(-2.0,stop=-1.0,length=num_step_sizes);

# @info "Running experiment 7"
# master_1D_experiment(integrators, num_repeats, potential, diffusion, T, sigma, stepsizes, bin_boundaries, save_dir; chunk_size=chunk_size, x0=nothing, noise_integrator=nothing, n=n)

# T = 10000000                           # length of simulation

# Experiment 1 ---------------------------------------------------------------
# exp_name = "quartic_cos_noise_MT2_10M"     # Name
# save_dir = "$(master_dir)/$(exp_name)"
# integrators = [eugen_gilles1D] 
# noise_integrator = MT2_1D
# n = nothing

# @info "Running experiment 7"
# master_1D_experiment(integrators, num_repeats, potential, diffusion, T, sigma, stepsizes, bin_boundaries, save_dir; chunk_size=chunk_size, x0=nothing, noise_integrator=noise_integrator, n=n)

# # Experiment 2 ---------------------------------------------------------------
# exp_name = "quartic_cos_noise_W2Ito1_10M"     # Name
# save_dir = "$(master_dir)/$(exp_name)"
# integrators = [eugen_gilles1D]
# noise_integrator = W2Ito1_1D
# n = nothing

# @info "Running experiment 8"
# master_1D_experiment(integrators, num_repeats, potential, diffusion, T, sigma, stepsizes, bin_boundaries, save_dir; chunk_size=chunk_size, x0=nothing, noise_integrator=noise_integrator, n=n)

# Experiment 3 ---------------------------------------------------------------
# exp_name = "quartic_cos_noise_LMVD_n=5_10M"     # Name
# save_dir = "$(master_dir)/$(exp_name)"
# integrators = [limit_method_with_variable_diffusion1D]
# n = 5

# @info "Running experiment 9"
# master_1D_experiment(integrators, num_repeats, potential, diffusion, T, sigma, stepsizes, bin_boundaries, save_dir; chunk_size=chunk_size, x0=nothing, noise_integrator=nothing, n=n)

# # Experiment 4 ---------------------------------------------------------------
# exp_name = "quartic_cos_noise_LMVD_n=2_10M"     # Name
# save_dir = "$(master_dir)/$(exp_name)"
# integrators = [limit_method_with_variable_diffusion1D]
# n = 2

# @info "Running experiment 10"
# master_1D_experiment(integrators, num_repeats, potential, diffusion, T, sigma, stepsizes, bin_boundaries, save_dir; chunk_size=chunk_size, x0=nothing, noise_integrator=nothing, n=n)

# Experiment 5 ---------------------------------------------------------------
# exp_name = "quartic_cos_noise_LMVD_n=1_10M"     # Name
# save_dir = "$(master_dir)/$(exp_name)"
# integrators = [limit_method_with_variable_diffusion1D]
# n = 1

# @info "Running experiment 11"
# master_1D_experiment(integrators, num_repeats, potential, diffusion, T, sigma, stepsizes, bin_boundaries, save_dir; chunk_size=chunk_size, x0=nothing, noise_integrator=nothing, n=n)

# # Experiment 6 ---------------------------------------------------------------
# exp_name = "quartic_cos_noise_euler_maruyama_10M"     # Name
# save_dir = "$(master_dir)/$(exp_name)"
# integrators = [euler_maruyama1D]
# n = nothing

# num_step_sizes = 8
# stepsizes = 10 .^ range(-2.0,stop=-1.2,length=num_step_sizes);

# @info "Running experiment 12"
# master_1D_experiment(integrators, num_repeats, potential, diffusion, T, sigma, stepsizes, bin_boundaries, save_dir; chunk_size=chunk_size, x0=nothing, noise_integrator=nothing, n=n)

# exp_name = "quartic_cos_noise_hummer_leimkuhler_matthews_10M"     # Name
# save_dir = "$(master_dir)/$(exp_name)"
# integrators = [hummer_leimkuhler_matthews1D]
# n = nothing

# num_step_sizes = 10
# stepsizes = 10 .^ range(-2.0,stop=-1.0,length=num_step_sizes);

# @info "Running experiment 13"
# master_1D_experiment(integrators, num_repeats, potential, diffusion, T, sigma, stepsizes, bin_boundaries, save_dir; chunk_size=1000000, x0=nothing, noise_integrator=nothing, n=n)


# T = 50000000                           # length of simulation

# Experiment 1 ---------------------------------------------------------------
# exp_name = "quartic_cos_noise_MT2_50M"     # Name
# save_dir = "$(master_dir)/$(exp_name)"
# integrators = [eugen_gilles1D] 
# noise_integrator = MT2_1D
# n = nothing

# @info "Running experiment 13"
# master_1D_experiment(integrators, num_repeats, potential, diffusion, T, sigma, stepsizes, bin_boundaries, save_dir; chunk_size=chunk_size, x0=nothing, noise_integrator=noise_integrator, n=n)

# # Experiment 2 ---------------------------------------------------------------
# exp_name = "quartic_cos_noise_W2Ito1_50M"     # Name
# save_dir = "$(master_dir)/$(exp_name)"
# integrators = [eugen_gilles1D]
# noise_integrator = W2Ito1_1D
# n = nothing

# @info "Running experiment 14"
# master_1D_experiment(integrators, num_repeats, potential, diffusion, T, sigma, stepsizes, bin_boundaries, save_dir; chunk_size=chunk_size, x0=nothing, noise_integrator=noise_integrator, n=n)

# Experiment 3 ---------------------------------------------------------------
# exp_name = "quartic_cos_noise_LMVD_n=5_50M"     # Name
# save_dir = "$(master_dir)/$(exp_name)"
# integrators = [limit_method_with_variable_diffusion1D]
# n = 5

# @info "Running experiment 15"
# master_1D_experiment(integrators, num_repeats, potential, diffusion, T, sigma, stepsizes, bin_boundaries, save_dir; chunk_size=chunk_size, x0=nothing, noise_integrator=nothing, n=n)

# # Experiment 4 ---------------------------------------------------------------
# exp_name = "quartic_cos_noise_LMVD_n=2_50M"     # Name
# save_dir = "$(master_dir)/$(exp_name)"
# integrators = [limit_method_with_variable_diffusion1D]
# n = 2

# @info "Running experiment 16"
# master_1D_experiment(integrators, num_repeats, potential, diffusion, T, sigma, stepsizes, bin_boundaries, save_dir; chunk_size=chunk_size, x0=nothing, noise_integrator=nothing, n=n)

# Experiment 5 ---------------------------------------------------------------
# exp_name = "quartic_cos_noise_LMVD_n=1_50M"     # Name
# save_dir = "$(master_dir)/$(exp_name)"
# integrators = [limit_method_with_variable_diffusion1D]
# n = 1

# @info "Running experiment 17"
# master_1D_experiment(integrators, num_repeats, potential, diffusion, T, sigma, stepsizes, bin_boundaries, save_dir; chunk_size=chunk_size, x0=nothing, noise_integrator=nothing, n=n)

# # Experiment 6 ---------------------------------------------------------------
# exp_name = "quartic_cos_noise_euler_maruyama_50M"     # Name
# save_dir = "$(master_dir)/$(exp_name)"
# integrators = [euler_maruyama1D]
# n = nothing

# num_step_sizes = 8
# stepsizes = 10 .^ range(-2.0,stop=-1.2,length=num_step_sizes);

# @info "Running experiment 18"
# master_1D_experiment(integrators, num_repeats, potential, diffusion, T, sigma, stepsizes, bin_boundaries, save_dir; chunk_size=chunk_size, x0=nothing, noise_integrator=nothing, n=n)

# exp_name = "quartic_cos_noise_hummer_leimkuhler_matthews_50M"     # Name
# save_dir = "$(master_dir)/$(exp_name)"
# integrators = [hummer_leimkuhler_matthews1D]
# n = nothing

# num_step_sizes = 10
# stepsizes = 10 .^ range(-2.0,stop=-1.0,length=num_step_sizes);

# @info "Running experiment 19"
# master_1D_experiment(integrators, num_repeats, potential, diffusion, T, sigma, stepsizes, bin_boundaries, save_dir; chunk_size=1000000, x0=nothing, noise_integrator=nothing, n=n)