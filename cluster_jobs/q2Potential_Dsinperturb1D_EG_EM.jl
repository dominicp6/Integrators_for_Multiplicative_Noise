include("../general_utils/integrators.jl")
include("../general_utils/integrator_utils.jl")
include("../general_utils/potentials.jl")
include("../general_utils/diffusion_tensors.jl")
include("../general_utils/probability_utils.jl")
include("../experiment_utils/experiments1D.jl")
include("../experiment_utils/experiments2D.jl")
include("../general_utils/calculus.jl")

import .Integrators: eugen_gilles1D, euler_maruyama1D, hummer_leimkuhler_matthews1D, leimkuhler_matthews1D, leimkuhler_matthews_markovian1D, limit_method_with_variable_diffusion1D, euler_maruyama2D, eugen_gilles2D, limit_method_with_variable_diffusion_RK6_1D, eugen_gilles_withEM1D
import .Potentials: softWell1D, q4Potential, softQuadrupleWell2D, q4Potential2D, q2Potential
import .DiffusionTensors: Dconst1D, Dabs1D, Dquadratic1D, Dcosperturb1D, Dconst2D, Dcosperturb2D, Dsinperturb1D
import .ProbabilityUtils: compute_1D_invariant_distribution
import .Experiments: master_1D_experiment, run_1D_experiment_until_given_error
import .Experiments2D: master_2D_experiment
import .IntegratorUtils: MT2_1D, W2Ito1_1D, W2Ito1_2D, MT2_2D
import .Calculus: differentiate1D

master_dir = "./cluster_results"

# LENGTH OF SIMULATION 
T = 50000000
sigma = 1
num_repeats = 12

xmin = -5
xmax = 5
n_bins = 30

bin_boundaries = range(xmin, stop=xmax, length=n_bins+1)

chunk_size = 100000;

# POTENTIAL AND DIFFUSION
potential = q2Potential
diffusion = Dsinperturb1D

# INTEGRATOR
noise_integrator = nothing
n = nothing
integrators = [eugen_gilles_withEM1D]

# EXPERIMENT NAME
exp_name = "q2Potential_Dsinperturb1D_EG_EM"

# STEPSIZES
number_of_stepsizes = 21
stepsizes = 10 .^ range(-2.0, 0.0, length=number_of_stepsizes)

save_dir = "$(master_dir)/$(exp_name)"

master_1D_experiment(integrators, num_repeats, potential, diffusion, T, sigma, stepsizes, bin_boundaries, save_dir; chunk_size=chunk_size, x0=nothing, noise_integrator=noise_integrator, n=n)

