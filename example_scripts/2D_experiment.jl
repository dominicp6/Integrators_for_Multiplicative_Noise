include("../general_utils/integrators.jl")
include("../general_utils/integrator_utils.jl")
include("../general_utils/potentials.jl")
include("../general_utils/diffusion_tensors.jl")
include("../general_utils/probability_utils.jl")
include("../experiment_utils/experiments2D.jl")
using LinearAlgebra
import .Integrators: euler_maruyama2D, leimkuhler_matthews2D, hummer_leimkuhler_matthews2D, stochastic_heun2D    
import .IntegratorUtils: MT2_1D, W2Ito1_1D, W2Ito1_2D, MT2_2D
import .Potentials: bowl2D, quadrupleWell2D, moroCardin2D, muller_brown, softQuadrupleWell2D
import .DiffusionTensors: Dconst2D, Dquadratic2D, DmoroCardin, Doseen
import .ProbabilityUtils:  compute_2D_invariant_distribution
import .Experiments2D: master_2D_experiment

"""
This script performs two-dimensional, variable-diffusion Brownian dynamics experiments and 
constructs plots of the weak convergence to the invariant measure for a range of specified step sizes.

Time rescalings are supported.
"""

# Name
exp_name = "2D_test"
master_dir = "outputs" # Directory to save results in
T = 10000               # length of simulation
sigma = 1              # value of kT (noise amplitude scaling)
num_repeats = 6

# The step sizes to use (to use a single step size, set stepsizes = [stepsize])
num_step_sizes = 10
integrators = [euler_maruyama2D, leimkuhler_matthews2D, hummer_leimkuhler_matthews2D]

# The range of stepsizes to use (defualt 10 step sizes in the range 10^{-1.5} to 10^{-0.5})
stepsizes = 10 .^ range(-1.5,stop=-0.5,length=num_step_sizes)

# Histogram parameters for binning
xmin = -3
xmax = 3
ymin = -3
ymax = 3
n_bins = 30   # number of bins in each dimension

# The potential and diffusion coefficient to use
potential = softQuadrupleWell2D
diffusion = DmoroCardin

# Whether to save checkpoints
checkpoint = false

# Do not modify below this line ----------------------------------------------
save_dir = "$(master_dir)/$(exp_name)"

max_retries = 5

# Run the experiments
@info "Running: $(exp_name)"
master_2D_experiment(integrators, num_repeats, potential, diffusion, T, sigma, stepsizes, xmin, ymin, xmax, ymax, n_bins, save_dir, chunk_size=50000, max_retries=max_retries, noise_integrator=MT2_1D)
   