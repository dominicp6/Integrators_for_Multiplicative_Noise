include("../general_utils/integrators.jl")
include("../general_utils/potentials.jl")
include("../general_utils/diffusion_tensors.jl")
include("../general_utils/probability_utils.jl")
include("../experiment_utils/experiments1D.jl")
import .Integrators: euler_maruyama1D, naive_leimkuhler_matthews1D, hummer_leimkuhler_matthews1D, milstein_method1D, stochastic_heun1D, limit_method_with_variable_diffusion1D  
import .Potentials: doubleWell1D, LM2013, localWell1D, transformedLocalWell1D, transformedLM2013, transformed2LM2013, softWell1D, transformedSoftWell1D, transformed2SoftWell1D
import .DiffusionTensors: Dconst1D, Dabs1D, Dquadratic1D
import .ProbabilityUtils: compute_1D_invariant_distribution
import .Experiments: master_1D_experiment, run_1D_experiment_until_given_error

"""
This script performs one-dimensional, variable-diffusion Brownian dynamics experiments and 
constructs plots of the weak convergence to the invariant measure for a range of specified step sizes.

Time rescalings and lamperti transforms are supported.
"""

exp_name = "1D_experiment_test" # Name
master_dir = "outputs"          # Directory to save results in
T = 100000                      # length of simulation
sigma = 1                         # value of kT (noise amplitude scaling)
num_repeats = 10     

# The step sizes to use (to use a single step size, set stepsizes = [stepsize])
num_step_sizes = 10

# The range of stepsizes to use (defualt 10 step sizes in the range 10^{-2} to 10^{-1})
stepsizes = 10 .^ range(-2,stop=-1,length=num_step_sizes)

# The integrators to use (comma separated list)
integrators = [milstein_method1D, naive_leimkuhler_matthews1D, euler_maruyama1D]

# The histogram parameters for binning
xmin = -5
xmax = 5
n_bins = 30

# The potential and diffusion coefficents to use
potential = softWell1D
diffusion = Dabs1D

# Information on the transformation
time_transform = false
space_transform = true
x_of_y = y -> (y/4) * (abs(y) + 4)  # This spatial transformation is specific to the Dabs1D diffusion coefficient (see paper for details)
                                    # If you want to use a different diffusion coefficient, you will need compute the appropriate mapping from the definition of the Lamperti transform

# Whether to save checkpoints
checkpoint = false

# Do not modify below this line ----------------------------------------------
bin_boundaries = range(xmin, xmax, length=n_bins+1)
save_dir = "$(master_dir)/$(exp_name)"

# Run the experiments
@info "Running: $(exp_name)"
master_1D_experiment(integrators, num_repeats, potential, diffusion, T, sigma, stepsizes, bin_boundaries, save_dir; chunk_size=1000000000, checkpoint=checkpoint, q0=nothing, time_transform=time_transform, space_transform=space_transform, x_of_y=x_of_y)
