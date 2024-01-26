#! /bin/sh

../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/q2Potential_Dconst1D_EG_MT2.jl &
../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/q2Potential_Dconst1D_EG_W2Ito1.jl &
../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/q2Potential_Dcosperturb1D_EG_MT2.jl &
../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/q2Potential_Dcosperturb1D_EG_W2Ito1.jl 

../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/q2Potential_Dsinperturb1D_EG_MT2.jl &
../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/q2Potential_Dsinperturb1D_EG_W2Ito1.jl &
../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/q4Potential_Dconst1D_EG_MT2.jl &
../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/q4Potential_Dconst1D_EG_W2Ito1.jl 

../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/q4Potential_Dcosperturb1D_EG_MT2.jl &
../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/q4Potential_Dcosperturb1D_EG_W2Ito1.jl &
../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/q4Potential_Dsinperturb1D_EG_MT2.jl &
../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/q4Potential_Dsinperturb1D_EG_W2Ito1.jl 

../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/softWell_Dconst1D_EG_MT2.jl &
../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/softWell_Dconst1D_EG_W2Ito1.jl &
../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/softWell_Dabs1D_EG_MT2.jl &
../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/softWell_Dabs1D_EG_W2Ito1.jl 