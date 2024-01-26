#! /bin/sh

../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/quadWell2D_anisotropicI_EG_MT2.jl &
../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/quadWell2D_anisotropicI_EG_W2Ito.jl &
../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/quadWell2D_anisotropicI_EM.jl &
../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/quadWell2D_anisotropicI_HLM.jl &

../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/quadWell2D_anisotropicII_EG_MT2.jl &
../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/quadWell2D_anisotropicII_EG_W2Ito.jl &
../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/quadWell2D_anisotropicII_EM.jl &
../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/quadWell2D_anisotropicII_HLM.jl &

../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/quadWell2D_anisotropicIII_EG_MT2.jl &
../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/quadWell2D_anisotropicIII_EG_W2Ito.jl &
../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/quadWell2D_anisotropicIII_EM.jl &
../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/quadWell2D_anisotropicIII_HLM.jl &

../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/doubleWellChannel2D_isotropic_EG_MT2.jl &
../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/doubleWellChannel2D_isotropic_EG_W2Ito.jl &
../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/doubleWellChannel2D_isotropic_EM.jl &
../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/doubleWellChannel2D_isotropic_HLM.jl &

../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/doubleWellChannel2D_anisotropicI_EG_MT2.jl &
../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/doubleWellChannel2D_anisotropicI_EG_W2Ito.jl &
../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/doubleWellChannel2D_anisotropicI_EM.jl &
../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/doubleWellChannel2D_anisotropicI_HLM.jl &

../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/doubleWellChannel2D_anisotropicII_EG_MT2.jl &
../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/doubleWellChannel2D_anisotropicII_EG_W2Ito.jl &
../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/doubleWellChannel2D_anisotropicII_EM.jl &
../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/doubleWellChannel2D_anisotropicII_HLM.jl &

../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/doubleWellChannel2D_anisotropicIII_EG_MT2.jl &
../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/doubleWellChannel2D_anisotropicIII_EG_W2Ito.jl &
../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/doubleWellChannel2D_anisotropicIII_EM.jl &
../julia-1.9.4/bin/julia --threads 12 ./cluster_jobs/doubleWellChannel2D_anisotropicIII_HLM.jl 