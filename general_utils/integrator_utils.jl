module IntegratorUtils
using StatsBase
export MT2_1D, W2Ito1_1D 

function MT2_1D(x0, dt, D, Rₖ)

    χ = rand([-1, 1])

    #ξ = sample([sqrt(3), -sqrt(3), 0], Weights([1/6, 1/6, 2/3]))
    J = dt * (Rₖ^2 - 1) / 2
    sqrt_dt_2 = sqrt(dt/2)

    D_x0 = D(x0)
    
    arg1 = x0 + D_x0 * J
    arg2 = x0 - D_x0 * J
    arg3 = x0 + sqrt_dt_2 * D_x0 * χ
    arg4 = x0 - sqrt_dt_2 * D_x0 * χ

    result = 0.5 * (D(arg1) - D(arg2)) + sqrt(dt)/2 * (D(arg3) + D(arg4)) * Rₖ

    return result

end

function W2Ito1_1D(x0, dt, D, Rₖ)

    χ1 = rand([-1, 1])
    #ξ = sample([sqrt(3), -sqrt(3), 0], Weights([1/6, 1/6, 2/3]))
    J = χ1 * (Rₖ^2  - 1) / 2
    sqrt_dt = sqrt(dt)

    D_x0 = D(x0)
    K1 = x0 + sqrt_dt/2 * D_x0 * χ1
    K2 = x0 - sqrt_dt * D_x0 * χ1 / 2

    D_K1 = D(K1)
    D_K2 = D(K2)

    result = sqrt_dt * (- D_x0 + D_K1 + D_K2) * Rₖ + 2 * sqrt_dt * (D_x0 - D_K2) * J

    return result

end

end # module

