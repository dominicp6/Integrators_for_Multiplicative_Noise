module IntegratorUtils
using StatsBase
export MT2_1D, W2Ito1_1D, W2Ito1_2D, MT2_2D, EM_noise_1D

function EM_noise_1D(x0, dt, D, sigma)
    # Number of substeps
    n = 20
    for i in 1:n
        x0 += sigma * D(x0) * sqrt(dt/n) * randn()
    end

    return x0 
end

function MT2_1D(x0, dt, D, Rₖ)

    χ = rand([-1, 1])
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

function W2Ito1_2D(x0, dt, D, D_1, D_2, Rₖ)

    χ1 = rand([-1, 1])
    χ2 = rand([-1, 1])
    J11 = χ1 * (Rₖ[1]^2  - 1) / 2
    J22 = χ1 * (Rₖ[2]^2  - 1) / 2
    J12 = Rₖ[2] * (1 - χ2) / 2
    J21 = Rₖ[1] * (1 + χ2) / 2
    sqrt_dt = sqrt(dt)

    D_1_x0 = D_1(x0...)
    D_2_x0 = D_2(x0...)

    K1_1 = x0 + sqrt_dt/2 * D_1_x0 * χ1 + sqrt_dt * D_2_x0 * J12
    K1_2 = x0 + sqrt_dt/2 * D_2_x0 * χ1 + sqrt_dt * D_1_x0 * J21
    K2_1 = x0 - sqrt_dt/2 * D_1_x0 * χ1 
    K2_2 = x0 - sqrt_dt/2 * D_2_x0 * χ1

    D_1_K1_1 = D_1(K1_1...)
    D_1_K2_1 = D_1(K2_1...)
    D_2_K1_2 = D_2(K1_2...)
    D_2_K2_2 = D_2(K2_2...)

    result = [0, 0]
    result += sqrt_dt * (- D_1_x0 + D_1_K1_1 + D_1_K2_1) * Rₖ[1] 
    result += sqrt_dt * (- D_2_x0 + D_2_K1_2 + D_2_K2_2) * Rₖ[2]
    result += 2 * sqrt_dt * (D_1_x0 - D_1_K2_1) * J11
    result += 2 * sqrt_dt * (D_2_x0 - D_2_K2_2) * J22

    return result
end


function MT2_2D(x0, dt, D, D_1, D_2, Rₖ)

    χ1 = rand([-1, 1])
    χ2 = rand([-1, 1])
    J11 = dt * (Rₖ[1]^2 - 1) / 2
    J22 = dt * (Rₖ[2]^2 - 1) / 2
    J12 = dt * (Rₖ[1] * Rₖ[2] + χ2) / 2
    J21 = dt * (Rₖ[1] * Rₖ[2] - χ1) / 2
    
    D_x0 = D(x0...)

    result = [0, 0]
    arg1 = x0 + D_x0 * [J11, J12]
    arg2 = x0 - D_x0 * [J11, J12]
    result += 0.5 * (D_1(arg1...) - D_1(arg2...))
    arg1 = x0 + D_x0 * [J21, J22]
    arg2 = x0 - D_x0 * [J21, J22]
    result += 0.5 * (D_2(arg1...) - D_2(arg2...))
    arg1 = x0 + sqrt(dt/2) * D_x0 * [χ1, χ2]
    arg2 = x0 - sqrt(dt/2) * D_x0 * [χ1, χ2]
    result += sqrt(dt)/2 * (D(arg1...) + D(arg2...)) * Rₖ

    return result
end

end # module

