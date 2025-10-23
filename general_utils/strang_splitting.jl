function strang_splitting1D(x0, Vprime, D, D2prime, sigma::Number, m::Integer, dt::Number, Rₖ=nothing, noise_integrator=nothing, n=nothing)

    # set up
    t = 0.0
    x = copy(x0)
    x_traj = zeros(m)

    # simulate
    for i in 1:m
        drift_term = x -> -(D(x)^2) * Vprime(x) + sigma^2 * D2prime(x) / 2

        # Perform 1 step of RK4 integration for hat_xₖ₊₁
        k1 = (dt / 2) * drift_term(x)
        k2 = (dt / 2) * drift_term(x + 0.5 * k1)
        k3 = (dt / 2) * drift_term(x + 0.5 * k2)
        k4 = (dt / 2) * drift_term(x + k3) 
            
        # Update state using weighted average of intermediate values
        x += (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) 

        Rₖ = randn()

        x += noise_integrator(x, dt, D, Rₖ)

        # Perform 1 step of RK4 integration for hat_xₖ₊₁
        k1 = (dt / 2) * drift_term(x)
        k2 = (dt / 2) * drift_term(x + 0.5 * k1)
        k3 = (dt / 2) * drift_term(x + 0.5 * k2)
        k4 = (dt / 2) * drift_term(x + k3) 
            
        # Update state using weighted average of intermediate values
        x += (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) 

        x_traj[i] = x

        # update the time
        t += dt
    end

    return x_traj, nothing
end


function strang_splitting2D(x0, Vprime, D, div_DDT, sigma::Number, m::Integer, dt::Number, Rₖ=nothing, noise_integrator=nothing, n=nothing)
    D² = (x, y) -> D(x, y)^2
    F = (x, y) -> D²(x, y) * (-Vprime(x, y)) + sigma^2 * div_DDT(x, y) /2

    # Define first and second columns of the matrix function D
    D_1 = (x, y) -> D(x, y)[:,1]
    D_2 = (x, y) -> D(x, y)[:,2]

    # set up
    t = 0.0
    x = copy(x0)
    x_traj = zeros(2, m)
    
    # simulate
    for i in 1:m
        # Perform 1 step of RK4 integration for hat_xₖ₊₁
        k1 = (dt / 2) * F(x[1], x[1])
        k2 = (dt / 2) * F(x[1] + 0.5 * k1[1], x[2] + 0.5 * k1[2])
        k3 = (dt / 2) * F(x[1] + 0.5 * k2[1], x[2] + 0.5 * k2[2])
        k4 = (dt / 2) * F(x[1] + k3[1], x[2] + k3[2]) 
            
        # Update state using weighted average of intermediate values
        x += (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) 

        Rₖ = randn(2)

        x += noise_integrator(x, dt, D, D_1, D_2, Rₖ)

        # Perform 1 step of RK4 integration for hat_xₖ₊₁
        k1 = (dt / 2) * F(x[1], x[1])
        k2 = (dt / 2) * F(x[1] + 0.5 * k1[1], x[2] + 0.5 * k1[2])
        k3 = (dt / 2) * F(x[1] + 0.5 * k2[1], x[2] + 0.5 * k2[2])
        k4 = (dt / 2) * F(x[1] + k3[1], x[2] + k3[2]) 
            
        # Update state using weighted average of intermediate values
        x += (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) 

        x_traj[:,i] .= x

        # update the time
        t += dt
    end

    return x_traj, nothing
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