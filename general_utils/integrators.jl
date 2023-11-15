module Integrators
include("calculus.jl")
using LinearAlgebra, Random, Plots, ForwardDiff, Base.Threads, ProgressBars
using .Calculus: symbolic_matrix_divergence2D
export euler_maruyama1D, leimkuhler_matthews1D, leimkuhler_matthews_markovian1D, hummer_leimkuhler_matthews1D, milstein_method1D, stochastic_heun1D, euler_maruyama2D, leimkuhler_matthews2D, hummer_leimkuhler_matthews2D, euler_maruyama2D_identityD, naive_leimkuhler_matthews2D_identityD, limit_method_with_variable_diffusion1D, limit_method_for_variable_diffusion2D, eugen

function euler_maruyama1D(x0, Vprime, D, D2prime, sigma::Number, m::Integer, dt::Number, Rₖ=nothing, noise_integrator=nothing, n=nothing)
    
    # set up
    t = 0.0
    x = copy(x0)
    x_traj = zeros(m)
    sqrt_dt = sqrt(dt)

    # simulate
    for i in 1:m
        # compute the drift and diffusion coefficients
        D_x = D(x)
        grad_V = Vprime(x)
        div_D2 = D2prime(x)
        drift = -(D_x^2) * grad_V + sigma^2 * div_D2 / 2
        diffusion = sigma * D_x * randn()
        
        # update the configuration
        x += drift * dt + diffusion * sqrt_dt
        x_traj[i] = x
        
        # update the time
        t += dt
    end
    
    return x_traj, nothing
end

function leimkuhler_matthews1D(x0, Vprime, D, D2prime, sigma::Number, m::Integer, dt::Number, Rₖ=nothing, noise_integrator=nothing, n=nothing)
    # Caution: does not converge for diffusion other than D(x) = 1.
    
    # set up
    t = 0.0
    x = copy(x0)
    x_traj = zeros(m)
    if Rₖ === nothing
        Rₖ = randn()
    end
    sqrt_dt = sqrt(dt)

    # simulate
    for i in 1:m
        # compute the drift and diffusion coefficients
        D_x = D(x)
        grad_V = Vprime(x)
        div_D2 = D2prime(x)
        drift = -(D_x^2) * grad_V + sigma^2 * div_D2 / 2
        Rₖ₊₁ = randn()
        diffusion = sigma * D_x * (Rₖ + Rₖ₊₁)/2 
        # update the configuration
        x += drift * dt + diffusion * sqrt_dt
        x_traj[i] = x
        
        # update the time
        t += dt

        # update the noise increment
        Rₖ = copy(Rₖ₊₁)      
    end 
    
    return x_traj, Rₖ
end

function leimkuhler_matthews_markovian1D(x0, Vprime, D, D2prime, sigma::Number, m::Integer, dt::Number, Rₖ=nothing, noise_integrator=nothing, n=nothing)
    # Caution: does not converge for diffusion other than D(x) = 1.

    D² = x -> D(x)^2
    
    # set up
    t = 0.0
    x = copy(x0)
    x_traj = zeros(m)
    sqrt_dt = sqrt(dt)

    for i in 1:m
        D_x = D(x)
        Rₖ = randn()
        x_bar = x + 0.5 * sqrt_dt * sigma * D_x * Rₖ
        x += dt * D²(x) * (-Vprime(x_bar)) + sqrt_dt * sigma * D_x * Rₖ

        x_traj[i] = x_bar

        # update the time
        t += dt
    end

    return x_traj, nothing
end


function eugen_gilles1D(x0, Vprime, D, D2prime, sigma::Number, m::Integer, dt::Number, Rₖ=nothing, noise_integrator=nothing, n=nothing)
    D² = x -> D(x)^2
    F = x -> D²(x) * (-Vprime(x)) + sigma^2 * D2prime(x) /2

    # set up
    t = 0.0
    x = copy(x0)
    x_traj = zeros(m)
    sqrt_dt = sqrt(dt)

    # simulate
    for i in 1:m
        Rₖ = randn()
        # choosing x_tilde = x
        x_bar = x + 0.5 * sqrt_dt * sigma * D(x) * Rₖ
        F_x_bar = F(x_bar)
        x += dt * F_x_bar + noise_integrator(x + dt * F_x_bar / 4, dt, D, Rₖ)

        x_traj[i] = x_bar

        # update the time
        t += dt
    end

    return x_traj, nothing
end

function eugen_gilles2D(x0, Vprime, D, div_DDT, sigma::Number, m::Integer, dt::Number, Rₖ=nothing, noise_integrator=nothing, n=nothing)
    D² = (x, y) -> D(x, y)^2
    F = (x, y) -> D²(x, y) * (-Vprime(x, y)) + sigma^2 * div_DDT(x, y) /2

    # Define first and second columns of the matrix function D
    D_1 = (x, y) -> D(x, y)[:,1]
    D_2 = (x, y) -> D(x, y)[:,2]

    # set up
    t = 0.0
    x = copy(x0)
    x_traj = zeros(2, m)
    sqrt_dt = sqrt(dt)

    # simulate
    for i in 1:m
        Rₖ = randn(2)
        # choosing x_tilde = x
        x_bar = x + 0.5 * sqrt_dt * sigma * D(x...) * Rₖ
        F_x_bar = F(x_bar...)
        x += dt * F_x_bar + noise_integrator(x + dt * F_x_bar / 4, dt, D, D_1, D_2, Rₖ)

        x_traj[:, i] .= x_bar

        # update the time
        t += dt
    end

    return x_traj, nothing
end


function limit_method_with_variable_diffusion1D(x0, Vprime, D, D2prime, sigma::Number, m::Integer, dt::Number, Rₖ=nothing, noise_integrator=nothing, n=5)
    
    Dprime = x -> D2prime(x) / 2D(x)

    # set up
    t = 0.0
    x = copy(x0)
    x_traj = zeros(m)
    if Rₖ === nothing
        Rₖ = randn()
    end

    sqrt_2_dt = sqrt(2 * dt)
    sqrt_2 = sqrt(2)
    sqrt_dt_2 = sqrt(dt / 2)
    inner_step = sqrt_dt_2 / n  

    # simulate
    for i in 1:m
        D_x = D(x)
        grad_V = Vprime(x)
        grad_D = Dprime(x)
        hat_pₖ₊₁ = sigma * Rₖ / sqrt_2 - sqrt_2_dt * D_x * grad_V + (sigma^2) * sqrt_dt_2 * grad_D 
        
        # Perform n steps of RK4 integration for hat_xₖ₊₁
        hat_xₖ₊₁ = x # Initialize hat_xₖ₊₁
        for j in 1:n
            # Compute intermediate values
            k1 = inner_step * D(hat_xₖ₊₁) 
            k2 = inner_step * D(hat_xₖ₊₁ + 0.5 * k1) 
            k3 = inner_step * D(hat_xₖ₊₁ + 0.5 * k2) 
            k4 = inner_step * D(hat_xₖ₊₁ + k3) 
            
            # Update state using weighted average of intermediate values
            hat_xₖ₊₁ += (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * hat_pₖ₊₁
        end
        
        # Perform n steps of RK4 integration for xₖ₊₁
        xₖ₊₁ = hat_xₖ₊₁ # Initialize xₖ₊₁
        Rₖ₊₁ = randn()
        for j in 1:n
            # Compute intermediate values
            k1 = inner_step * D(xₖ₊₁) 
            k2 = inner_step * D(xₖ₊₁ + 0.5 * k1) 
            k3 = inner_step * D(xₖ₊₁ + 0.5 * k2) 
            k4 = inner_step * D(xₖ₊₁ + k3) 
            
            # Update state using weighted average of intermediate values
            xₖ₊₁ += (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * sigma * Rₖ₊₁ / sqrt(2)
        end
        
        # Update the trajectory
        x = xₖ₊₁
        x_traj[i] = x
        
        # update the time
        t += dt

        # update the noise increment
        Rₖ = copy(Rₖ₊₁)
    end

    return x_traj, Rₖ
end

function hummer_leimkuhler_matthews1D(x0, Vprime, D, D2prime, sigma::Number, m::Integer, dt::Number, Rₖ=nothing, noise_integrator=nothing, n=nothing)
    
    # set up
    t = 0.0
    x = copy(x0)
    x_traj = zeros(m)
    if Rₖ === nothing
        Rₖ = randn()
    end
    sqrt_dt = sqrt(dt)

    # simulate
    for i in 1:m
        # compute the drift and diffusion coefficients
        D_x = D(x)
        grad_V = Vprime(x)
        div_D2 = D2prime(x)
        drift = -(D_x^2) * grad_V + (3/4) * sigma^2 * div_D2 / 2
        Rₖ₊₁ = randn()
        diffusion = sigma * D_x * (Rₖ + Rₖ₊₁)/2 
        
        # update the configuration
        x += drift * dt + diffusion * sqrt_dt
        x_traj[i] = x
        
        # update the time
        t += dt

        # update the noise increment
        Rₖ = copy(Rₖ₊₁)      
    end 
    
    return x_traj, Rₖ
end


function euler_maruyama2D(x0, Vprime, D, div_DDT, sigma::Number, m::Integer, dt::Number, Rₖ=nothing, noise_integrator=nothing, n=nothing)
    
    # set up
    t = 0.0
    x = copy(x0)
    n = 2 # dimension
    x_traj = zeros(n, m)
    sqrt_dt = sqrt(dt)

    # simulate
    for i in 1:m
        # compute the drift and diffusion coefficients
        grad_V = Vprime(x[1], x[2])
        D_x = D(x[1], x[2])
        DDT_x = D_x * D_x'
        div_DDT_x = div_DDT(x[1], x[2])
        drift = -DDT_x * grad_V + (sigma^2) * div_DDT_x / 2 
        diffusion = sigma * D_x * randn(n)
        
        # update the configuration
        x += drift * dt + diffusion * sqrt_dt
        x_traj[:,i] .= x
        
        # update the time
        t += dt
    end
    
    return x_traj, nothing
end


function leimkuhler_matthews2D(x0, Vprime, D, div_DDT, sigma::Number, m::Integer, dt::Number, Rₖ=nothing)
    
    # set up
    t = 0.0
    x = copy(x0)
    n = 2
    x_traj = zeros(n, m)
    if Rₖ === nothing
        Rₖ = randn(n)
    end
    sqrt_dt = sqrt(dt)

    # simulate
    for i in 1:m
        # compute the drift and diffusion coefficients
        grad_V = Vprime(x[1], x[2])
        D_x = D(x[1], x[2])
        DDT_x = D_x * D_x'
        div_DDT_x = div_DDT(x[1], x[2])
        drift = -DDT_x * grad_V + (sigma^2)/2 * div_DDT_x 
        Rₖ₊₁ = randn(n)
        diffusion = sigma * D_x * (Rₖ + Rₖ₊₁)/2 
        
        # update the configuration
        x += drift * dt + diffusion * sqrt_dt
        x_traj[:,i] .= x
        
        # update the time
        t += dt

        # update the noise increment
        Rₖ = copy(Rₖ₊₁)      
    end 
    
    return x_traj, Rₖ
end

function hummer_leimkuhler_matthews2D(x0, Vprime, D, div_DDT, sigma::Number, m::Integer, dt::Number, Rₖ=nothing, noise_integrator=nothing, n=nothing)
    
    # set up
    t = 0.0
    x = copy(x0)
    n = 2
    x_traj = zeros(n, m)
    if Rₖ === nothing
        Rₖ = randn(n)
    end
    sqrt_dt = sqrt(dt)

    # simulate
    for i in 1:m
        # compute the drift and diffusion coefficients
        grad_V = Vprime(x[1], x[2])
        D_x = D(x[1], x[2])
        DDT_x = D_x * D_x'
        div_DDT_x = div_DDT(x[1], x[2])
        drift = -DDT_x * grad_V + (3/8) * (sigma^2) * div_DDT_x 
        Rₖ₊₁ = randn(n)
        diffusion = sigma * D_x * (Rₖ + Rₖ₊₁)/2 
        
        # update the configuration
        x += drift * dt + diffusion * sqrt_dt
        x_traj[:,i] .= x
        
        # update the time
        t += dt

        # update the last increment
        Rₖ = copy(Rₖ₊₁)      
    end 
    
    return x_traj, Rₖ
end


end # module Integrators