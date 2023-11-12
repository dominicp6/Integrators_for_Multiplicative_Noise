module Integrators
include("calculus.jl")
using LinearAlgebra, Random, Plots, ForwardDiff, Base.Threads, ProgressBars
using .Calculus: symbolic_matrix_divergence2D
export euler_maruyama1D, leimkuhler_matthews1D, leimkuhler_matthews_markovian1D, hummer_leimkuhler_matthews1D, milstein_method1D, stochastic_heun1D, euler_maruyama2D, naive_leimkuhler_matthews2D, hummer_leimkuhler_matthews2D, euler_maruyama2D_identityD, naive_leimkuhler_matthews2D_identityD, limit_method_with_variable_diffusion1D, limit_method_for_variable_diffusion2D

function euler_maruyama1D(q0, Vprime, D, D2prime, sigma::Number, m::Integer, dt::Number, Rₖ=nothing, noise_integrator=nothing, n=nothing)
    
    # set up
    t = 0.0
    q = copy(q0)
    q_traj = zeros(m)

    # simulate
    for i in 1:m
        # compute the drift and diffusion coefficients
        Dq = D(q)
        grad_V = Vprime(q)
        div_D2 = D2prime(q)
        drift = -(Dq^2) * grad_V + sigma^2 * div_D2 / 2
        diffusion = sigma * Dq * randn()
        
        # update the configuration
        q += drift * dt + diffusion * sqrt(dt)
        q_traj[i] = q
        
        # update the time
        t += dt
    end
    
    return q_traj, nothing
end

function leimkuhler_matthews1D(q0, Vprime, D, D2prime, sigma::Number, m::Integer, dt::Number, Rₖ=nothing, noise_integrator=nothing, n=nothing)
    
    # set up
    t = 0.0
    q = copy(q0)
    q_traj = zeros(m)
    if Rₖ === nothing
        Rₖ = randn()
    end

    # simulate
    for i in 1:m
        # compute the drift and diffusion coefficients
        Dq = D(q)
        grad_V = Vprime(q)
        div_D2 = D2prime(q)
        drift = -(Dq^2) * grad_V + sigma^2 * div_D2 / 2
        Rₖ₊₁ = randn()
        diffusion = sigma * Dq * (Rₖ + Rₖ₊₁)/2 
        # update the configuration
        q += drift * dt + diffusion * sqrt(dt) 
        q_traj[i] = q
        
        # update the time
        t += dt

        # update the noise increment
        Rₖ = copy(Rₖ₊₁)      
    end 
    
    return q_traj, Rₖ
end

function leimkuhler_matthews_markovian1D(x0, Vprime, D, D2prime, sigma::Number, m::Integer, dt::Number, Rₖ=nothing, noise_integrator=nothing, n=nothing)
    # Caution: does not support variable diffusion other than D(x) = 1.

    D² = x -> D(x)^2
    
    # set up
    t = 0.0
    x = copy(x0)
    x_traj = zeros(m)

    for i in 1:m
        D_x = D(x)
        Rₖ = randn()
        x_bar = x + 0.5 * sqrt(dt) * sigma * D_x * Rₖ
        x += dt * D²(x) * (-Vprime(x_bar)) + sqrt(dt) * sigma * D_x * Rₖ

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

    # simulate
    for i in 1:m
        Rₖ = randn()
        # choosing x_tilde = x
        x_bar = x + 0.5 * sqrt(dt) * sigma * D(x) * Rₖ
        F_x_bar = F(x_bar)
        x += dt * F_x_bar + noise_integrator(x + dt * F_x_bar / 4, dt, D, Rₖ)

        x_traj[i] = x_bar

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

    # simulate
    for i in 1:m
        D_x = D(x)
        grad_V = Vprime(x)
        grad_D = Dprime(x)
        hat_pₖ₊₁ = sigma * Rₖ / sqrt(2) - sqrt_2_dt * D_x * grad_V + (sigma^2) * sqrt_2_dt * grad_D / 2
        
        sqrt_h_2 = sqrt(dt / 2)
        inner_step = sqrt_h_2 / n  # Divide by n for each internal RK4 step
        
        # Perform n steps of RK4 integration for hat_xₖ₊₁
        hat_xₖ₊₁ = x # Initialize hat_xₖ₊₁
        for j in 1:n
            # Compute intermediate values
            k1 = inner_step * D(hat_xₖ₊₁) * hat_pₖ₊₁
            k2 = inner_step * D(hat_xₖ₊₁ + 0.5 * k1) * hat_pₖ₊₁
            k3 = inner_step * D(hat_xₖ₊₁ + 0.5 * k2) * hat_pₖ₊₁
            k4 = inner_step * D(hat_xₖ₊₁ + k3) * hat_pₖ₊₁
            
            # Update state using weighted average of intermediate values
            hat_xₖ₊₁ += (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        end
        
        # Perform n steps of RK4 integration for xₖ₊₁
        xₖ₊₁ = hat_xₖ₊₁ # Initialize xₖ₊₁
        Rₖ₊₁ = randn()
        for j in 1:n
            # Compute intermediate values
            k1 = inner_step * D(xₖ₊₁) * Rₖ₊₁
            k2 = inner_step * D(xₖ₊₁ + 0.5 * k1) * Rₖ₊₁
            k3 = inner_step * D(xₖ₊₁ + 0.5 * k2) * Rₖ₊₁
            k4 = inner_step * D(xₖ₊₁ + k3) * Rₖ₊₁
            
            # Update state using weighted average of intermediate values
            xₖ₊₁ += (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * sigma / sqrt(2)
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

function hummer_leimkuhler_matthews1D(q0, Vprime, D, D2prime, sigma::Number, m::Integer, dt::Number, Rₖ=nothing, noise_integrator=nothing, n=nothing)
    
    # set up
    t = 0.0
    q = copy(q0)
    q_traj = zeros(m)
    if Rₖ === nothing
        Rₖ = randn()
    end

    # simulate
    for i in 1:m
        # compute the drift and diffusion coefficients
        Dq = D(q)
        grad_V = Vprime(q)
        div_D2 = D2prime(q)
        drift = -(Dq^2) * grad_V + (3/4) * sigma^2 * div_D2 / 2
        Rₖ₊₁ = randn()
        diffusion = sigma * Dq * (Rₖ + Rₖ₊₁)/2 
        
        # update the configuration
        q += drift * dt + diffusion * sqrt(dt) 
        q_traj[i] = q
        
        # update the time
        t += dt

        # update the noise increment
        Rₖ = copy(Rₖ₊₁)      
    end 
    
    return q_traj, Rₖ
end

function milstein_method1D(q0, Vprime, D, D2prime, sigma::Number, m::Integer, dt::Number, Rₖ=nothing, noise_integrator=nothing, n=nothing)
    # TODO: need to double-check this implementation for D -> D^2 notation
    
    # set up
    t = 0.0
    q = copy(q0)
    q_traj = zeros(m)

    # simulate
    for i in 1:m
        # compute the drift and diffusion coefficients
        Dq = D(q)
        grad_V = Vprime(q)
        div_D2 = D2prime(q)
        drift = -(Dq^2) * grad_V + sigma^2 * div_D2 / 2
        Rₖ = randn()
        diffusion = sigma * Dq * Rₖ
        second_order_correction = (sigma^2 / 4) * div_D2 * (Rₖ^2 - 1) 
        
        # update the configuration
        q += drift * dt + diffusion * sqrt(dt) + second_order_correction * dt
        q_traj[i] = q
        
        # update the time
        t += dt   
    end 
    
    return q_traj, nothing
end

function stochastic_heun1D(q0, Vprime, D, D2prime, sigma::Number, m::Integer, dt::Number, Rₖ=nothing, noise_integrator=nothing, n=nothing)
    # TODO: need to update this implementation to reflect notation change


    # set up
    t = 0.0
    q = copy(q0)
    q_traj = zeros(m)
    
    # simulate
    for i in 1:m
        # Compute drift and diffusion coefficients at the current position
        Dq = D(q)
        grad_V = Vprime(q)
        grad_D = Dprime(q)
        drift = -Dq * grad_V + 0.5 * sigma * grad_D   #  gradient term gets 0.5 factor to correct for Stratanovich interpretation
        diffusion = sqrt(2 * sigma * Dq)
        
        # Compute the predicted next state using Euler-Maruyama
        Rₖ = randn()
        q_pred = q + drift * dt + diffusion * Rₖ * sqrt(dt)
        
        # Compute drift and diffusion coefficients at the predicted position
        Dq_pred = D(q_pred)
        grad_V_pred = Vprime(q_pred)
        grad_D_pred = Dprime(q_pred)
        drift_pred = -Dq_pred * grad_V_pred + 0.5 * sigma * grad_D_pred   # gradient term gets 0.5 factor to correct for Stratanovich interpretation
        diffusion_pred = sqrt(2 * sigma * Dq_pred)
        
        # Compute the corrected next state using a weighted average
        q += 0.5 * (drift + drift_pred) * dt + 0.5 * (diffusion + diffusion_pred) * Rₖ * sqrt(dt)
        q_traj[i] = q
        
        # Update the time
        t += dt
    end
    
    return q_traj, nothing
end


function euler_maruyama2D(q0, Vprime, D, div_DDT, sigma::Number, m::Integer, dt::Number, Rₖ=nothing)
    
    # set up
    t = 0.0
    q = copy(q0)
    n = length(q0)
    q_traj = zeros(n, m)

    # simulate
    for i in 1:m
        # compute the drift and diffusion coefficients
        grad_V = Vprime(q[1], q[2])
        Dq = D(q[1], q[2])
        DDTq = Dq * Dq'
        div_DDTq = div_DDT(q[1], q[2])
        drift = -DDTq * grad_V + sigma * div_DDTq 
        diffusion = sqrt(2 * sigma) * Dq * randn(n)
        
        # update the configuration
        q += drift * dt + diffusion * sqrt(dt)
        q_traj[:,i] .= q
        
        # update the time
        t += dt
    end
    
    return q_traj, nothing
end

"""
Optimised version of euler_maruyama2D for the case where D is the identity matrix
"""
function euler_maruyama2D_identityD(q0, Vprime, D, div_DDT, sigma::Number, m::Integer, dt::Number, Rₖ=nothing)
    
    # set up
    t = 0.0
    q = copy(q0)
    n = length(q0)
    q_traj = zeros(n, m)

    # simulate
    for i in 1:m
        # compute the drift and diffusion coefficients
        grad_V = Vprime(q[1], q[2])
        drift = -grad_V
        diffusion = sqrt(2 * sigma) * randn(n)
        
        # update the configuration
        q += drift * dt + diffusion * sqrt(dt)
        q_traj[:,i] .= q
        
        # update the time
        t += dt
    end
    
    return q_traj, nothing
end

"""
Optimised version of naive_leimkuhler_matthews2D for the case where D is the identity matrix
"""
function naive_leimkuhler_matthews2D_identityD(q0, Vprime, D, div_DDT, sigma::Number, m::Integer, dt::Number, Rₖ=nothing)
    
    # set up
    t = 0.0
    q = copy(q0)
    n = length(q0)
    q_traj = zeros(n, m)
    if Rₖ === nothing
        Rₖ = randn(n)
    end

    # simulate
    for i in 1:m
        # compute the drift and diffusion coefficients
        grad_V = Vprime(q[1], q[2])
        drift = -grad_V
        Rₖ₊₁ = randn(n)
        diffusion = sqrt(2 * sigma) * (Rₖ + Rₖ₊₁)/2 
        
        # update the configuration
        q += drift * dt + diffusion * sqrt(dt) 
        q_traj[:,i] .= q
        
        # update the time
        t += dt

        # update the noise increment
        Rₖ = copy(Rₖ₊₁)      
    end 
    
    return q_traj, Rₖ
end

function naive_leimkuhler_matthews2D(q0, Vprime, D, div_DDT, sigma::Number, m::Integer, dt::Number, Rₖ=nothing)
    
    # set up
    t = 0.0
    q = copy(q0)
    n = length(q0)
    q_traj = zeros(n, m)
    if Rₖ === nothing
        Rₖ = randn(n)
    end

    # simulate
    for i in 1:m
        # compute the drift and diffusion coefficients
        grad_V = Vprime(q[1], q[2])
        Dq = D(q[1], q[2])
        DDTq = Dq * Dq'
        div_DDTq = div_DDT(q[1], q[2])
        drift = -DDTq * grad_V + sigma * div_DDTq 
        Rₖ₊₁ = randn(n)
        diffusion = sqrt(2 * sigma) * Dq * (Rₖ + Rₖ₊₁)/2 
        
        # update the configuration
        q += drift * dt + diffusion * sqrt(dt) 
        q_traj[:,i] .= q
        
        # update the time
        t += dt

        # update the noise increment
        Rₖ = copy(Rₖ₊₁)      
    end 
    
    return q_traj, Rₖ
end

function hummer_leimkuhler_matthews2D(q0, Vprime, D, div_DDT, sigma::Number, m::Integer, dt::Number, Rₖ=nothing)
    
    # set up
    t = 0.0
    q = copy(q0)
    n = length(q0)
    q_traj = zeros(n, m)
    if Rₖ === nothing
        Rₖ = randn(n)
    end

    # simulate
    for i in 1:m
        # compute the drift and diffusion coefficients
        grad_V = Vprime(q[1], q[2])
        Dq = D(q[1], q[2])
        DDTq = Dq * Dq'
        div_DDTq = div_DDT(q[1], q[2])
        drift = -DDTq * grad_V + (3/4) * sigma * div_DDTq 
        Rₖ₊₁ = randn(n)
        diffusion = sqrt(2 * sigma) * Dq * (Rₖ + Rₖ₊₁)/2 
        
        # update the configuration
        q += drift * dt + diffusion * sqrt(dt) 
        q_traj[:,i] .= q
        
        # update the time
        t += dt

        # update the last increment
        Rₖ = copy(Rₖ₊₁)      
    end 
    
    return q_traj, Rₖ
end

function stochastic_heun2D(q0, Vprime, D, div_DDT, sigma::Number, m::Integer, dt::Number, Rₖ=nothing)
    
    # set up
    t = 0.0
    q = copy(q0)
    n = length(q0)
    q_traj = zeros(n, m)
    
    # simulate
    for i in 1:m
        # Compute drift and diffusion coefficients at the current position
        grad_V = Vprime(q[1], q[2])
        Dq = D(q[1], q[2])
        DDTq = Dq * Dq'
        div_DDTq = div_DDT(q[1], q[2])
        drift = -DDTq * grad_V + 0.5 * sigma * div_DDTq   #  gradient term gets 0.5 factor to correct for Stratanovich interpretation
        diffusion = sqrt(2 * sigma) * Dq
        
        # Compute the predicted next state using Euler-Maruyama
        Rₖ = randn(n)
        q_pred = q + drift * dt + diffusion * Rₖ * sqrt(dt)
        
        # Compute drift and diffusion coefficients at the predicted position
        grad_V_pred = Vprime(q_pred[1], q_pred[2])
        Dq_pred = D(q_pred[1], q_pred[2])
        DDTq_pred = Dq_pred * Dq_pred'
        div_DDTq_pred = div_DDT(q_pred[1], q_pred[2])
        drift_pred = -DDTq_pred * grad_V_pred + 0.5 * sigma * div_DDTq_pred   # gradient term gets 0.5 factor to correct for Stratanovich interpretation
        diffusion_pred = sqrt(2 * sigma) * Dq_pred
        
        # Compute the corrected next state using a weighted average
        q += 0.5 * (drift + drift_pred) * dt + 0.5 * (diffusion + diffusion_pred) * Rₖ * sqrt(dt)
        q_traj[:,i] .= q
        
        # Update the time
        t += dt
    end
    
    return q_traj, nothing
end


"""
Optimised version of stochastic_heun2D for the case where D is the identity matrix
"""
function stochastic_heun2D_identityD(q0, Vprime, D, div_DDT, sigma::Number, m::Integer, dt::Number, Rₖ=nothing)
    
    # set up
    t = 0.0
    q = copy(q0)
    n = length(q0)
    q_traj = zeros(n, m)
    
    # simulate
    for i in 1:m
        # Compute drift and diffusion coefficients at the current position
        grad_V = Vprime(q[1], q[2])
        drift = -grad_V  
        diffusion = sqrt(2 * sigma)
        
        # Compute the predicted next state using Euler-Maruyama
        Rₖ = randn(n)
        q_pred = q + drift * dt + diffusion * Rₖ * sqrt(dt)
        
        # Compute drift and diffusion coefficients at the predicted position
        grad_V_pred = Vprime(q_pred[1], q_pred[2])
        drift_pred = -grad_V_pred 
        
        # Compute the corrected next state using a weighted average
        q += 0.5 * (drift + drift_pred) * dt + diffusion * Rₖ * sqrt(dt)
        q_traj[:,i] .= q
        
        # Update the time
        t += dt
    end
    
    return q_traj, nothing
end

end # module Integrators