module Potentials
using Symbolics
export bowl2D, doubleWell1D, quadrupleWell2D, moroCardin2D, muller_brown, LM2013, localWell1D, transformedLocalWell1D, transformedLM2013, transformed2LM2013, softWell1D, transformedSoftWell1D, transformed2SoftWell1D, softQuadrupleWell2D, q4Potential2D, doubleWellChannel2D, q1Soft2D, ringND

function bowl2D(q::AbstractVector{T}) where T<:Real
    # 2D bowl potential
    x, y = q
    0.5*(x^2+y^2)
end

function q4Potential(q::T) where T<:Real
    # 1D quartic potential
    return q^4 / 4
end

function q2Potential(q::T) where T<:Real
    # 1D quadratic potential
    return q^2 / 2
end

function q2Potential2D(x::T, y::T) where T<:Real
    # 2D quadratic potential
    return x^2 / 2 + y^2 / 2
end

function q1Soft2D(x::T, y::T) where T<:Real
    return sqrt((x^2 + y^2) * (4/5))
end

function q4Potential2D(x::T, y::T) where T<:Real
    # 2D quartic potential
    return x^4 / 4 + y^4 / 4
end

function doubleWell1D(q::T) where T<:Real
    # 1D double well potential
    h = 2
    c = 2
    return -(1/4)*(q^2)*(h^4) + (1/2)*(c^2)*(q^4)
end

function LM2013(q::T) where T<:Real
    # 1D potential from L. M. 2013
    return q^4 /4 + sin(1 + 5q)
end

function softWell1D(q::T) where T<:Real
    # 1D soft well potential
    return q^2 / 2 + sin(1 + 3q)
end

function transformedSoftWell1D(q::T) where T<:Real
    # transformed soft well 1D to remove linear diffusion D(q) = 1 + |q| (global transform)
    # sigma = 1
    r = (q/4)*(abs(q)+4)
    return softWell1D(r) - log(1+ abs(q) + q^2 / 4)
end

function doubleWellChannel2D(x::T, y::T) where T<:Real
    term1 = (5*y)^2  / ((5*x)^2 + 1)
    term2 = (1/6) * (sqrt(4 * (1-x^2 - y^2)^2 + 2*(x^2 - 2)^2 + ((x+y)^2 -1)^2 +((x-y)^2 -1)^2) -2)

    return term1 + term2
end

function transformed2SoftWell1D(q::T) where T<:Real
    # transformed soft well 1D to remove linear diffusion D(q) = 1 + |q| (time rescaling)
    sigma = 1
    return softWell1D(q) - sigma*log(1+ abs(q))
end

function transformedLM2013(q::T) where T<:Real
    # transformed LM2013 to remove quadratic diffusion D(q) = 1 + q^2 (global transform)
    r = sinh(q)
    sigma = 1
    return LM2013(r) - sigma*log(1+r^2)
end

function transformed2LM2013(q::T) where T<:Real
    # transformed LM2013 to remove quadratic diffusion D(q) = 1 + q^2 (time rescaling)
    sigma = 1
    return LM2013(q) - sigma*log(1+q^2)
end

function localWell1D(q::T) where T<:Real
    # 1D localised well potential with metastable states
    # only valid for q in [-0.75, 0.75]
    
    return 1/(q+3/4) + 1/(3/4 - q) + 0.2 * sin(10q)
end

function transformedLocalWell1D(q::T) where T<:Real
    # 1D localised well potential with metastable states
    # only valid for r in [-1, sqrt(7)-2]
    # obtained from local well 1D after a transformation to remove linear diffusion D(q) = 1 + q
    r = (q^2)/4 + q
    
    return localWell1D(r) - 2*log(abs(q/2 + 1))
end

function quadrupleWell2D(x::T, y::T) where T<:Real
    # 2D quadruple well potential
    h = 2
    c = 2
    return -(1/4)*(x^2)*(h^4) + (1/2)*(c^2)*(x^4) + -(1/4)*(y^2)*(h^4) + (1/2)*(c^2)*(y^4)
end

function softQuadrupleWell2D(x::T, y::T) where T<:Real
    # soft 2D quadruple well potential
    return  sqrt(17/16 - 2x^2 + x^4) + sqrt(17/16 - 2y^2 + y^4)
end

function moroCardin2D(q::AbstractVector{T}) where T<:Real
    # 2D Moro-Cardin potential
    x, y = q
    return 5*(x^2-1)^2 + 10*atan(7*pi/9)*y^2
end

function muller_brown(q::AbstractVector{T}) where T <: Real
    A = [-200.0; -100.0; -170.0; 15.0]
    a = [-1.0; -1.0; -6.5; 0.7]
    b = [0.0; 0.0; 11.0; 0.6]
    c = [-10.; -10.; -6.5; 0.7]
    x_ = [1.; 0.; -0.5; -1.] 
    y_ = [0.; 0.5; 1.5; 1.]

    z = sum(A .* exp.(a .* (q[1] .- x_).^2 .+ b .* (q[1] .- x_) .* (q[2] .- y_) .+ c .* (q[2] .- y_).^2))
    
    return z
end

function q2Ring(x::T, y::T) where T <: Real
    return 25 * (1 - sqrt(x^2 + y^2))^4
end

end # module Potentials