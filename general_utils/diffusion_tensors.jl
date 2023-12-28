module DiffusionTensors
using LinearAlgebra
export Dconst1D, Dabs1D, Dquadratic1D, Dconst2D, Dabs2D, Dquadratic2D, DmoroCardin, Doseen, Dcosperturb1D, Dcosperturb2D, DabsSquareRoot1D, DmoroCardinAnisotropic, DdoubleWellChannelAnisotropic

# This script defines preset diffusion tensors to test the code

function Dconst1D(q::T) where T<:Real
    return 1.0
end

function Dabs1D(q::T) where T<:Real
    return 1.0 + abs(q)
end

function DabsSquareRoot1D(q::T) where T<:Real
    return (1.0 + abs(q))^0.5
end
    
function Dquadratic1D(q::T) where T<:Real
    return 1.0 + q^2 
end

function Dcosperturb1D(q::T) where T<:Real
    return 1.5 + 0.5 * cos(q)
end

function Dsinperturb1D(q::T) where T<:Real
    return 1.5 + 0.5 * sin(q)
end

function Dconst2D(x::T, y::T) where T<:Real
    return Matrix{Float64}(I, 2, 2)
end

function Dcosperturb2D(x::T, y::T) where T<:Real
    return (1.5 + 0.25 * (cos(x) + cos(y))) * Matrix{Float64}(I, 2, 2)
end

function Dabs2D(x::T, y::T) where T<:Real
    return (abs(x) + abs(y) + 0.001) * Matrix{Float64}(I, 2, 2)
end

function Dquadratic2D(x::T, y::T) where T<:Real
    return (x^2 + y^2 + 0.001) * Matrix{Float64}(I, 2, 2)
end

function DmoroCardin(x::T, y::T) where T<:Real
    return (1.0 + 5.0 * exp(- (x^2 + y^2) / (2 * 0.3^2)))^(-1) * Matrix{Float64}(I, 2, 2)
end

function DmoroCardinAnisotropic(x::T, y::T) where T<:Real
    theta_x = atan(y/x)
    theta_y = atan(x/y)
    return (1.0 + 5.0 * exp(- (x^2 + y^2) / (2 * 0.3^2)))^(-1) * [cos(theta_x)^2 cos(theta_x)*cos(theta_y); cos(theta_x)*cos(theta_y) cos(theta_y)^2]
end

function DdoubleWellChannelAnisotropic(x::T, y::T) where T<:Real
    second_partial_y_derivative = 10 / (1 + 25 * x^2) + (sqrt(2) * (20 * x^6 + 2 * x^4 * (-31 + 18 * y^2) + 5 * x^2 * (13 - 18 * y^2 + 9 * y^4) + 3 * (-7 + 21 * y^2 - 9 * y^4 + 3 * y^6))) / (3 * (7 + 4 * x^4 - 6 * y^2 + 3 * y^4 + 10 * x^2 * (-1 + y^2))^(3/2))
    term1 = (250 * (-1 + 75 * x^2) * y^2) / (1 + 25 * x^2)^3
    term2 = (sqrt(2) * (16 * x^6 + 60 * x^4 * (-1 + y^2) + 12 * x^2 * (7 - 6 * y^2 + 3 * y^4) + 5 * (-7 + 13 * y^2 - 9 * y^4 + 3 * y^6))) / (3 * (7 + 4 * x^4 - 6 * y^2 + 3 * y^4 + 10 * x^2 * (-1 + y^2))^(3/2))
    second_partial_x_derivative = term1 + term2

    potential_term_1 = (5*y)^2  / ((5*x)^2 + 1)
    potential_term_2 = (1/6) * (sqrt(4 * (1-x^2 - y^2)^2 + 2*(x^2 - 2)^2 + ((x+y)^2 -1)^2 +((x-y)^2 -1)^2) - 2)
    potential = potential_term_1 + potential_term_2

    return [max(1.0, 2.0 - potential)/(1+min(abs(second_partial_x_derivative), 5)) 0; 0 max(1.0, 2.0 - potential)/(1+min(abs(second_partial_y_derivative), 5))]

function Doseen(x::T, y::T) where T<:Real
    r2 = x^2 + y^2
    return [1.0 + x^2/r2 x*y/r2; x*y/r2 1 + y^2/r2]
end

end # module DiffusionTensors