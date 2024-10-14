module DiffusionTensors
using LinearAlgebra
export Dconst1D, Dabs1D, Dquadratic1D, Dconst2D, Dabs2D, Dquadratic2D, DmoroCardin, Doseen, Dcosperturb1D, Dcosperturb2D, DabsSquareRoot1D, DdoubleWellChannelAnisotropic, DanisotropicI, DanisotropicII, DanisotropicIIreversed, DanisotropicIII, DanisotropicIV, DanisotropicV, Dradial, Dannulus

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

function Dradial(x::T, y::T) where T<:Real
    return [0.01 + 0.5 * x^2 0.5 * x*y; 0.5 * x*y 0.01 + 0.5 * y^2]
end

function Dannulus(x::T, y::T) where T<:Real
    return [0.01 + 0.5 * y^2 -0.5 * x*y; -0.5 * x*y 0.01 + 0.5 * x^2]
end

function DanisotropicI(x::T, y::T) where T<:Real
    return [1.0 0; 0 1.5]
end

function DanisotropicII(x::T, y::T) where T<:Real
    theta = atan(y, x)
    return [1-sin(theta)^2 / 2 cos(theta) * sin(theta) / 2; cos(theta) * sin(theta)/2 1 - cos(theta)^2 / 2]
end

function DanisotropicIIreversed(x::T, y::T) where T<:Real
    theta = atan(y, x) - π/2
    return [1-sin(theta)^2 / 2 cos(theta) * sin(theta) / 2; cos(theta) * sin(theta)/2 1 - cos(theta)^2 / 2]
end

function DanisotropicIII(x::T, y::T) where T<:Real
    theta = atan(y, x)
    return (1.0 + 5.0 * exp(- (x^2 + y^2) / (2 * 0.3^2)))^(-1) * [1+sin(theta)^2 / 2 -cos(theta) * sin(theta) / 2; -cos(theta) * sin(theta)/2 1 + cos(theta)^2 / 2]
end

function _softQuadrupleWell2D(x::T, y::T) where T<:Real
    # soft 2D quadruple well potential
    return  sqrt(17/16 - 2x^2 + x^4) + sqrt(17/16 - 2y^2 + y^4)
end

function DanisotropicIIIqw(x::T, y::T) where T <:Real
    theta = atan(y, x)
    prefactor = exp(0.5 * _softQuadrupleWell2D(x,y) - 0.25 * x^2 - 0.25 * y^2)
    return prefactor * (1.0 + 5.0 * exp(- (x^2 + y^2) / (2 * 0.3^2)))^(-1) * [1+sin(theta)^2 / 2 -cos(theta) * sin(theta) / 2; -cos(theta) * sin(theta)/2 1 + cos(theta)^2 / 2]
end

function DanisotropicIV(x::T, y::T) where T<:Real
    theta = atan(y, x)
    return [1-sin(theta)^2 / 2 cos(theta) * sin(theta) / 2; cos(theta) * sin(theta)/2 1 - cos(theta)^2 / 2] * sin(3*x) * (2 * (exp(y) + 0.5) / (1 + exp(y)))
end

function function_1D(x, a, b)
    result = 0.0
    for i in 1:length(a)
        result += a[i] * cos(i * x + b[i]) / sqrt(i+1)
    end

    return result
end

function DanisotropicV(x::T, y::T) where T<:Real
    list_of_a = [
    [1.10609511, -1.54558424, 0.54125102, -0.39846411, -1.82174564],
    [-0.00367704, -0.57833659, 0.8950395, -1.69792379, 0.88727313],
    [-0.37859394, 0.26414013, -0.97688349, 0.84549204, -1.61668411],
    [-1.33109584, 2.48349712, -1.46783594, -1.05530731, 0.04105189],
    [0.20287527, -1.15666989, 2.17669036, 1.06308684, -0.12260126],
    [0.24829756, 0.27780516, 1.02279586, -0.59432211, 0.19903113],
    [-1.00978779, -1.04440693, -0.16481296, 0.45701152, -0.73295171],
    [-0.09623664, -0.6489985, -0.60125891, -0.17327419, -1.80877051]
    ]
    list_of_b = [
        [-0.9795475 , 0.39279739, -0.742023, -0.02982569, 1.23266377],
        [-0.96997948, -1.03639312, 0.44054334, 0.14259723, 0.73073318],
        [0.26579704, 0.86919929, -0.68803677, 1.55283354, -1.30792944],
        [-0.36751069, 1.49241352, 0.93835149, 0.84830419, -1.20974195],
        [-0.27629405, 0.1472921, 1.19804383, 1.34801694, -1.41093702],
        [0.08145364, 1.76403037, 1.22359871, 0.39626338, 0.72073047],
        [-0.43352454, 0.35754441, -0.67806083, 0.86632526, -0.00166055],
        [0.64026383, -0.65200475, 2.223142, 0.7834324, 0.03410119]
    ]

    D1 = function_1D(x, list_of_a[1], list_of_b[1])^2 * function_1D(y, list_of_a[2], list_of_b[2])^2 + 0.2
    D2 = function_1D(x, list_of_a[3], list_of_b[3])^2 * function_1D(y, list_of_a[4], list_of_b[4])^2 + 0.2
    D = Diagonal([D1, D2])
    P = [function_1D(x, list_of_a[5], list_of_b[5]) * function_1D(y, list_of_a[6], list_of_b[6]) + 0.2 function_1D(x, list_of_a[7], list_of_b[7]) * function_1D(y, list_of_a[8], list_of_b[8]) + 0.2; function_1D(x, list_of_a[7], list_of_b[7]) * function_1D(y, list_of_a[8], list_of_b[8]) + 0.2 function_1D(x, list_of_a[5], list_of_b[5]) * function_1D(y, list_of_a[6], list_of_b[6]) + 0.2]
    M1 = similar(P)
    M2 = similar(P)
    mul!(M1, P, D)
    mul!(M2, M1, P')

    return M2
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
end

function Doseen(x::T, y::T) where T<:Real
    r2 = x^2 + y^2
    return [1.0 + x^2/r2 x*y/r2; x*y/r2 1 + y^2/r2]
end

end # module DiffusionTensors