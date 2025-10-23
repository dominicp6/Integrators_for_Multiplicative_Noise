module Calculus
using ForwardDiff, Symbolics
export matrix_divergence, differentiate1D, differentiateND, differentiate2D, symbolic_vector_divergence2D, gradientND, symbolic_matrix_divergenceND

"""
Compute the symbolic derivative of a scalar function of a single variable
"""
function differentiate1D(f)
    @variables x
    gradGen = Differential(x)(f(x))
    gradExp = expand_derivatives(gradGen)
    gradFn = Symbolics.build_function(gradExp, x, expression=false)

    return gradFn
end

"""
Compute the symbolic gradient of a scalar function of two variables
"""
function differentiate2D(f::Function)
    @variables x y
    grad_f_expr = Symbolics.gradient(f(x,y), [x,y])
    grad1 = build_function(grad_f_expr[1], [x,y], expression=false)
    grad2 = build_function(grad_f_expr[2], [x,y], expression=false)
    grad = (x, y) -> begin
        [grad1([x, y]), grad2([x, y])]
    end

    return grad
end

"""
Compute the symbolic divergence of a vector function of two variables
"""
function symbolic_vector_divergence2D(V)
    @variables x y
    div1 = Differential(x)(V(x,y)[1])
    div2 = Differential(y)(V(x,y)[2])
    div1 = expand_derivatives(div1)
    div2 = expand_derivatives(div2)
    div1Fn = Symbolics.build_function(div1, [x,y], expression=false)
    div2Fn = Symbolics.build_function(div2, [x,y], expression=false)
    div = (x, y) -> begin
        div1Fn([x, y]) + div2Fn([x, y])
    end

    return div   #println(div(1.0, 2.0)) # expect 17.0
end

"""
Compute the symbolic matrix divergence of a matrix function of two variables
"""
function symbolic_matrix_divergence2D(M)
    # The matrix divergence is defined as the column vector resulting from the vector divergence of each row
    V1 = (x, y) -> M(x,y)[1,:]
    V2 = (x, y) -> M(x,y)[2,:]
    div_M1 = symbolic_vector_divergence2D(V1)
    div_M2 = symbolic_vector_divergence2D(V2)
    div_M = (x, y) -> begin
        [div_M1(x, y),  div_M2(x, y)]
    end
    
    return div_M
end

"""
Compute the symbolic matrix divergence of a matrix function of a vector variable
"""
function symbolic_matrix_divergenceND(M, dim::Int)
    @variables q[1:dim]

    row_divergences = []
    for i in 1:dim
        V_i = q -> M(q)[i, :]

        div_V_i = Symbolics.divergence(V_i(q), q)
        append!(row_divergences, [expand_derivatives(div_V_i)])
    end

    div_M_fn = Symbolics.build_function(vcat(row_divergences...), q, expression=false)

    return div_M_fn
end

"""
Compute the symbolic gradient of a scalar function of multiple variables
"""
function differentiateND(f::Function)
    @variables x[1:length(f.args)...]
    gradGen = Symbolics.gradient(f(x...), x)
    gradFn = Symbolics.build_function(gradGen, x, expression=false)

    return gradFn
end

"""
Compute the symbolic gradient of a scalar function of a vector variable
"""
function gradientND(f::Function, dim::Int)
    @variables x[1:dim]
    gradGens = [Differential(x[i])(f(x)) for i in 1:dim]
    built_derivatives = []
    for gradGen in gradGens
        temp = expand_derivatives(gradGen)
        gradFn = Symbolics.build_function(gradExp, x, expression=false)
        push!(build_derivatives, gradFn)
    end
    grad = (x) -> begin
        [gradFn(x) for gradFn in built_derivatives]
    end

    return grad
end

"""
Compute the divergence of a vector function evaluated at vector variable q
"""
function vector_divergence(V, q)
    div = 0
    for i in eachindex(q)
        div += ForwardDiff.gradient(x -> V(x)[i], q)[i]
    end
    return div
end

"""
Compute the matrix divergence of a matrix function evaluated at vector variable q
"""
function matrix_divergence(M, q)
    # The matrix divergence is defined as the column vector resulting from the vector divergence of each row
    div_M = zeros(size(q))
    for i in eachindex(q)
        # Take the divergence of the ith row of the matrix function
        div_M[i] = vector_divergence(x -> M(x)[i,:], q)
    end
    return div_M
end

end # module Calculus