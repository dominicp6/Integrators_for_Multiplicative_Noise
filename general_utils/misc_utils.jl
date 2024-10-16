module MiscUtils
export init_x0, assert_isotropic_diagonal_diffusion, is_identity_diffusion, create_directory_if_not_exists, find_row_indices, remove_rows
using LinearAlgebra

function init_x0(x0; dim::Int = 1) 
    if x0 === nothing
        x0 = randn(dim)
    end
    if dim == 1
        x0 = x0[1]
    end
    return x0
end

"""
Assert that D is a diagonal, isotropic matrix
"""
function assert_isotropic_diagonal_diffusion(D) 
    D1 = (x,y) -> D(x,y)[1,1]
    D2 = (x,y) -> D(x,y)[2,2]
    Doff1 = (x,y) -> D(x,y)[1,2]
    Doff2 = (x,y) -> D(x,y)[2,1]
    @assert Doff1(0.123,-0.736) == Doff2(0.123,-0.736) == 0 "D must be diagonal"
    @assert D1(0.123,-0.736) == D2(0.123,-0.736) "D must be isotropic"
end

function is_identity_diffusion(D)
    if D(0,0) == D(1,1) == D(-0.2354345, 0.21267) == I
        identity_diffusion = true
    else
        identity_diffusion = false
    end

    return identity_diffusion
end

function create_directory_if_not_exists(dir_path)
    if !isdir(dir_path)
        mkpath(dir_path)
        @info "Created directory $dir_path"
    end
end


"""
Finds the row indices in an array/matrix where any element in that row is a 'nothing' type
"""
function find_row_indices(arr::Matrix, value::Int)
    indices = Int[]
    for i in 1:size(arr, 1)
        if any(x -> x === value, arr[i, :])
            push!(indices, i)
        end
    end
    return indices
end

"""
Removes rows in place in a matrix based on row indices to remove, returns the trimmed matrix
"""
function remove_rows(matrix::Matrix, indices::Vector{Int})
    # Sort the indices in descending order
    sorted_indices = sort(indices, rev=true)
    
    # Remove rows from the matrix
    for i in sorted_indices
        if i > size(matrix, 1) || i < 1
            error("Index out of range")
        end
        matrix = vcat(matrix[1:i-1, :], matrix[i+1:end, :])
    end
    return matrix
end

end # module MiscUtils