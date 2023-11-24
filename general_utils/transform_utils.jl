module TransformUtils
include("../general_utils/diffusion_tensors.jl")
using FHist, StatsBase
import .DiffusionTensors: Dconst1D
export increment_g_counts, time_transformed_potential, increment_I_counts, increment_g_counts2D, transform_potential_and_diffusion

function time_transformed_potential(x, V, D, sigma)
    return V(x) - sigma * log(D(x))
end

function scale_range(range, transform)
    return transform.(range)
end

"""
De-bias a histogram by transforming the bin edges according to a given function
"""
function debias_hist(hist, transform)
    bin_edges = binedges(hist)
    new_hist_edges = scale_range(bin_edges, transform)
    debias_hist = Hist1D(Histogram(new_hist_edges, bincounts(hist)))

    return debias_hist
end

function increment_g_counts(q_chunk, D, bin_boundaries, ΣgI, Σg)
    # Used for reweighting time-transformed trajectories (see paper for details)
    g(x) = 1/D(x)
    
    # Iterate through trajectory points and assign to corresponding bin
    for q in q_chunk
        Σg += g(q)

        # Find the index of the histogram bin that q is in
        bin_index = searchsortedfirst(bin_boundaries, q) - 1
        # only count points that are in the domain of the specified bins
        if bin_index != 0 && bin_index != length(bin_boundaries)
            ΣgI[bin_index] += g(q)
        end
    end

    return ΣgI, Σg
end


function increment_g_counts2D(q_chunk, D, x_bins, y_bins, ΣgI, Σg, R)
    # Used for reweighting time-transformed trajectories (see paper for details)
    g(x,y) = 1/D(x,y)^2
    
    # Iterate through trajectory points and assign to corresponding bin
    for q in eachcol(q_chunk)
        Σg += g(q[1], q[2])

        # Find the index of the histogram bin that q is in
        bin_index1 = searchsortedfirst(x_bins, R[1,1]*q[1]+R[1,2]*q[2]) - 1
        bin_index2 = searchsortedfirst(y_bins, R[2,1]*q[1]+R[2,2]*q[2]) - 1
        # only count points that are in the domain of the specified bins
        if bin_index1 != 0 && bin_index1 != length(x_bins) && bin_index2 != 0 && bin_index2 != length(y_bins)
            ΣgI[bin_index1, bin_index2] += g(q[1], q[2])
        end
    end

    return ΣgI, Σg
end

"""
Used for reweighting space-transformed trajectories (see paper for details)
"""
function increment_I_counts(q_chunk, x_of_y, bin_boundaries, ΣI)
    # Iterate through trajectory points and assign to corresponding bin
    for q in q_chunk
        # Find the index of the histogram bin that q is in
        bin_index = searchsortedfirst(bin_boundaries, x_of_y(q)) - 1
        # only count points that are in the domain of the specified bins
        if bin_index != 0 && bin_index != length(bin_boundaries)
            ΣI[bin_index] += 1
        end
    end

    return ΣI
end

function transform_potential_and_diffusion(original_V, original_D, sigma, time_transform, space_transform, x_of_y)
    @assert !(time_transform && space_transform) "Not supported to run both time and space transforms"
    
    if time_transform
        V = x -> original_V(x) - sigma^2 * log(original_D(x)) / 2
        D = Dconst1D
    end

    if space_transform
        @assert x_of_y !== nothing "x_of_y must be defined for space-transformed integrators"
        V = y -> original_V(x_of_y(y)) - 0.25 * sigma^2 * log(original_D(x_of_y(y)))
        D = Dconst1D
    end

    if !(time_transform || space_transform)
        V = original_V
        D = original_D
    end

    return V, D
end

end # module