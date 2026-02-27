# Calculate central differences of the data for given orders. Pads boundaries with nearest
# neighbor values to keep size constant.
function calculate_diffs(R, diff_order::Tuple{Vararg{Int}}, dt)
    order_max = maximum(diff_order)
    R_diffs = Vector{Matrix{Float64}}(undef, order_max)

    # Do not mutate the input R. Use alternating buffers to avoid allocations.
    current_diff = copy(R)
    next_diff = similar(R)

    for ii in 1:order_max
        # Calculate central difference keeping size constant
        @inbounds for j in 2:(size(R, 2) - 1)
            for i in axes(R, 1)
                next_diff[i, j] = (current_diff[i, j+1] - current_diff[i, j-1]) / (2 * dt)
            end
        end

        # Pad boundaries (zero-order extrapolation / nearest neighbor)
        @inbounds for i in axes(R, 1)
            next_diff[i, 1] = next_diff[i, 2]
            next_diff[i, end] = next_diff[i, end-1]
        end

        # Store if the order is requested
        if ii in diff_order
            R_diffs[ii] = copy(next_diff)
        else
            R_diffs[ii] = Matrix{Float64}(undef, 0, 0)
        end

        # Swap buffers for the next iteration (pointers only, zero allocation)
        current_diff, next_diff = next_diff, current_diff
    end

    return R_diffs
end


# Inverse CDF function
function invcdf(x, cdf, nr, cont=1)::Vector{Float64}
    xi = x[:]
    cdfi = cdf[:]
    rr = range(1.01 * minimum(cdf), stop=0.99 * maximum(cdf), length=nr)
    r = zeros( Float64, nr )
    n_xi = length(xi)

    for i in 1:nr

        arg = sum( rr[i] .> cdfi ) + 1
        ind = min( arg, n_xi )

        if ind == 1 || cont == 2
            r[i] = xi[ind]
        else
            r[i] = ( xi[ind] - xi[ind-1] ) / ( cdfi[ind] - cdfi[ind-1] ) *
                   ( rr[i] - cdfi[ind-1] ) + xi[ind-1]
        end
    end

    return r
end
