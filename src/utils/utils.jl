# Calculate central differences of the data for given orders. Pads boundaries with nearest
# neighbor values to keep size constant.
function calculate_diffs(R, diff_order::Tuple{Vararg{Int}}, dt_obs::Float64)
    order_max = maximum(diff_order)
    R_diffs = Vector{Matrix{Float64}}(undef, order_max)

    # Do not mutate the input R. Use alternating buffers to avoid allocations.
    current_diff = copy(R)
    next_diff = similar(R)

    for ii in 1:order_max
        # Calculate central difference keeping size constant
        @inbounds for j in 2:(size(R, 2) - 1)
            for i in axes(R, 1)
                next_diff[i, j] = (current_diff[i, j+1] - current_diff[i, j-1]) / (2 * dt_obs)
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


# Same as above but mutates the provided buffer instead of allocating new matrices.
# This is used in the MCMC loop to avoid allocations.
function calculate_diffs!( diff_buffer, R, diff_order::Tuple{Vararg{Int}}, dt_obs::Float64)
    order_max = maximum(diff_order)

    # Do not mutate the input R. Use alternating buffers to avoid allocations.
    current_diff = diff_buffer[end-1]
    next_diff = diff_buffer[end]

    current_diff .= R
    for ii in 1:order_max
        # Calculate central difference keeping size constant
        @inbounds for j in 2:(size(R, 2) - 1)
            for i in axes(R, 1)
                next_diff[i, j] = (current_diff[i, j+1] - current_diff[i, j-1]) / (2 * dt_obs)
            end
        end

        # Pad boundaries (zero-order extrapolation / nearest neighbor)
        @inbounds for i in axes(R, 1)
            next_diff[i, 1] = next_diff[i, 2]
            next_diff[i, end] = next_diff[i, end-1]
        end

        # Store if the order is requested
        if ii in diff_order
            diff_buffer[ii] .= next_diff
        end

        # Swap buffers for the next iteration (pointers only, zero allocation)
        current_diff, next_diff = next_diff, current_diff
    end

    return nothing
end


# Inverse CDF function for y-axis based bin creation
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


# Recursive Welford's algorithm for online mean and covariance estimation.
# This is used in the adaptive MCMC loops to update the mean and covariance of the parameters without storing all samples.
function recursive_welford!( running_mean, running_cov, current_params, diff1, diff2, npar, ii )
    if ii == 1
        running_mean .= current_params
    else
        # diff1 = x_t - mean_{t-1}
        @inbounds for j in 1:npar
            diff1[j] = current_params[j] - running_mean[j]
        end

        # mean_t = mean_{t-1} + diff1 / t
        @inbounds for j in 1:npar
            running_mean[j] += diff1[j] / ii
        end

        # diff2 = x_t - mean_t
        @inbounds for j in 1:npar
            diff2[j] = current_params[j] - running_mean[j]
        end

        # C_t = ((t-2)/(t-1)) * C_{t-1} + (diff1 * diff2^T) / (t-1)
        weight1 = (ii - 2) / (ii - 1)
        weight2 = 1 / (ii - 1)
        @inbounds for col in 1:npar
            @inbounds for row in 1:npar
                running_cov[row, col] = weight1 * running_cov[row, col] + weight2 * diff1[row] * diff2[col]
            end
        end
    end
end


# Reconstruct a new model instance with updated parameters.
# This is a generic function that can be used for any model type.
function reconstruct( m::AbstractSimulationModel, new_params::AbstractVector{<:Real} )
    param_names = keys( get_params(m) )
    param_values = ntuple( ii -> new_params[ii], length(param_names) )
    new_param_tuple = NamedTuple{ param_names }( param_values )
    return setproperties(m, new_param_tuple)
end


# Get the initial state of a model.
initial_state( m::AbstractSimulationModel ) = m.x0

# Standardize summary statistics using provided mean and standard deviation.
standardize!( ss, standard_mean, standard_sd ) = ( ss - standard_mean ) / standard_sd


# Clean "show" method for printing summary statistic objects without printing the entire
# structure including type signatures. This works for string() and print() commands.
function Base.show( io::IO, summary::AbstractSummaryStatistic )
    # Get the name without the {Type} brackets
    summary_type = typeof(summary)
    type_name = nameof( summary_type )

    # Get all field values as a tuple

    field_values = ( something( getfield( summary, field ), "-") for field in fieldnames( summary_type ) )

    # Print it in the format: Name(val1, val2, ...)
    print( io, "$type_name(", join(field_values, ", "), ")")
end

# Same as above, but prints stuff prettily in the REPL
Base.show(io::IO, ::MIME"text/plain", x::AbstractSummaryStatistic) = show( io, x )
