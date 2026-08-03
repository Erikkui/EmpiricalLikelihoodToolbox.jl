function empcdf( data::AbstractArray{<:Real}, nbins::Int, bins::Vector{Float64} )

    n_data = length(data)
    cdf_out = zeros( nbins )

    # Precompute the inverse to use multiplication instead of division in the loop
    # Multiplication is significantly faster on the CPU
    inv_n = 1.0 / n_data

    @inbounds for ii in 1:nbins
        b = bins[ii]
        c = 0

        @inbounds @simd for jj in eachindex(data)
            c += data[jj] <= b
        end

        cdf_out[ii] = c * inv_n
    end

    return cdf_out
end

function empcdf!(out_view::AbstractVector, data, nbins::Int, bins::AbstractVector)
    n_data = length(data)

    # Precompute the inverse to use multiplication instead of division in the loop
    # Multiplication is significantly faster on the CPU
    inv_n = 1.0 / n_data

    @inbounds for ii in 1:nbins
        b = bins[ii]
        c = 0

        @inbounds @simd for jj in eachindex(data)
            c += data[jj] <= b
        end

        out_view[ii] = c * inv_n
    end

    return nothing
end
