# function empcdf( data::AbstractArray{<:Real}, nbins::Int, bins::Vector{Float64} )

#     n_data = length(data)
#     cdf_out = zeros( nbins )

#     # Precompute the inverse to use multiplication instead of division in the loop
#     # Multiplication is significantly faster on the CPU
#     inv_n = 1.0 / n_data

#     @inbounds for ii in 1:nbins
#         b = bins[ii]
#         c = 0

#         @inbounds @simd for jj in eachindex(data)
#             c += data[jj] <= b
#         end

#         cdf_out[ii] = c * inv_n
#     end

#     return cdf_out
# end

# function empcdf!(out_view::AbstractVector, data, nbins::Int, bins::AbstractVector)
#     n_data = length(data)

#     # Precompute the inverse to use multiplication instead of division in the loop
#     # Multiplication is significantly faster on the CPU
#     inv_n = 1.0 / n_data

#     @inbounds for ii in 1:nbins
#         b = bins[ii]
#         c = 0

#         @inbounds @simd for jj in eachindex(data)
#             c += data[jj] <= b
#         end

#         out_view[ii] = c * inv_n
#     end

#     return nothing
# end
using SpecialFunctions: erf
using Statistics: std

function empcdf(
                       data::AbstractVector,
                          nbins::Int,
                       bins::AbstractVector)

    n = length(data)
    out_view = zeros( nbins )

    if n == 0
        fill!(out_view, 0.0)
        return out_view
    end

    s = std(data, corrected=false)
    h = 1.06 * (s + 1e-12) * n^(-1/5) + 1e-12
    inv_n = 1.0 / n
    inv_sqrt2 = inv(sqrt(2.0))

    @inbounds for ii in eachindex(bins)
        b = bins[ii]

        acc = 0.0

        @inbounds @simd for jj in eachindex(data)
            z = (b - data[jj]) / h
            acc += 0.5 * (1.0 + erf(z * inv_sqrt2))
        end

        out_view[ii] = acc * inv_n
    end

    return out_view
end

function empcdf!(out_view::AbstractVector,
                       data::AbstractVector,
                          nbins::Int,
                       bins::AbstractVector)

    n = length(data)

    if n == 0
        fill!(out_view, 0.0)
        return nothing
    end

    s = std(data, corrected=false)
    h = 1.06 * (s + 1e-12) * n^(-1/5) + 1e-12
    inv_n = 1.0 / n
    inv_sqrt2 = inv(sqrt(2.0))

    @inbounds for ii in eachindex(bins)
        b = bins[ii]

        acc = 0.0

        @inbounds @simd for jj in eachindex(data)
            z = (b - data[jj]) / h
            acc += 0.5 * (1.0 + erf(z * inv_sqrt2))
        end

        out_view[ii] = acc * inv_n
    end

    return nothing
end
