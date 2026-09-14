"""
    ChamferECDF{B, T, BUF}

Empirical CDF of repeatedly resampled Chamfer distances.

Where [`ChamferDistance`](@ref) returns a single distance per neighbour, this draws
`dists_for_ecdf` fresh resamplings of the data, computes a Chamfer distance for each, and returns
the ECDF of that collection over `nbin` bins. The result describes the *distribution* of the Chamfer
distance under the resampler rather than one realization of it, giving a summary of
`length(neighbors) * nbin` values.

!!! warning "Cost"
    This resamples internally, on top of whatever resampling the caller is already doing.
    Evaluating it once performs `dists_for_ecdf` (fixed at 1000) resampling operations, so building
    a target with `training_resamplings = 1000` costs on the order of a million. Budget accordingly.

Unlike the other two-set summaries, the `x_inds`/`y_inds` handed in by the caller are ignored: this
summary draws its own splits from `MethodsOptions.resampling_type`.

# Fields
- `bins::B`: One vector of bin edges per requested neighbour. If `nothing`, calculated from the data.
- `nbin::Int`: The number of bins per neighbour.
- `neighbors::T`: The neighbour indices to use, always stored as a vector.
- `highest_neighbor::Int`: `maximum(neighbors)`, the depth the k-d tree search must reach.
- `summary_length::Int`: Equal to `length(neighbors) * nbin`.
- `dists_for_ecdf::Int`: How many Chamfer distances to draw for the ECDF. Fixed at `1000`.
- `buffer::BUF`: Scratch space, filled in by `TargetData`. `nothing` until then.

# Examples
```julia
stat = ChamferECDF(10, 1)        # summary_length == 10
stat = ChamferECDF(10, [1, 3])   # summary_length == 20
```

!!! warning "Supplied bin edges are currently discarded"
    `TargetData` recomputes bin edges from the data for every eCDF-based summary and overwrites
    whatever was passed to the constructor, so the `bins` constructor has no effect on a full
    pipeline run. It does take effect when the summary is evaluated directly.

See also [`ChamferDistance`](@ref), [`ID`](@ref).
"""
struct ChamferECDF{B, T, BUF} <: AbstractECDFSummary
    bins::B
    nbin::Int
    neighbors::T
    highest_neighbor::Int
    summary_length::Int
    dists_for_ecdf::Int
    buffer::BUF
end

function ChamferECDF( nbin::Int, neighbors::Int )
    return ChamferECDF( nothing, nbin, [neighbors], neighbors, nbin, 1000, nothing )
end

function ChamferECDF( nbin::Int, neighbors::Vector{Int} )
    return ChamferECDF( nothing, nbin, neighbors, maximum(neighbors), length(neighbors)*nbin, 1000, nothing )
end

function ChamferECDF( bins::AbstractVector{<:Real}, neighbors::Int )
    bins_vec = collect( vec(bins) )
    nbin = length( bins_vec )
    return ChamferECDF( [bins_vec], nbin, [neighbors], neighbors, nbin, 1000, nothing )
end

function ChamferECDF( bins::AbstractVector{<:AbstractVector{<:Real}}, neighbors::Vector{Int} )
    length(bins) == length(neighbors) || throw( ArgumentError(
        "bins and neighbors must have the same length, got $(length(bins)) and $(length(neighbors))" ) )

    bins_vecs = [ collect(vec(b)) for b in bins ]
    nbin = length( bins_vecs[1] )
    all( length(b) == nbin for b in bins_vecs ) || throw( ArgumentError(
        "all bins vectors must have the same length" ) )

    return ChamferECDF( bins_vecs, nbin, neighbors, maximum(neighbors), length(neighbors)*nbin, 1000, nothing )
end

function calculate_summary_statistic!(      # To be used in target and bin initialization
    view_out::AbstractVector{Float64},
    summary_statistic::ChamferECDF,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    data::DataContainer,
    buffers::BufferContainer )

    empcdf! = data.options.ecdf_function

    nbins = summary_statistic.nbin
    bins = summary_statistic.bins
    kvals = summary_statistic.neighbors

    stat_buffer = summary_statistic.buffer
    buffer = stat_buffer.chamfer
    chamfers = stat_buffer.chamfers

    # Loop for calculating chamfer distances from which an ecdf is finally calculated
    n_resample = summary_statistic.dists_for_ecdf
    for ii in 1:n_resample
        x_inds, y_inds = data.options.resampling_type( data, data.options, buffers.index_cache )
        data_X = @view data.observations[ :, x_inds ]
        data_Y = @view data.observations[ :, y_inds ]
        chamfer_distance!( buffer, data_X, data_Y, kvals )
        chamfers[ii, :] .= buffer
    end
    for jj in eachindex( kvals )
        view_jj = @view view_out[ (jj-1)*nbins+1 : jj*nbins ]
        bins_jj = bins[jj]
        empcdf!( view_jj, @view( chamfers[:, jj] ), nbins, bins_jj )
    end
    return nothing
end

function calculate_summary_statistic!(      # To be used in MCMC
    view_out::AbstractVector{Float64},
    summary_statistic::ChamferECDF,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    target::TargetData,
    sim_data_all::DataContainer,
    buffers::BufferContainer )

    empcdf! = target.data.options.ecdf_function

    nbins = summary_statistic.nbin
    bins = summary_statistic.bins
    kvals = summary_statistic.neighbors

    R0 = target.data.observations
    Rsim = sim_data_all.observations[ :, y_inds ]
    ytree = KDTree( Rsim )

    stat_buffer = summary_statistic.buffer
    buffer = stat_buffer.chamfer
    chamfers = stat_buffer.chamfers

    # Loop for calculating chamfer distances from which an ecdf is finally calculated
    resampler = target.data.options.resampling_type
    n_resample = summary_statistic.dists_for_ecdf
    for ii in 1:n_resample
        x_inds, _ = resampler( target.data, target.data.options, buffers.index_cache )
        data_X = @view R0[ :, x_inds ]
        chamfer_distance!( buffer, data_X, Rsim, ytree, kvals )
        chamfers[ii, :] .= buffer
    end

    # Calculate the empirical CDF for each k-value
    for jj in eachindex( kvals )
        view_jj = @view view_out[ (jj-1)*nbins+1 : jj*nbins ]
        bins_jj = bins[jj]
        empcdf!( view_jj, @view( chamfers[:, jj] ), nbins, bins_jj )
    end
    return nothing
end



function get_bin_quantity( summary_statistic::ChamferECDF, data::DataContainer, inds_X, inds_Y )
    kvals = summary_statistic.neighbors
    R0 = data.observations
    n_resampling = round( Int, 2000/data.options.bins_resamplings )
    chamfers = Matrix{Float64}( undef, n_resampling, length( kvals ) )
    indices = collect( 1:size(R0, 2) )
    for ii in 1:n_resampling
        x_inds, y_inds = data.options.resampling_type( data, data.options, indices )
        data_X = @view R0[ :, x_inds ]
        data_Y = @view R0[ :, y_inds ]
        chamfers[ii, :] = chamfer_distance( data_X, data_Y, kvals )
    end
    return chamfers
end


function allocate_buffer( statistic::ChamferECDF, data::DataContainer )
    nk = length( statistic.neighbors )
    return (
        chamfer = Vector{Float64}( undef, nk ),
        chamfers = Matrix{Float64}( undef, statistic.dists_for_ecdf, nk ),
        )
end

required_diff_order(stat::ChamferECDF) = 0

function generate_stat_name( stat::ChamferECDF )
    return "ChamferECDF_k=$(stat.neighbors)"
end

get_summary_length(stat::ChamferECDF, data::DataContainer) = stat.summary_length

function finalize_summary( stat::ChamferECDF, data::DataContainer, buffers::BufferContainer )
    return @set stat.buffer = buffers.summary_buffers[ Symbol( generate_stat_name( stat ) ) ]
end
