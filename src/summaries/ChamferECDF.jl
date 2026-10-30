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

Unlike the other two-set summaries, the `x_inds`/`y_inds` handed in by the caller are ignored in
training: this summary draws its own splits from `MethodsOptions.resampling_type`, on a private copy
of the index cache, so the caller's split (shared with the other summaries of a
`JointSummaryStatistics`) is left untouched. During MCMC the simulated set is the caller's `y_inds`.

# Fields
- `bins::B`: One vector of bin edges per requested neighbour. If `nothing`, calculated from the data.
- `nbin::Int`: The number of bins per neighbour.
- `neighbors::T`: The neighbour indices to use, always stored as a vector.
- `highest_neighbor::Int`: `maximum(neighbors)`, the depth the k-d tree search must reach.
- `summary_length::Int`: Equal to `length(neighbors) * nbin`.
- `dists_for_ecdf::Int`: How many Chamfer distances to draw for the ECDF. Fixed at `1000`.
- `buffer::BUF`: Scratch space, filled in by `TargetData`. `nothing` until then.
- `name::String`: The statistic's name from `generate_stat_name`, set by the constructors.

# Examples
```julia
stat = ChamferECDF(10, 1)        # summary_length == 10
stat = ChamferECDF(10, [1, 3])   # summary_length == 20
```

See also [`ChamferDistance`](@ref), [`ID`](@ref).
"""
struct ChamferECDF{B, T, BUF} <: ChamferECDFSummary
    bins::B
    nbin::Int
    neighbors::T
    highest_neighbor::Int
    summary_length::Int
    dists_for_ecdf::Int
    buffer::BUF
    name::String
end

function ChamferECDF( nbin::Int, neighbors::Int )
    return with_name( ChamferECDF( nothing, nbin, [neighbors], neighbors, nbin, 1000, nothing, "" ) )
end

function ChamferECDF( nbin::Int, neighbors::Vector{Int} )
    return with_name( ChamferECDF( nothing, nbin, neighbors, maximum(neighbors), length(neighbors)*nbin, 1000, nothing, "" ) )
end

function ChamferECDF( bins::AbstractVector{<:Real}, neighbors::Int )
    bins_vec = collect( vec(bins) )
    nbin = length( bins_vec )
    return with_name( ChamferECDF( [bins_vec], nbin, [neighbors], neighbors, nbin, 1000, nothing, "" ) )
end

function ChamferECDF( bins::AbstractVector{<:AbstractVector{<:Real}}, neighbors::Vector{Int} )
    length(bins) == length(neighbors) || throw( ArgumentError(
        "bins and neighbors must have the same length, got $(length(bins)) and $(length(neighbors))" ) )

    bins_vecs = [ collect(vec(b)) for b in bins ]
    nbin = length( bins_vecs[1] )
    all( length(b) == nbin for b in bins_vecs ) || throw( ArgumentError(
        "all bins vectors must have the same length" ) )

    return with_name( ChamferECDF( bins_vecs, nbin, neighbors, maximum(neighbors), length(neighbors)*nbin, 1000, nothing, "" ) )
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
    workspace = stat_buffer.workspace
    index_cache = stat_buffer.index_cache

    # Loop for calculating chamfer distances from which an ecdf is finally calculated
    n_resample = summary_statistic.dists_for_ecdf
    for ii in 1:n_resample
        x_inds, y_inds = data.options.resampling_type( data, data.options, index_cache )
        data_X = @view data.observations[ :, x_inds ]
        data_Y = @view data.observations[ :, y_inds ]
        chamfer_distance!( buffer, workspace, data_X, data_Y, kvals )
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
    Rsim = @view sim_data_all.observations[ :, y_inds ]

    stat_buffer = summary_statistic.buffer
    buffer = stat_buffer.chamfer
    chamfers = stat_buffer.chamfers
    workspace = stat_buffer.workspace
    index_cache = stat_buffer.index_cache
    obs_to_sim = stat_buffer.obs_to_sim

    # The simulated set is the same for every resample: build its tree once, and look up the
    # observed-to-simulated distances of every observed point once (each resample only picks
    # which of them to sum)
    _load_set_B!( workspace, Rsim )
    _knn_distance_table!( obs_to_sim, workspace, R0, index_cache, kvals )

    # Loop for calculating chamfer distances from which an ecdf is finally calculated
    resampler = target.data.options.resampling_type
    n_resample = summary_statistic.dists_for_ecdf
    for ii in 1:n_resample
        x_inds, _ = resampler( target.data, target.data.options, index_cache )
        _chamfer_distance_from_table!( buffer, workspace, obs_to_sim, R0, x_inds, kvals )
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
    dimension, n_columns = size( data.observations )
    effective_nobs = data.options.effective_N_obs
    return (
        chamfer = Vector{Float64}( undef, nk ),
        chamfers = Matrix{Float64}( undef, statistic.dists_for_ecdf, nk ),
        workspace = ChamferWorkspace( dimension, effective_nobs, statistic.neighbors ),
        obs_to_sim = Matrix{Float64}( undef, nk, n_columns ),   # indexed by observed column
        index_cache = Vector{Int}( undef, effective_nobs ),     # private copy, filled by finalize_summary
        )
end

required_diff_order(stat::ChamferECDF) = 0

function generate_stat_name( stat::ChamferECDF )
    return "ChamferECDF_k=$(stat.neighbors)"
end

get_summary_length(stat::ChamferECDF, data::DataContainer) = stat.summary_length

function finalize_summary( stat::ChamferECDF, data::DataContainer, buffers::BufferContainer )
    stat_buffer = buffers.summary_buffers[ Symbol( stat.name ) ]
    # The inner resampling uses its own index cache: RademacherSplit shuffles the cache in place,
    # which would change the split the caller passed to the other summaries of a Joint
    copyto!( stat_buffer.index_cache, buffers.index_cache )
    return @set stat.buffer = stat_buffer
end
