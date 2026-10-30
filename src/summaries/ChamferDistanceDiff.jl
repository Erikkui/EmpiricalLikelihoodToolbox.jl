"""
    ChamferDistanceDiff{T, BUF}

Chamfer distance computed on a numerical derivative of the data.

Identical to [`ChamferDistance`](@ref), except the point sets compared are the `diff_order`-th
central difference of the series rather than the raw series. `TargetData` trims `diff_order` points
from each end of the index cache, because central differences fabricate their boundary columns by
padding.

# Fields
- `neighbors::T`: The neighbour index (or indices) to evaluate. An `Int` or a vector of `Int`.
- `highest_neighbor::Int`: `maximum(neighbors)`, the depth the k-d tree search must reach.
- `diff_order::Int`: How many times to differentiate. `1` is the first derivative.
- `summary_length::Int`: `1` for a scalar `neighbors`, otherwise `length(neighbors)`.
- `buffer::BUF`: Scratch space (a `ChamferWorkspace`), filled in by `TargetData`. `nothing` until then.
- `name::String`: The statistic's name from `generate_stat_name`, set by the constructors.

# Examples
```julia
stat = ChamferDistanceDiff(1, 1)        # first derivative, nearest neighbour
stat = ChamferDistanceDiff([1, 3], 1)   # summary_length == 2
```

See also [`ChamferDistance`](@ref).
"""
struct ChamferDistanceDiff{T, BUF} <: AbstractChamferSummary
    neighbors::T
    highest_neighbor::Int
    diff_order::Int
    summary_length::Int
    buffer::BUF
    name::String
end

function ChamferDistanceDiff( neighbors::Int, diff_order::Int )
    return with_name( ChamferDistanceDiff( neighbors, maximum(neighbors), diff_order, 1, nothing, "" ) )
end

function ChamferDistanceDiff( neighbors::AbstractVector{<:Int}, diff_order::Int )
    return with_name( ChamferDistanceDiff( collect(vec(neighbors)), maximum(neighbors), diff_order, length(neighbors), nothing, "" ) )
end

function calculate_summary_statistic!(  # To be used in target and bin initialization
    view_out::AbstractVector{Float64},
    summary_statistic::ChamferDistanceDiff,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    data::DataContainer,
    buffers::BufferContainer )

    diff_order = summary_statistic.diff_order
    stat_buffers = summary_statistic.buffer

    data_X = @view data.differences[ diff_order ][ :, x_inds ]
    data_Y = @view data.differences[ diff_order ][ :, y_inds ]

    kvals = summary_statistic.neighbors

    chamfer_distance!( view_out, stat_buffers, data_X, data_Y, kvals )

    return nothing
end

function calculate_summary_statistic!(  # To be used in MCMC
    view_out::AbstractVector{Float64},
    summary_statistic::ChamferDistanceDiff,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    target::TargetData,
    sim_data_all::DataContainer,
    buffers::BufferContainer )

    diff_order = summary_statistic.diff_order
    stat_buffers = summary_statistic.buffer

    data_X = @view target.data.differences[ diff_order ][ :, x_inds ]
    Rsim = @view sim_data_all.differences[ diff_order ][ :, y_inds ]

    kvals = summary_statistic.neighbors

    chamfer_distance!( view_out, stat_buffers, data_X, Rsim, kvals )

    return nothing
end

function allocate_buffer( statistic::ChamferDistanceDiff, data::DataContainer )
    dimension = size( data.observations, 1 )
    return ChamferWorkspace( dimension, data.options.effective_N_obs, statistic.neighbors )
end

required_diff_order(stat::ChamferDistanceDiff) = stat.diff_order

function generate_stat_name( stat::ChamferDistanceDiff )
    return "ChamferDiff=$(stat.diff_order)_k=$(stat.neighbors)"
end

get_summary_length(stat::ChamferDistanceDiff, data::DataContainer) = stat.summary_length

function finalize_summary( stat::ChamferDistanceDiff, data::DataContainer, buffers::BufferContainer )
    return @set stat.buffer = buffers.summary_buffers[ Symbol( stat.name ) ]
end
