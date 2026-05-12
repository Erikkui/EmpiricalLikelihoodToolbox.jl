struct ChamferDistanceDiff{T} <: AbstractChamferSummary
    neighbors::T
    highest_neighbor::Int
    diff_order::Int
    dt_obs::Float64
    summary_length::Int
end

function ChamferDistanceDiff( neighbors::Int, diff_order::Int, dt_obs::Float64 )
    return ChamferDistanceDiff( neighbors, maximum(neighbors), diff_order, dt_obs, 1 )
end

function ChamferDistanceDiff( neighbors::AbstractVector{<:Int}, diff_order::Int, dt_obs::Float64 )
    return ChamferDistanceDiff( collect(vec(neighbors)), maximum(neighbors), diff_order, dt_obs, length(neighbors) )
end

function calculate_summary_statistic!(  # To be used in target and bin initialization
    view_out::AbstractVector{Float64},
    summary_statistic::ChamferDistanceDiff,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    data::DataContainer,
    buffers::BufferContainer )

    diff_order = summary_statistic.diff_order

    data_X = @view data.differences[ diff_order ][ :, x_inds ]
    data_Y = @view data.differences[ diff_order ][ :, y_inds ]

    max_neighbor = summary_statistic.highest_neighbor

    chamfer_distance!( view_out, data_X, data_Y, k=max_neighbor )

    return nothing
end

function calculate_summary_statistic!(  # To be used in MCMC
    view_out::AbstractVector{Float64},
    summary_statistic::ChamferDistanceDiff,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    obs_data_all::DataContainer,
    sim_data_all::DataContainer,
    buffers::BufferContainer )

    diff_order = summary_statistic.diff_order

    data_X = @view obs_data_all.differences[ diff_order ][ :, x_inds ]
    Rsim = @view sim_data_all.differences[ diff_order ][ :, y_inds ]
    ytree = KDTree( Rsim )

    max_neighbor = summary_statistic.highest_neighbor

    chamfer_distance!( view_out, data_X, Rsim, ytree, k=max_neighbor )

    return nothing
end

function allocate_buffer( statistic::ChamferDistanceDiff, data::DataContainer )
    buffer = Vector{Float64}( undef, statistic.summary_length )
    return buffer
end

required_diff_order(stat::ChamferDistanceDiff) = stat.diff_order

function generate_stat_name( stat::ChamferDistanceDiff )
    return "ChamferDistanceDiff_k=$(stat.neighbors)_diff=$(stat.diff_order)"
end
