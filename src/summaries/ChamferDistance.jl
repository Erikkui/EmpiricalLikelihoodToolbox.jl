#------------Chamfer Distance
struct ChamferDistance{T} <: AbstractChamferSummary
    neighbors::T
    highest_neighbor::Int
    summary_length::Int
end

function ChamferDistance( neighbors::Int )
    return ChamferDistance( neighbors, maximum(neighbors), 1 )
end

function ChamferDistance( neighbors::AbstractVector{<:Int} )
    return ChamferDistance( collect(vec(neighbors)), maximum(neighbors), length(neighbors) )
end

function calculate_summary_statistic!(  # To be used in target and bin initialization
    view_out::AbstractVector{Float64},
    summary_statistic::ChamferDistance,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    data::DataContainer,
    buffers::BufferContainer )

    data_X = @view data.observations[ :, x_inds ]
    data_Y = @view data.observations[ :, y_inds ]

    max_neighbor = summary_statistic.highest_neighbor

    chamfer_distance!( view_out, data_X, data_Y, k=max_neighbor )

    return nothing
end

function calculate_summary_statistic!(  # To be used in MCMC
    view_out::AbstractVector{Float64},
    summary_statistic::ChamferDistance,
    x_inds::AbstractVector{<:Integer},
    obs_data_all::DataContainer,
    sim_data_all::DataContainer,
    buffers::BufferContainer )

    data_X = @view obs_data_all.observations[ :, x_inds ]

    rsim_half = round( Int, size(sim_data_all.observations, 2) / 2 )
    Rsim = @view sim_data_all.observations[ :, rsim_half+1:end ]
    ytree = KDTree( Rsim )

    max_neighbor = summary_statistic.highest_neighbor

    # key = nameof( typeof(summary_statistic) )
    # buffer = buffers.summary_buffers[ key ]

    chamfer_distance!( view_out, data_X, Rsim, ytree, k=max_neighbor )

    return nothing
end

function allocate_buffer( statistic::ChamferDistance, data::DataContainer )
    buffer = Vector{Float64}( undef, statistic.summary_length )
    return buffer
end

required_diff_order(stat::ChamferDistance) = 0
