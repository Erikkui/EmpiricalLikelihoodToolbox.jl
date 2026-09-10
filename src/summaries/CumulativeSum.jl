struct CumulativeSum{S} <: AbstractCumulativeSumSummary
    contracting_window::Int
    summary_length::Int
    normalization_factor::S
end

function CumulativeSum( contracting_window::Int, Ndata::Int )
    summary_length = div( Ndata, contracting_window )
    return CumulativeSum( contracting_window, summary_length, NaN )
end

function calculate_summary_statistic!(      # To be used in target and bin initialization
    view_out::AbstractVector{Float64},
    summary_statistic::CumulativeSum,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    data::DataContainer,
    buffers::BufferContainer )

    contracting_window = summary_statistic.contracting_window
    R0 = data.observations

    key = Symbol( generate_stat_name( summary_statistic ) )
    buffer = buffers.summary_buffers[ key ]

    data_X = @view R0[ :, x_inds ]
    cumsum!( buffer, data_X, dims = 2 )
    contract!( view_out, buffer, contracting_window )

    view_out ./= summary_statistic.normalization_factor

    return nothing
end

function calculate_summary_statistic!(      # To be used in MCMC
    view_out::AbstractVector{Float64},
    summary_statistic::CumulativeSum,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    target::TargetData,
    sim_data_all::DataContainer,
    buffers::BufferContainer )

    contracting_window = summary_statistic.contracting_window
    Rsim = sim_data_all.observations

    key = Symbol( generate_stat_name( summary_statistic ) )
    buffer = buffers.summary_buffers[ key ]

    data_X = @view Rsim[ :, y_inds]
    cumsum!( buffer, data_X, dims = 2 )
    contract!( view_out, buffer, contracting_window )

    view_out ./= summary_statistic.normalization_factor

    return nothing
end


function allocate_buffer( statistic::CumulativeSum, data::DataContainer )
    Ndata = data.options.effective_N_obs
    N_sums = div( Ndata, statistic.contracting_window )
    buffer = zeros( N_sums )
    return buffer
end

get_summary_length(stat::CumulativeSum, data::DataContainer) = stat.summary_length


required_diff_order(stat::CumulativeSum) = 0

function generate_stat_name( stat::CumulativeSum )
    return "CumulativeSum_l=$(stat.contracting_window)"
end

function finalize_summary( stat::CumulativeSum, data::DataContainer, buffers::BufferContainer )
    inds = buffers.index_cache
    observations = @view data.observations[ :, inds ]
    sum_cumul = cumsum( observations, dims = 2 )
    sum_cumul_contracted = contract( sum_cumul, stat.contracting_window )
    stat = @set stat.normalization_factor = sum_cumul_contracted[end]
    return stat
end
