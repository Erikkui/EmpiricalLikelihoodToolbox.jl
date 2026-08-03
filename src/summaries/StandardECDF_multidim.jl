#------------Standard ECDF
struct StandardECDFMultiDimensional{B} <: ECDFMultiDimensionalSummary
    bins::B
    nbin::Int
    ndim::Int
    summary_length::Int
end

function StandardECDF( nbin::Int, ndim::Int)
    return StandardECDFMultiDimensional( nothing, nbin, ndim, ndim*nbin )
end



function calculate_summary_statistic!(  # To be used in target and bin initialization
    view_out::AbstractVector{Float64},
    summary_statistic::StandardECDFMultiDimensional,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    data::DataContainer,
    buffers::BufferContainer )

    nbins = summary_statistic.nbin
    bins = summary_statistic.bins

    data_X = @view data.observations[ :, x_inds ]

    start_ind = 1
    for (ii, row) in enumerate( eachrow(data_X) )
        end_ind = start_ind + summary_statistic.nbin - 1

        ecdf_view_out = @view view_out[ start_ind:end_ind ]
        empcdf!( ecdf_view_out, row, nbins, bins[ii] )

        start_ind += summary_statistic.nbin
    end

    return nothing
end

function calculate_summary_statistic!(  # To be used in MCMC
    view_out::AbstractVector{Float64},
    summary_statistic::StandardECDFMultiDimensional,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    obs_data_all::DataContainer,
    sim_data_all::DataContainer,
    buffers::BufferContainer )

    nbins = summary_statistic.nbin
    bins = summary_statistic.bins

    data_X = @view sim_data_all.observations[ :, x_inds ]

    start_ind = 1
    for (ii, row) in enumerate( eachrow(data_X) )
        end_ind = start_ind + summary_statistic.nbin - 1

        ecdf_view_out = @view view_out[ start_ind:end_ind ]
        empcdf!( ecdf_view_out, row, nbins, bins[ii] )

        start_ind += summary_statistic.nbin
    end

    return nothing
end

function get_bin_quantity( summary_statistic::StandardECDFMultiDimensional, data::DataContainer, inds_X, inds_Y )
    data_X = @view data.observations[ :, inds_X ]
    return data_X
end

function allocate_buffer( statistic::StandardECDFMultiDimensional, data::DataContainer )
    buffer = Vector{Float64}( undef, statistic.summary_length )
    return buffer
end

required_diff_order(stat::StandardECDFMultiDimensional) = 0

function generate_stat_name( stat::StandardECDFMultiDimensional )
    return "StandardECDF_k=$(stat.nbin)_ndim=$(stat.ndim)"
end
