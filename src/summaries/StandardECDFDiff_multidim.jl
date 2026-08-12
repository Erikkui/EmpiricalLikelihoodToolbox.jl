#------------Standard ECDF
struct StandardECDFDiffMultiDimensional{B} <: ECDFMultiDimensionalSummary
    bins::B
    nbin::Int
    ndim::Int
    dt_obs::Float64
    diff_order::Int
    summary_length::Int
end

function StandardECDFDiff( nbin::Int, ndim::Int, diff_order::Int, dt_obs::Float64 )
    if ndim == 1
        return StandardECDFDiff( nothing, nbin, dt_obs, diff_order, nbin )
    else
        return StandardECDFDiffMultiDimensional( nothing, nbin, ndim, dt_obs, diff_order, ndim*nbin )
    end
end



function calculate_summary_statistic!(  # To be used in target and bin initialization
    view_out::AbstractVector{Float64},
    summary_statistic::StandardECDFDiffMultiDimensional,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    data::DataContainer,
    buffers::BufferContainer )

    diff_order = summary_statistic.diff_order

    nbins = summary_statistic.nbin
    bins = summary_statistic.bins

    data_X = @view data.differences[ diff_order ][ :, x_inds ]

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
    summary_statistic::StandardECDFDiffMultiDimensional,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    obs_data_all::DataContainer,
    sim_data_all::DataContainer,
    buffers::BufferContainer )

    use_ecdf_sampling = obs_data_all.options.use_ecdf_sampling

    nbins = summary_statistic.nbin
    bins = summary_statistic.bins
    diff_order = summary_statistic.diff_order

    Rsim_diff = sim_data_all.differences[ diff_order ]

    ndata = size( Rsim_diff, 2 )

    start_ind = 1
    if use_ecdf_sampling
        data_X = @view Rsim_diff[ :, : ]
        yax_values = rand( ndata )
        for (ii, row) in enumerate( eachrow(data_X) )
            end_ind = start_ind + summary_statistic.nbin - 1

            xmin, xmax = minimum(row), maximum(row)
            bins_dense_temp = range( 1.01*xmin, 0.99*xmax, length = ndata ) |> collect
            ecdf_view_out = @view view_out[ start_ind:end_ind ]

            ecdf_sim = empcdf( row, nbins, bins_dense_temp )
            data_X_new = invcdf( yax_values, ecdf_sim, nbins, 1 )
            empcdf!( ecdf_view_out, data_X_new, nbins, bins[ii] )

            start_ind += summary_statistic.nbin
        end
    else
        data_X = @view Rsim_diff[ :, x_inds ]
        for (ii, row) in enumerate( eachrow(data_X) )
            end_ind = start_ind + summary_statistic.nbin - 1

            ecdf_view_out = @view view_out[ start_ind:end_ind ]
            empcdf!( ecdf_view_out, row, nbins, bins[ii] )

            start_ind += summary_statistic.nbin
        end
    end

    return nothing
end

function get_bin_quantity( summary_statistic::StandardECDFDiffMultiDimensional, data::DataContainer, inds_X, inds_Y )
    diff_ind = summary_statistic.diff_order
    data_X = data.differences[ diff_ind ]
    data_X =  data_X[ :, inds_X ]
    return data_X
end

function allocate_buffer( statistic::StandardECDFDiffMultiDimensional, data::DataContainer )
    buffer = Vector{Float64}( undef, statistic.summary_length )
    return buffer
end

required_diff_order(stat::StandardECDFDiffMultiDimensional) = stat.diff_order

function generate_stat_name( stat::StandardECDFDiffMultiDimensional )
    return "StandardECDFDiff_k=$(stat.nbin)_ndim=$(stat.ndim)_diff_order=$(stat.diff_order)"
end
