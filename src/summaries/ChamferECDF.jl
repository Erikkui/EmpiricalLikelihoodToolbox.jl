struct ChamferECDF{B, T} <: AbstractECDFSummary
    bins::B
    nbin::Int
    neighbors::T
    highest_neighbor::Int
    summary_length::Int
end

function ChamferECDF( nbin::Int, neighbors::Int )
    return ChamferECDF( nothing, nbin, [neighbors], neighbors, nbin )
end

function ChamferECDF( nbin::Int, neighbors::Vector{Int} )
    return ChamferECDF( nothing, nbin, neighbors, maximum(neighbors), length(neighbors)*nbin )
end

function calculate_summary_statistic!(      # To be used in target and bin initialization
    view_out::AbstractVector{Float64}, summary_statistic::ChamferECDF,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    data::DataContainer,
    buffers::BufferContainer )

    nbins = summary_statistic.nbin
    bins = summary_statistic.bins

    key = nameof( typeof(summary_statistic) )
    buffer = buffers.summary_buffers[ key ]

    # Loop for calculating chamfer distances from which an ecdf is finally calculated
    n_resample = data.options.training_resamplings
    chamfers = zeros( n_resample, summary_statistic.highest_neighbor )
    for ii in 1:n_resample
        x_inds, y_inds = data.options.resampling_type( data, data.options, buffers.indices_buffer )
        data_X = @view data.observations[ :, x_inds ]
        data_Y = @view data.observations[ :, y_inds ]
        chamfer_distance!( buffer, data_X, data_Y, k=summary_statistic.highest_neighbor )
        chamfers[ii, :] .= buffer
    end
    for jj in 1:summary_statistic.highest_neighbor
        view_jj = @view view_out[ (jj-1)*nbins+1 : jj*nbins ]
        bins_jj = bins[jj]
        empcdf!( view_jj, chamfers[:, jj], nbins, bins_jj )
    end
    return nothing
end

function calculate_summary_statistic!(      # To be used in target and bin initialization
    view_out::AbstractVector{Float64}, summary_statistic::ChamferECDF,
    y_inds::AbstractVector{<:Integer},
    obs_data_all::DataContainer,
    sim_data_all::DataContainer,
    buffers::BufferContainer )

    R0 = obs_data_all.observations
    rsim_half = round( Int, size(sim_data_all.observations, 2) / 2 )
    Rsim = @view sim_data_all.observations[ :, rsim_half+1:end ]
    ytree = KDTree( Rsim )

    nbins = summary_statistic.nbin
    bins = summary_statistic.bins

    key = nameof( typeof(summary_statistic) )
    buffer = buffers.summary_buffers[ key ]

    # Loop for calculating chamfer distances from which an ecdf is finally calculated
    n_resample = obs_data_all.options.training_resamplings
    chamfers = zeros( n_resample, summary_statistic.highest_neighbor )
    for ii in 1:n_resample

        x_inds, _ = obs_data_all.options.resampling_type( obs_data_all, obs_data_all.options, buffers.indices_buffer )
        data_X = @view R0[ :, x_inds ]
        chamfer_distance!( buffer, data_X, Rsim, ytree, k=summary_statistic.highest_neighbor )
        chamfers[ii, :] .= buffer
    end
    for jj in 1:summary_statistic.highest_neighbor
        view_jj = @view view_out[ (jj-1)*nbins+1 : jj*nbins ]
        bins_jj = bins[jj]
        empcdf!( view_jj, chamfers[:, jj], nbins, bins_jj )
    end
    return nothing
end



function get_bin_quantity( summary_statistic::ChamferECDF, data::DataContainer, inds_X, inds_Y )
    R0 = data.observations
    n_resampling = data.options.bins_resamplings
    chamfers = Matrix{Float64}( undef, n_resampling, summary_statistic.highest_neighbor )
    indices = collect( 1:size(R0, 2) )
    for ii in 1:n_resampling
        x_inds, y_inds = data.options.resampling_type( data, data.options, indices )
        data_X = @view R0[ :, x_inds ]
        data_Y = @view R0[ :, y_inds ]
        chamfers[ii, :] = chamfer_distance( data_X, data_Y, k=summary_statistic.highest_neighbor )
    end
    return chamfers
end

function allocate_buffer( statistic::ChamferECDF, data::DataContainer )
    buffer = Vector{Float64}( undef, statistic.highest_neighbor )
    return buffer
end

required_diff_order(stat::ChamferECDF) = 0
