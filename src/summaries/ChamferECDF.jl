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
    view_out::AbstractVector{Float64},
    summary_statistic::ChamferECDF,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    data::DataContainer,
    buffers::BufferContainer )

    nbins = summary_statistic.nbin
    bins = summary_statistic.bins
    kvals = summary_statistic.neighbors

    key = Symbol( generate_stat_name( summary_statistic ) )
    buffer = buffers.summary_buffers[ key ]

    # Loop for calculating chamfer distances from which an ecdf is finally calculated
    n_resample = data.options.training_resamplings
    chamfers = zeros( n_resample, length( kvals ) )
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
        empcdf!( view_jj, chamfers[:, jj], nbins, bins_jj )
    end
    return nothing
end

function calculate_summary_statistic!(      # To be used in MCMC
    view_out::AbstractVector{Float64},
    summary_statistic::ChamferECDF,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    obs_data_all::DataContainer,
    sim_data_all::DataContainer,
    buffers::BufferContainer )

    nbins = summary_statistic.nbin
    bins = summary_statistic.bins
    kvals = summary_statistic.neighbors

    R0 = obs_data_all.observations
    Rsim = @view sim_data_all.observations[ :, y_inds]
    ytree = KDTree( Rsim )


    key = Symbol( generate_stat_name( summary_statistic ) )
    buffer = buffers.summary_buffers[ key ]

    # Loop for calculating chamfer distances from which an ecdf is finally calculated
    resampler = obs_data_all.options.resampling_type
    n_resample = obs_data_all.options.training_resamplings
    chamfers = zeros( n_resample, length( kvals ) )
    for ii in 1:n_resample
        x_inds, _ = resampler( obs_data_all, obs_data_all.options, buffers.index_cache )
        data_X = @view R0[ :, x_inds ]
        chamfer_distance!( buffer, data_X, Rsim, ytree, kvals )
        chamfers[ii, :] .= buffer
    end
    for jj in eachindex( kvals )
        view_jj = @view view_out[ (jj-1)*nbins+1 : jj*nbins ]
        bins_jj = bins[jj]
        empcdf!( view_jj, chamfers[:, jj], nbins, bins_jj )
    end
    return nothing
end



function get_bin_quantity( summary_statistic::ChamferECDF, data::DataContainer, inds_X, inds_Y )
    kvals = summary_statistic.neighbors
    R0 = data.observations
    n_resampling = data.options.bins_resamplings
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
    buffer = Vector{Float64}( undef, length( statistic.neighbors ) )
    return buffer
end

required_diff_order(stat::ChamferECDF) = 0

function generate_stat_name( stat::ChamferECDF )
    return "ChamferECDF_k=$(stat.neighbors)"
end
