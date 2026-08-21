# Half-half random split
function (RS::StandardResampling)( data::DataContainer, options::MethodsOptions, index_cache )
    ntot = size( data.observations, 2 )
    ntot_half = div( ntot, 2 )
    shuffle!( index_cache )

    x_inds = @view index_cache[1:ntot_half]
    y_inds = @view index_cache[(ntot_half+1):end]
    # x = @view data[ :, x_inds ]
    # y = @view data[ :, y_inds ]

    return x_inds, y_inds
end

function get_index_size( sampler::StandardResampling, data, options )
    return size( data, 2 )
end



# Time series resampling: sample a contiguous block from the data
function (RS::TimeseriesResampling)( data::DataContainer, options::MethodsOptions, index_cache )
    block_size = RS.timeseries_block_size

    start_ind = rand( index_cache[1:(end-block_size)] )
    end_ind = start_ind + block_size - 1
    x_inds = @view index_cache[ start_ind:end_ind ]
    y_inds = setdiff( index_cache, x_inds )

    # x = @view data[ :, x_inds ]
    # y = @view data[ :, y_inds ]
    return x_inds, y_inds
end

function get_index_size( sampler::TimeseriesResampling, data, options )
    return size( data, 2 )
end



# Random inverse cdf resampling using only trained mean
function (RS::InverseCDFResampling  )( data::DataContainer, options::MethodsOptions, index_cache )

    y_inds = Vector{Int}(undef, 0)
    if RS.training_phase
        x_inds = @view index_cache[:]
    else
        x_inds = @view rand!( index_cache )[:]
    end

    return x_inds, y_inds
end


function resample_sizes(
    sampler::StandardResampling,
    ndata::Int
)
    nx = div(ndata, 2)
    ny = ndata - nx
    return nx, ny
end

function resample_sizes(
    sampler::TimeseriesResampling,
    ndata::Int
)
    nx = sampler.timeseries_block_size
    1 <= nx < ndata ||
        throw(ArgumentError(
            "timeseries_block_size must satisfy " *
            "1 <= block_size < ndata"
        ))
    ny = ndata - nx
    return nx, ny
end
