# Half-half random split
function (::StandardResampling)( data::DataContainer, options::MethodsOptions, index_cache )
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
function (::TimeseriesResampling)( data::DataContainer, options::MethodsOptions, index_cache )
    block_size = options.timeseries_block_size

    start_ind = rand( index_cache[1:(end-block_size)] )
    end_ind = start_ind + block_size
    x_inds = @view index_cache[ start_ind:end_ind ]
    y_inds = setdiff( index_cache, x_inds )

    # x = @view data[ :, x_inds ]
    # y = @view data[ :, y_inds ]
    return x_inds, y_inds
end

function get_index_size( sampler::TimeseriesResampling, data, options )
    return length( data )
end
