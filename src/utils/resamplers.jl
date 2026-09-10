abstract type AbstractResampler end

abstract type LengthPreservingSampler <: AbstractResampler end
abstract type LengthChangingSampler <: AbstractResampler end

# Resampling types

struct RademacherSplit <: LengthChangingSampler end

function (RS::RademacherSplit)( data::DataContainer, options::MethodsOptions, index_cache )
    ntot = length( index_cache )
    ntot_half = div( ntot, 2 )
    shuffle!( index_cache )

    x_inds = @view index_cache[ 1:ntot_half ]
    y_inds = @view index_cache[ (ntot_half+1):end ]
    # x = @view data[ :, x_inds ]
    # y = @view data[ :, y_inds ]

    return x_inds, y_inds
end

function get_index_size( sampler::RademacherSplit, data, options )
    return size( data, 2 )
end

function resample_sizes(
    sampler::RademacherSplit,
    ndata::Int
)
    nx = div(ndata, 2)
    ny = ndata - nx
    return nx, ny
end


# Time series resampling: sample a contiguous block from the data
Base.@kwdef struct ContiguosBlockSplit <: LengthChangingSampler
    timeseries_block_size::Int = 100
end

function (RS::ContiguosBlockSplit)( data::DataContainer, options::MethodsOptions, index_cache )
    block_size = RS.timeseries_block_size

    ndata = length( index_cache )
    last_start_ind = ndata - block_size + 1

    start_ind = rand( 1:last_start_ind )
    end_ind = start_ind + block_size - 1

    x_inds = @view index_cache[ start_ind:end_ind ]
    y_inds = setdiff( index_cache, x_inds )

    return x_inds, y_inds
end

function get_index_size( sampler::ContiguosBlockSplit, data, options )
    return size( data, 2 )
end

function resample_sizes(
    sampler::ContiguosBlockSplit,
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



# No resampling: returns all available data as both "x" and "y" sets, unchanged. Useful whenever a
# deterministic, non-resampled evaluation of a summary statistic is wanted, e.g. for the fixed
# observed-data reference statistic under BSL.
struct NoResampling <: LengthPreservingSampler end

function (RS::NoResampling)( data::DataContainer, options::MethodsOptions, index_cache )
    return index_cache, index_cache
end

function get_index_size( sampler::NoResampling, data, options )
    return size( data, 2 )
end

function resample_sizes(
    sampler::NoResampling,
    ndata::Int
)
    return ndata, ndata
end


# Standard bootstrap resampling
struct StandardBootstrap <: LengthPreservingSampler end

function (RS::StandardBootstrap)( data::DataContainer, options::MethodsOptions, index_cache )
    ntot = length( index_cache )
    x_inds = rand( 1:ntot, ntot )
    y_inds = rand( 1:ntot, ntot )

    return x_inds, y_inds
end

function get_index_size( sampler::StandardBootstrap, data, options )
    return size( data, 2 )
end

function resample_sizes(
    sampler::StandardBootstrap,
    ndata::Int
)
    return ndata, ndata
end
