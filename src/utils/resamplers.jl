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
Base.@kwdef struct ContiguousBlockSplit <: LengthChangingSampler
    timeseries_block_size::Int = 100
end

function (RS::ContiguousBlockSplit)( data::DataContainer, options::MethodsOptions, index_cache )
    block_size = RS.timeseries_block_size

    ndata = length( index_cache )
    last_start_ind = ndata - block_size + 1

    start_ind = rand( 1:last_start_ind )
    end_ind = start_ind + block_size - 1

    x_inds = @view index_cache[ start_ind:end_ind ]
    y_inds = setdiff( index_cache, x_inds )

    return x_inds, y_inds
end

function get_index_size( sampler::ContiguousBlockSplit, data, options )
    return size( data, 2 )
end

function resample_sizes(
    sampler::ContiguousBlockSplit,
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

    x_inds = rand( index_cache, ntot )
    y_inds = rand( index_cache, ntot )

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


# Moving block bootstrap: builds each of the x/y sets by concatenating random
# contiguous blocks of length `block_size` (sampled with replacement, ie. blocks may overlap or
# repeat) until reaching length `ndata` (the last block is truncated to fit exactly). Unlike
# ContiguousBlockSplit - which takes a single block as "x" and everything else as "y", an asymmetric
# length-changing split - this is a LengthPreservingSampler: both x_inds and y_inds always have
# length exactly `ndata`, the same size used everywhere else (training and MCMC), so summaries
# computed from it remain comparable across training and simulation.
Base.@kwdef struct MovingBlockBootstrap <: LengthPreservingSampler
    block_size::Int = 100
end

function (RS::MovingBlockBootstrap)( data::DataContainer, options::MethodsOptions, index_cache )
    ndata = length( index_cache )

    x_inds = Vector{Int}( undef, ndata )
    y_inds = Vector{Int}( undef, ndata )
    _fill_moving_block_bootstrap!( x_inds, index_cache, RS.block_size )
    _fill_moving_block_bootstrap!( y_inds, index_cache, RS.block_size )

    return x_inds, y_inds
end

function _fill_moving_block_bootstrap!( out::Vector{Int}, index_cache, block_size::Int )
    ndata = length( index_cache )
    last_start_ind = ndata - block_size + 1

    pos = 1
    while pos <= ndata
        start_ind = rand( 1:last_start_ind )
        len = min( block_size, ndata - pos + 1 )
        @views out[ pos:pos+len-1 ] .= index_cache[ start_ind:start_ind+len-1 ]
        pos += len
    end

    return out
end

function get_index_size( sampler::MovingBlockBootstrap, data, options )
    return size( data, 2 )
end

function resample_sizes(
    sampler::MovingBlockBootstrap,
    ndata::Int
)
    1 <= sampler.block_size <= ndata ||
        throw(ArgumentError(
            "block_size must satisfy 1 <= block_size <= ndata"
        ))
    return ndata, ndata
end
