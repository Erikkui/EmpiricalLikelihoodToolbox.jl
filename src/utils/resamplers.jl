"""
    AbstractResampler

Supertype for all resamplers.

A resampler decides how a dataset is split into an "x" and a "y" index set. Those sets are what the
two-set summaries ([`CIL`](@ref), [`ID`](@ref), [`ChamferDistance`](@ref)) compare against each
other, and repeatedly resampling is also how GSL builds the target's sampling distribution and how
each simulation is summarized during MCMC. The choice of resampler therefore changes what a
two-set summary actually measures, not just how noisy it is.

Every resampler is a callable struct implementing three methods:

- `(sampler)(data::DataContainer, options::MethodsOptions, index_cache) -> (x_inds, y_inds)`
- `get_index_size(sampler, data, options) -> Int` — how many indices the cache should hold.
- `resample_sizes(sampler, ndata::Int) -> (nx, ny)` — the lengths the call above will return.

`resample_sizes` must agree with the lengths actually returned, because every summary buffer is
sized from it ahead of time. It is also where argument validation lives, so an invalid block size is
reported when buffers are allocated rather than mid-chain.

Subtypes are split into [`LengthPreservingSampler`](@ref) and [`LengthChangingSampler`](@ref).
"""
abstract type AbstractResampler end

"""
    LengthPreservingSampler <: AbstractResampler

Resamplers whose x and y sets each have the same length as the original data.

This matters beyond bookkeeping: `calculate_loss` takes a faster path that skips resampling
entirely when `n_summaries <= 1` and the resampler is length-preserving, since the full index cache
can be used directly.
"""
abstract type LengthPreservingSampler <: AbstractResampler end
"""
    LengthChangingSampler <: AbstractResampler

Resamplers whose x and y sets are smaller than the original data, or of unequal size.

Summaries computed from these are not directly comparable with summaries of the full dataset, so
the same resampler must be used consistently when building the target and when summarizing
simulations.
"""
abstract type LengthChangingSampler <: AbstractResampler end

# Resampling types

"""
    RademacherSplit()

Split the indices into two random halves. This is the default `resampling_type`.

The index cache is shuffled and cut in the middle: `x` gets `div(n, 2)` indices and `y` gets the
rest, so the two sets are disjoint and together cover every index. With an odd `n` the `y` set is
one longer.

Randomly reshuffling destroys temporal ordering, so for time series where that ordering matters use
[`ContiguousBlockSplit`](@ref) or [`MovingBlockBootstrap`](@ref) instead.

!!! warning "Shuffles in place and returns views"
    This shuffles the caller's `index_cache` and returns views into it. A second call overwrites the
    arrays the previous call handed back, and the cache ends up permuted after use. Copy the
    returned indices if you need them to outlive the next call.

# Examples
```julia
resampler = RademacherSplit()
```
"""
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
"""
    ContiguousBlockSplit(; timeseries_block_size = 100)

Take one random contiguous block as the "x" set and everything else as "y".

Intended for time series, where the random reshuffling of [`RademacherSplit`](@ref) would destroy
the temporal structure the summary is meant to capture. The block start is drawn uniformly from
every valid position, so `x` always has exactly `timeseries_block_size` indices and `y` has the
remaining `ndata - timeseries_block_size`, in ascending order.

This is an asymmetric, length-changing split. If you want contiguous blocks but both sets the same
size as the data, use [`MovingBlockBootstrap`](@ref).

# Fields
- `timeseries_block_size::Int`: Length of the contiguous "x" block. Must satisfy
  `1 <= timeseries_block_size < ndata`, checked in `resample_sizes`.

# Examples
```julia
resampler = ContiguousBlockSplit(timeseries_block_size = 24)
```
"""
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
"""
    NoResampling()

Return all the data, unchanged, as both the "x" and the "y" set.

Completely deterministic — there is no randomness at all, and repeated calls give identical results.
Useful under BSL, where the observed summary is a fixed reference value that needs no resampling,
and for any deterministic evaluation of a summary statistic.

Two consequences worth knowing:

- `standardize = true` is rejected with an `ArgumentError`, because every training draw is
  identical and there is no spread to standardize against.
- Under GSL the training covariance is exactly zero, so the target covariance reduces to the ridge
  term alone.

!!! warning "Returns the same object twice"
    `x_inds` and `y_inds` are the very same array as `index_cache`, not copies. Mutating one
    mutates all three.

# Examples
```julia
resampler = NoResampling()
```
"""
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
"""
    StandardBootstrap()

Draw two independent bootstrap samples with replacement, each the size of the original data.

Indices are drawn individually, so the x and y sets overlap freely, repeat indices, and carry no
temporal structure. For time series use [`MovingBlockBootstrap`](@ref), which preserves correlation
within each block.

# Examples
```julia
resampler = StandardBootstrap()
```
"""
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
"""
    MovingBlockBootstrap(; block_size = 100)

Moving block bootstrap: build each set by concatenating random contiguous blocks.

Blocks of `block_size` consecutive observations are drawn with replacement (so they may overlap or
repeat) and concatenated until the set reaches the original data length, with the final block
truncated to fit exactly. Both x and y always have exactly `ndata` indices.

This keeps short-range temporal correlation inside each block, unlike [`RademacherSplit`](@ref) and
[`StandardBootstrap`](@ref), which resample individual points and destroy it. Unlike
[`ContiguousBlockSplit`](@ref) it is length-preserving, so summaries stay comparable with those
computed on the full dataset.

Two degenerate cases: `block_size = 1` reduces to [`StandardBootstrap`](@ref), and
`block_size = ndata` gives a single deterministic block in the data's original order.

# Fields
- `block_size::Int`: Length of each contiguous block. Must satisfy `1 <= block_size <= ndata`,
  checked in `resample_sizes`.

# Examples
```julia
resampler = MovingBlockBootstrap(block_size = 24)
```
"""
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
