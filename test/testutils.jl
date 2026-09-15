using EmpiricalLikelihoodToolbox
using Test
using Random
using LinearAlgebra
using Statistics
using StableRNGs
using Distributions

using EmpiricalLikelihoodToolbox:
    DataContainer, BufferContainer, MCMCState, ResultsBuffer,
    calculate_diffs, calculate_diffs!, recursive_welford!, regularized_cholesky,
    donsker_covariance, contract, contract!, invcdf, standardize!,
    update_model_parameters, initial_state,
    empcdf_raw, empcdf_raw!, empcdf_kernelsmoothed, empcdf_kernelsmoothed!, resolve_ecdf,
    chamfer_distance, chamfer_distance!,
    _bin_select, _calculate_bin_bounds, create_bins,
    resample_sizes, get_index_size,
    allocate_buffer, get_summary_length, required_diff_order, generate_stat_name,
    finalize_summary, get_bin_quantity, calculate_summary_statistic!,
    StandardECDFMultiDimensional, StandardECDFDiffMultiDimensional,
    evaluate_log_prior, evaluate_single_prior, calculate_loss,
    allocate_buffers, initialize_datacontainer, train_target, compute_log_path,
    AbstractResampler, LengthPreservingSampler, LengthChangingSampler, ResultsBuffer,
    TwoSetSummary, ChamferECDFSummary, step!

"Deterministic 1xN row vector of Float64, the shape every model's `solve_model` returns."
row(v) = reshape(Float64.(collect(v)), 1, :)

"""
A deterministic random walk. Use wherever a summary needs *non-degenerate* data: derivative-based
neighbour summaries (ID/IDDiff) collapse on a linear ramp, because a constant derivative makes every
pairwise distance zero and every intrinsic-dimension ratio non-finite.
"""
wiggly(n::Int; seed::Int = 17) = row(cumsum(randn(StableRNG(seed), n)))

"""
    make_options(; N_obs, kwargs...)

`MethodsOptions` with test-friendly defaults: quiet, few resamplings, deterministic resampler.
"""
make_options(; N_obs::Int, kwargs...) =
    MethodsOptions(; N_obs = N_obs, verbose = false, training_resamplings = 50,
                     bins_resamplings = 10, kwargs...)

"""
    make_container(data; kwargs...)

A bare `DataContainer` over `data` with no derivative data. For summaries that need derivatives or
allocated buffers, go through `make_target` instead.
"""
function make_container(data::AbstractMatrix{Float64}; kwargs...)
    opts = make_options(; N_obs = size(data, 2), kwargs...)
    return DataContainer(observations = data, options = opts)
end

"""
    make_target(data, stats...; priors=nothing, loss=LogLikelihood(), kwargs...)

Build a `TargetData` the same way the real pipeline does. Always construct the summary statistics
fresh at the call site: `finalize_summary` binds a summary instance to one `BufferContainer`, so a
shared instance would alias buffers across tests.
"""
function make_target(data::AbstractMatrix{Float64}, stats...;
                     priors = nothing, loss = LogLikelihood(), kwargs...)
    opts = make_options(; N_obs = size(data, 2), kwargs...)
    return TargetData(data, JointSummaryStatistics(stats...), opts; priors = priors, loss = loss)
end

# BlowflyModel{E} declares a type parameter no field uses, so `BlowflyModel()` cannot infer it and
# throws. The explicit `{Int}` is the only way to build one; pinned in test_known_issues.jl.
blowfly() = BlowflyModel{Int}()

"Every concrete simulation model, with parameters left at their defaults."
all_models() = (
    Lorenz63Model(), OUModel(), NormalModel(),
    blowfly(), NegExpModel(), RickerModel(),
)

"""
Models whose parameters can actually be updated. `Accessors.setproperties` rebuilds a struct
positionally, which re-triggers BlowflyModel's uninferable type parameter, so BlowflyModel is
excluded -- it cannot be driven by MCMC at all. Pinned in test_known_issues.jl.
"""
updatable_models() = (
    Lorenz63Model(), OUModel(), NormalModel(), NegExpModel(), RickerModel(),
)

"""
    prepare_summary(data, stat; kwargs...) -> (stat, data_container, buffers, summary_length)

Run the parts of `TargetData` that allocate and finalize a single summary. Bins are auto-selected
only when the summary carries an uninitialized `bins` field, so explicitly-constructed bins survive
intact -- which is what makes hand-calculated expectations possible.
"""
function prepare_summary(data::AbstractMatrix{Float64}, stat; kwargs...)
    opts = make_options(; N_obs = size(data, 2), kwargs...)
    stats = (stat,)
    diff_orders = map(required_diff_order, stats)
    dc = initialize_datacontainer(data, stats, opts, diff_orders)
    buffers, dc, len = allocate_buffers(stats, dc, opts, diff_orders)

    if hasproperty(stat, :bins) && isnothing(getproperty(stat, :bins))
        stat = create_bins(dc, stat, opts, buffers.index_cache)
    end

    return finalize_summary(stat, dc, buffers), dc, buffers, len
end

"""
    summarize(data, stat; kwargs...) -> Vector{Float64}

`prepare_summary` followed by one training-mode evaluation. The index sets come from the configured
resampler, because every summary buffer is sized from `resample_sizes` -- passing the full cache to a
length-changing resampler's buffers would be a dimension mismatch. Under `NoResampling` the split is
the whole cache, twice.
"""
function summarize(data::AbstractMatrix{Float64}, stat; kwargs...)
    stat, dc, buffers, len = prepare_summary(data, stat; kwargs...)
    x_inds, y_inds = dc.options.resampling_type(dc, dc.options, buffers.index_cache)
    out = fill(NaN, len)
    calculate_summary_statistic!(out, stat, x_inds, y_inds, dc, buffers)
    return out
end

"""
    with_priors(target, priors)

Copy a `TargetData` changing only its priors, so two losses can be compared with an identical
likelihood term. Rebuilding via `make_target` would re-run the randomized training and change
`obs_mean`/`cov_factorization` too.
"""
with_priors(t::TargetData, priors) = TargetData(
    t.data, t.summary_statistics, priors, t.options, t.buffers,
    t.obs_mean, t.cov_factorization, t.summary_length,
    t.standardization_mean, t.standardization_sd,
)

# A model whose loss throws part-way through a chain, for exercising mcmcrun's recovery path.
struct ExplodingModel <: AbstractSimulationModel
    p1::Float64
    p2::Float64
    all_parameters::Tuple{Vararg{Symbol}}
    active_parameters::Tuple{Vararg{Symbol}}
end
ExplodingModel() = ExplodingModel(0.0, 0.0, (:p1, :p2), (:p1, :p2))

const EXPLODE_AFTER = Ref(20)
const EXPLODE_CALLS = Ref(0)

function EmpiricalLikelihoodToolbox.calculate_loss(
        params, target, ::ExplodingModel, mcmc_options; rng_seed::UInt64 = rand(UInt64))
    EXPLODE_CALLS[] += 1
    EXPLODE_CALLS[] > EXPLODE_AFTER[] && error("simulated mid-chain failure")
    return -0.5 * sum(abs2, params)
end
