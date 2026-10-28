# Abstract types
abstract type AbstractSummaryStatistic end

abstract type AbstractInferenceMethod end

abstract type AbstractECDFSummary <: AbstractSummaryStatistic end
abstract type AbstractChamferSummary <: AbstractSummaryStatistic end
abstract type AbstractCumulativeSumSummary <: AbstractSummaryStatistic end

abstract type StandardECDFSummary <: AbstractECDFSummary end
abstract type TwoSetSummary <: AbstractECDFSummary end

abstract type CILSummary <: TwoSetSummary end
abstract type IDSummary <: TwoSetSummary end
abstract type ChamferECDFSummary <: TwoSetSummary end

abstract type AbstractSimulationModel end

# Fill the `name` field of a summary statistic, once its other fields are set
with_name( stat ) = @set stat.name = generate_stat_name( stat )


# Container structs
#------------Main options struct
Base.@kwdef struct MethodsOptions{R, F, E, IM}
    axis_uniform::Symbol = :xax
    covariance_type::Symbol = :cov
    bins_resamplings::Int = 40
    resampling_type::R = RademacherSplit()
    training_resamplings::Int = 1000
    N_obs::Int = 0                  # Raw series length, filled in by TargetData from the data
    effective_N_obs::Int = N_obs    # Post-embedding, post-trim length, filled in by TargetData
    mcmc_resamplings::Int = training_resamplings
    n_summaries::Int = 1
    n_loss_evals::Int = 1
    standardize::Bool = false
    verbose::Bool = false
    ecdf_calculation_type::Symbol = :default    # :default, :kernel_smoothed
    ecdf_function::F = resolve_ecdf( ecdf_calculation_type )
    embedding_dim::E = 0
    embedding_type::Symbol = :delay    # :delay, :diff
    inference_method::IM = GSL()
end

#------------Buffer container for non-allocating in-place computations
struct BufferContainer{S, I, M, SD, ST, MB}
    summary_buffers::S
    training_buffer::M
    mcmc_buffer::M
    simulation_diffs::SD
    simulation_statistic::ST
    index_cache::I
    method_buffer::MB    # Inference-method specific scratch space, see allocate_method_buffer
end

#------------Container for passing data, differences and options to functions
Base.@kwdef struct DataContainer{M <: AbstractMatrix{Float64}, D <: AbstractVector{Matrix{Float64}}, O, P}
    observations::M
    differences::D = Vector{Matrix{Float64}}(undef, 0)
    difference_orders::O = [0]
    options::P = MethodsOptions()
end

#------------MCMC options struct
# - `update_interval`: the AM "noisy likelihood" heuristic kicks the chain (re-evaluates the current
#   loss) after this many consecutive rejections. `update_interval <= 0` disables the heuristic.
# - `discard_noisy_updates`: on a kick, also rewind to the last known good state (AM only).
# - `reevaluate_current_loss`: re-evaluate the current state's loss at every step.
#   BSL and ABC are exact pseudo-marginal methods only with the heuristic off (`update_interval <= 0`)
#   and `reevaluate_current_loss = false`. ABC also needs a fixed ε: with ε adaptation
#   (`EpsilonAdaptation`, AM only), only the chain after the adaptation phase is a valid ABC-MCMC
#   chain, at the final ε (`results.epsilon`).
# - `likelihood_noise_scale`: sd of Gaussian noise added to each log-target evaluation (testing aid).
#   Under ABC ε adaptation, the recomputed current log-target does not get this noise again.
# - `loss_function`: `nothing` (default) uses the loss stored in `target.options.inference_method`
#   (ABC has no loss, so it must stay `nothing`).
#   If given, it must equal that loss, otherwise `mcmcrun` throws.
Base.@kwdef struct MCMCOptions{A, G, F}
    nsteps::Int = 1000
    update_interval::Int = 50
    discard_noisy_updates::Bool = false
    reevaluate_current_loss::Bool = false
    mcmc_algorithm::A
    initial_params::G = nothing
    loss_function::F = nothing
    likelihood_noise_scale::Float64 = 0.0
end

#------------MCMC target
struct TargetData{C, S, P, O, B, T, OM, IC, SL, M}
    data::C
    summary_statistics::S
    priors::P
    options::O
    buffers::B
    obs_mean::OM
    cov_factorization::IC
    summary_length::SL
    standardization_mean::T
    standardization_sd::T
    model::M    # The simulator: mcmcrun starts from it, and its dt_obs is the step of the observed data's differences
end
