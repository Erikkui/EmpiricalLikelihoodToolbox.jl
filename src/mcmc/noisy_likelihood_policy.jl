#------------Noisy-likelihood policy, shared by AM and DRAM
# With a noisy (simulated) log-likelihood, a state whose estimate came out too high can trap the
# chain: every proposal is rejected against it. The policy is built in mcmcrun from MCMCOptions:
# - `reevaluate_current_loss`: the sampler re-evaluates the current state at every step, with the
#   same random seed as the proposal (common random numbers).
# - `update_interval > 0`: after more than `update_interval` consecutive rejections the current state
#   is re-evaluated ("kick", counted in results.stuck_kicks). `update_interval <= 0` disables it.
# - `discard_noisy_updates`: a kick instead rewinds the chain to the state before the last
#   acceptance (backup_params, plus the proposal's running mean/covariance), re-evaluates it and
#   repeats the step without storing it.
# All three make the chain inexact: GSL is approximate anyway, but BSL and ABC are exact
# pseudo-marginal samplers only with `update_interval <= 0` and `reevaluate_current_loss = false`.
mutable struct NoisyLikelihoodPolicy
    update_interval::Int
    discard_noisy_updates::Bool
    reevaluate_current_loss::Bool
    n_stuck::Int                    # consecutive rejections
    backup_params::Vector{Float64}  # the current state before the last acceptance
end

function NoisyLikelihoodPolicy( mcmc_options, initial_params::AbstractVector{<:Real} )
    policy = NoisyLikelihoodPolicy(
        mcmc_options.update_interval,
        mcmc_options.discard_noisy_updates,
        mcmc_options.reevaluate_current_loss,
        0,
        Vector{Float64}( initial_params ),
        )
    return policy
end

# Count the step and return the action for the sampler: :none, :kick (re-evaluate the current
# state) or :rewind (restore the backup, re-evaluate it and repeat the step).
function check_stuck_chain!( policy::NoisyLikelihoodPolicy, accepted::Bool )
    if accepted
        policy.n_stuck = 0
        return :none
    end

    policy.n_stuck += 1
    if policy.update_interval > 0 && policy.n_stuck > policy.update_interval
        policy.n_stuck = 0
        return policy.discard_noisy_updates ? :rewind : :kick
    end
    return :none
end

# Called on acceptance, before the current state is replaced: saves what a rewind restores.
function backup_state!( policy::NoisyLikelihoodPolicy, current, proposal::AdaptiveProposal )
    if policy.discard_noisy_updates
        copyto!( policy.backup_params, current.params )
        backup_proposal_adaptation!( proposal )
    end
    return nothing
end

# Carry out the action of check_stuck_chain! on the current state. Returns true when the step must be
# repeated without storing it (:rewind).
function apply_stuck_chain_action!( action::Symbol, policy::NoisyLikelihoodPolicy, current, proposal::AdaptiveProposal,
                                    target, model, mcmc_options, inference_state, results )
    if action === :none
        return false
    end

    if action === :rewind
        # Restore the last known "good" state; it is simulated again, so its method data needs no backup
        copyto!( current.params, policy.backup_params )
        restore_proposal_adaptation!( proposal )
    end

    # Fresh random numbers (no common seed): the point is a new estimate of the current state
    evaluate_log_target!( current, target, model, current.params, mcmc_options, inference_state )
    results.stuck_kicks[] += 1

    return action === :rewind
end
