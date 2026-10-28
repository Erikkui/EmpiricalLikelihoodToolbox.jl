#------------ABC-MCMC: tolerance adaptation, run state (ε of a running chain) and results

#------------Tolerance adaptation (Vihola and Franks 2020)
"""
    EpsilonAdaptation( ; target_acceptance = 0.1, n_adapt, step_exponent = 2/3 )

Adapt the ABC tolerance ε during the first `n_adapt` AM iterations towards a target acceptance
rate (Vihola and Franks 2020, Biometrika, "On the use of approximate Bayesian computation Markov
chain Monte Carlo with inflated tolerance and post-correction", Algorithm 3). Set it in
`ABC( adapt_epsilon = EpsilonAdaptation( n_adapt = ... ) )`. AM only: DRAM throws.

After the accept/reject step of iteration `k ≤ n_adapt`:

    log ε ← log ε + γ_k ( α* - α_k ),    γ_k = k^(-step_exponent)

with `α*` the `target_acceptance` and `α_k = min( 1, exp( log_ratio ) )` the acceptance probability
of the step (`0` for a `NaN` or `-Inf` log ratio). A low acceptance raises ε, a high one lowers it.
After `n_adapt` iterations ε is frozen, and the final value is returned in `results.epsilon`.

The adapted ε lives in the run, not in the target: the target passed to `mcmcrun` is not changed.
When ε changes, the current state's log-target is recomputed for the new ε from its stored
distances, without simulating again. The `likelihood_noise_scale` noise of `MCMCOptions` is then
not added again.

The chain is a valid ABC-MCMC chain (at the final ε) only after the adaptation phase: discard at
least `n_adapt` samples as burn-in (eg. `abc_post_correct( results, target, delta; burnin = n_adapt )`).

# Fields
- `target_acceptance::Float64`: The target acceptance rate α* (default: 0.1, the value Vihola and Franks recommend in Section 4 and use in their experiments).
- `n_adapt::Int`: The number of adapted iterations (required).
- `step_exponent::Float64`: The exponent r of the step sizes γ_k = k^(-r) (default: 2/3, used by Vihola and Franks). Their convergence result (Assumption 9) needs r in (1/2, 1].

See also [`ABC`](@ref), [`abc_post_correct`](@ref).
"""
struct EpsilonAdaptation
    target_acceptance::Float64
    n_adapt::Int
    step_exponent::Float64

    function EpsilonAdaptation( target_acceptance::Real, n_adapt::Integer, step_exponent::Real )
        if !( 0 < target_acceptance < 1 )
            throw( ArgumentError( "EpsilonAdaptation target_acceptance must be in (0, 1), got $(target_acceptance)." ) )
        end
        if n_adapt < 1
            throw( ArgumentError( "EpsilonAdaptation n_adapt must be at least 1, got $(n_adapt)." ) )
        end
        if !( 0.5 < step_exponent <= 1 )
            throw( ArgumentError( "EpsilonAdaptation step_exponent must be in (1/2, 1], got $(step_exponent)." ) )
        end
        return new( Float64( target_acceptance ), Int( n_adapt ), Float64( step_exponent ) )
    end
end
EpsilonAdaptation( ; target_acceptance = 0.1, n_adapt, step_exponent = 2/3 ) =
    EpsilonAdaptation( target_acceptance, n_adapt, step_exponent )


_epsilon_adaptation( ::AbstractInferenceMethod ) = nothing
_epsilon_adaptation( abc::ABC ) = abc.adapt_epsilon

_check_epsilon_adaptation( ::Nothing, method ) = nothing

function _check_epsilon_adaptation( ::EpsilonAdaptation, method )
    if isnan( method.epsilon )
        throw( ArgumentError( "ABC epsilon is NaN, so it cannot be adapted. Build the target with TargetData, which resolves it." ) )
    end
    return nothing
end


# No early rejection during the ε adaptation: the Robbins-Monro step needs the true acceptance
# probability of every proposal, so every proposal is simulated.
early_rejection_allowed( abc::ABC, ii ) = _early_rejection_allowed( abc.adapt_epsilon, ii )

_early_rejection_allowed( ::Nothing, ii ) = true
_early_rejection_allowed( adaptation::EpsilonAdaptation, ii ) = ii > adaptation.n_adapt


#------------Run state (see interface.jl)

# The tolerance of a running chain. mcmcrun allocates it from the target's (resolved) method, and
# log_likelihood! reads ε from here, so the ε adaptation changes it without rebuilding the target.
mutable struct ABCRunState
    epsilon::Float64
end

allocate_inference_state( abc::ABC ) = ABCRunState( abc.epsilon )

# The ε adaptation step of iteration `ii`. Returns true when ε (and so the current log-target) changed.
adapt_inference_state!( inference_state::ABCRunState, abc::ABC, current, acceptance_probability, ii ) =
    _adapt_epsilon!( abc.adapt_epsilon, inference_state, abc.kernel, current, acceptance_probability, ii )

_adapt_epsilon!( ::Nothing, inference_state, kernel, current, acceptance_probability, ii ) = false

# One Robbins-Monro step of the Vihola-Franks tolerance adaptation (see EpsilonAdaptation): a new ε
# in the run state, and the current state's log-target recomputed at the new ε from its stored
# distances. After n_adapt iterations nothing changes.
function _adapt_epsilon!( adaptation::EpsilonAdaptation, inference_state, kernel, current, acceptance_probability, ii )
    if ii > adaptation.n_adapt
        return false
    end

    step_size = ii^( -adaptation.step_exponent )
    log_epsilon = log( inference_state.epsilon ) + step_size * ( adaptation.target_acceptance - acceptance_probability )
    inference_state.epsilon = exp( log_epsilon )

    _refresh_log_target!( current, kernel, inference_state.epsilon )
    return true
end

# The log-target of an evaluated state at tolerance `epsilon`, from its stored distances (no new
# simulation). Mirrors evaluate_log_target!, without the likelihood_noise_scale noise. A state that was not
# simulated (zero prior density) keeps its log-target of -Inf.
function _refresh_log_target!( evaluated_state, kernel, epsilon )
    if !evaluated_state.simulated
        return nothing
    end

    log_likelihood = _log_kernel_average( kernel, evaluated_state.method_data, epsilon )
    evaluated_state.log_likelihood = log_likelihood
    if isfinite( log_likelihood )
        evaluated_state.log_target = log_likelihood + evaluated_state.log_prior
    else
        evaluated_state.log_target = -Inf
    end
    return nothing
end


#------------Results: the distances and ε of every stored state, and the final ε

# Distances of every stored state: n_sim × chain_length.
function allocate_results( abc::ABC, model, chain_length, n_stages )
    distance_chain = fill( NaN, abc.n_sim, chain_length )
    return _allocate_results_buffer( model, chain_length, n_stages, distance_chain )
end

function record_chain_step!( results, ::ABC, inference_state::ABCRunState, current, ii )
    results.distance_chain[:, ii] .= current.method_data
    results.epsilon_chain[ii] = inference_state.epsilon
    _record_chain_state!( results, current, ii )
    return nothing
end

# The final ε is the run state's: the last adapted value, or the fixed ε.
function finalize_results( results, ::ABC, inference_state::ABCRunState, last_step )
    results = _slice_results( results, last_step )
    results = @set results.epsilon = inference_state.epsilon
    return results
end
