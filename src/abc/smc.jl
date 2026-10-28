#------------ABC-SMC with replenishment (Drovandi & Pettitt 2011, Biometrics 67(1), 225-233).
# A population of particles is moved through a sequence of ABC posteriors with decreasing ε. Each
# generation drops the particles with the largest distances, sets ε_t to the largest kept distance,
# replenishes the population by resampling the survivors, and diversifies the duplicates with
# random-walk Metropolis-Hastings moves that leave the ABC posterior at ε_t invariant. The tolerance
# schedule is adaptive: no sequence of ε has to be given.

#------------Options
"""
    ABCSMCOptions

Settings for [`abc_smc`](@ref) (replenishment ABC-SMC, Drovandi & Pettitt 2011).

# Fields
- `n_particles::Int`: The number of particles (default: 1000). Generation 0 draws this many parameter vectors from the priors.
- `drop_fraction::Float64`: The fraction of particles with the largest distances dropped (and replenished) per generation, δ in the paper (default: 0.5). The `floor( n_particles * (1 - drop_fraction) )` survivors set ε_t.
- `c::Float64`: The target probability that a replenished particle is not moved by any of its MCMC moves (default: 0.01). Sets the number of moves R_t = ⌈log(c) / log(1 - p_acc)⌉.
- `p_acc_min::Float64`: Stop when the MCMC acceptance rate of a generation falls below this value (default: 0.01).
- `epsilon_target::Float64`: Stop when ε reaches this tolerance (default: 0.0, ie. no tolerance stop). If the largest kept distance falls below it, ε_t is set to `epsilon_target`, the last generation is run at exactly this ε, and the algorithm stops.
- `max_generations::Int`: The maximum number of generations after generation 0 (default: 100).
- `proposal_scale::Float64`: The MCMC proposal covariance is `proposal_scale` times the sample covariance of the survivors (default: 2.0).
- `max_mcmc_repeats::Int`: The upper bound on R_t, the number of MCMC moves per replenished particle (default: 500, enough for `c = 0.01` at an acceptance rate of 0.01).
- `verbose::Bool`: Print one line per generation (default: false).

"""
Base.@kwdef struct ABCSMCOptions
    n_particles::Int = 1000
    drop_fraction::Float64 = 0.5
    c::Float64 = 0.01
    p_acc_min::Float64 = 0.01
    epsilon_target::Float64 = 0.0
    max_generations::Int = 100
    proposal_scale::Float64 = 2.0
    max_mcmc_repeats::Int = 500
    verbose::Bool = false
end

# Returns the number of survivors per generation.
function _check_smc_options( smc_options::ABCSMCOptions )
    if !( 0 < smc_options.drop_fraction < 1 )
        throw( ArgumentError( "drop_fraction must be in (0, 1), got $(smc_options.drop_fraction)." ) )
    end
    n_alive = floor( Int, smc_options.n_particles * ( 1 - smc_options.drop_fraction ) )
    if !( 2 <= n_alive < smc_options.n_particles )
        throw( ArgumentError( "n_particles = $(smc_options.n_particles) and drop_fraction = $(smc_options.drop_fraction) give $(n_alive) survivors per generation. At least 2 must survive (for the proposal covariance) and at least 1 must be dropped." ) )
    end
    if !( 0 < smc_options.c < 1 )
        throw( ArgumentError( "c must be in (0, 1), got $(smc_options.c)." ) )
    end
    if !( 0 <= smc_options.p_acc_min < 1 )
        throw( ArgumentError( "p_acc_min must be in [0, 1), got $(smc_options.p_acc_min)." ) )
    end
    if smc_options.epsilon_target < 0
        throw( ArgumentError( "epsilon_target must be non-negative, got $(smc_options.epsilon_target)." ) )
    end
    if smc_options.max_generations < 1 || smc_options.max_mcmc_repeats < 1
        throw( ArgumentError( "max_generations and max_mcmc_repeats must be at least 1." ) )
    end
    if !( smc_options.proposal_scale > 0 )
        throw( ArgumentError( "proposal_scale must be positive, got $(smc_options.proposal_scale)." ) )
    end
    return n_alive
end

# The replenishment ABC-SMC moves target prior × 1{ρ ≤ ε_t}: only the uniform kernel fits, and one
# simulation per move (the indicator of a single simulation).
function _check_smc_method( abc::ABC )
    if !( abc.kernel isa UniformKernel )
        throw( ArgumentError( "abc_smc needs ABC( kernel = UniformKernel() ), got $(_kernel_string( abc.kernel ))." ) )
    end
    if abc.n_sim != 1
        throw( ArgumentError( "abc_smc uses one simulation per MCMC move, but the target has ABC( n_sim = $(abc.n_sim) ). Build the target with n_sim = 1." ) )
    end
    return nothing
end


#------------Helpers

# R_t = ⌈log(c) / log(1 - p_acc)⌉: with this many moves, a particle stays unmoved with probability
# ≈ c. Clamped to [1, max_mcmc_repeats]; p_acc = 0 gives the upper bound, p_acc = 1 a single move.
function _mcmc_repeats( acceptance_rate::Float64, c::Float64, max_mcmc_repeats::Int )
    if acceptance_rate <= 0
        return max_mcmc_repeats
    elseif acceptance_rate >= 1
        return 1
    end
    repeats_real = log( c ) / log1p( -acceptance_rate )
    repeats_capped = min( repeats_real, Float64( max_mcmc_repeats ) )
    return max( 1, ceil( Int, repeats_capped ) )
end

# Lower Cholesky factor of proposal_scale × the sample covariance of the survivors (columns of
# survivor_params). A singular covariance (eg. many duplicated survivors, or fewer distinct
# survivors than parameters) gets a diagonal jitter, starting at 1e-10 × the mean variance and
# growing ×10 until the factorization succeeds.
function _proposal_cholesky_factor( survivor_params::AbstractMatrix{Float64}, proposal_scale::Float64 )
    npar = size( survivor_params, 1 )
    survivor_cov = cov( survivor_params, dims=2 )
    proposal_cov = Symmetric( proposal_scale .* survivor_cov )

    factorization = cholesky( proposal_cov; check=false )
    mean_variance = mean( abs, diag( proposal_cov ) )
    jitter = 1e-10 * ( mean_variance > 0 ? mean_variance : 1.0 )
    n_attempts = 0
    while !issuccess( factorization ) && n_attempts < 20
        jittered_cov = Symmetric( proposal_cov + jitter * I )
        factorization = cholesky( jittered_cov; check=false )
        jitter *= 10
        n_attempts += 1
    end
    if !issuccess( factorization )
        throw( ArgumentError( "The ABC-SMC proposal covariance of the $(npar) parameters could not be factorized, even with a diagonal jitter." ) )
    end

    return Matrix( factorization.L )
end

# Run n_moves random-walk MH moves on particle `particle_index` (a column of `particles`), in place.
# Target: prior × 1{ρ ≤ ε}. The proposal is symmetric, so a move is accepted when the prior ratio
# test passes and the simulated distance is within ε. Both tests must pass, so the prior test runs
# first, and a failing one (or a zero prior) skips the simulation (early rejection). Each
# simulation gets its own Xoshiro seeded from `rng`, as in the param-level calculate_loss.
# Returns (n_accepted, n_simulated).
function _abc_mh_moves!( particles, distances, log_priors, particle_index::Int, n_moves::Int,
                         proposal_factor, epsilon::Float64, target, proposal_buffer, noise_buffer, rng )
    model = target.model
    priors = target.priors
    active_param_names = model.active_parameters
    current_params = @view particles[:, particle_index]

    n_accepted = 0
    n_simulated = 0
    for _ in 1:n_moves
        randn!( rng, noise_buffer )
        mul!( proposal_buffer, proposal_factor, noise_buffer )
        proposal_buffer .+= current_params

        proposal_log_prior = evaluate_log_prior( proposal_buffer, priors, active_param_names )
        isfinite( proposal_log_prior ) || continue      # zero prior density: no simulation

        log_prior_ratio = proposal_log_prior - log_priors[particle_index]
        log_uniform = log( rand( rng ) )
        log_uniform < log_prior_ratio || continue       # fails the prior test: no simulation

        proposal_model = update_model_parameters( model, proposal_buffer )
        simulation_rng = Xoshiro( rand( rng, UInt64 ) )
        proposal_distance = simulate_distance( target, proposal_model, simulation_rng )
        n_simulated += 1
        proposal_distance <= epsilon || continue

        current_params .= proposal_buffer
        distances[particle_index] = proposal_distance
        log_priors[particle_index] = proposal_log_prior
        n_accepted += 1
    end

    return n_accepted, n_simulated
end

# One round of n_moves moves on every replenished particle (indices first_moved:n_particles).
# Returns (n_accepted, n_simulated) summed over the particles.
function _move_replenished!( particles, distances, log_priors, first_moved::Int, n_moves::Int,
                             proposal_factor, epsilon::Float64, target, proposal_buffer, noise_buffer, rng )
    n_particles = size( particles, 2 )
    total_accepted = 0
    total_simulated = 0
    for particle_index in first_moved:n_particles
        n_accepted, n_simulated = _abc_mh_moves!( particles, distances, log_priors, particle_index, n_moves,
                                                  proposal_factor, epsilon, target, proposal_buffer, noise_buffer, rng )
        total_accepted += n_accepted
        total_simulated += n_simulated
    end
    return total_accepted, total_simulated
end


#------------Main algorithm
"""
    abc_smc( target, smc_options::ABCSMCOptions; rng = Random.default_rng() )

Replenishment ABC-SMC (Drovandi & Pettitt 2011). Moves a population of `n_particles` particles
through ABC posteriors `π(θ) 1{ρ(θ) ≤ ε_t}` with an adaptive, non-increasing tolerance sequence ε_t.

Algorithm:
0. Draw `n_particles` parameter vectors from the priors and simulate their distances (failed
   simulations have distance `Inf` and are allowed here).
1. Sort the particles by distance. Keep the `n_alive = floor( n_particles * (1 - drop_fraction) )`
   closest ones; ε_t is the largest kept distance (or `epsilon_target`, if that is larger: then all
   particles within `epsilon_target` are kept and this is the last generation).
2. Replace the dropped particles by uniform draws (with replacement) from the survivors.
3. Move each replenished particle with R_t random-walk MH moves, proposal `N( θ, Σ )`, `Σ =
   proposal_scale × cov(survivors)`. A move is accepted when its prior is positive, it passes the
   prior-ratio test, and a fresh simulation has `ρ ≤ ε_t`. R_t = ⌈log(c) / log(1 - p_acc)⌉ uses the
   acceptance rate p_acc of the previous generation. In the first generation, a trial of one move
   per particle estimates p_acc, and R_1 - 1 more moves follow.
4. Stop when p_acc < `p_acc_min`, when ε_t reached `epsilon_target`, or after `max_generations`
   generations.

The target must be built with `inference_method = ABC( kernel = UniformKernel(), n_sim = 1 )` (the
defaults) and proper priors for every active parameter. The target's own ε is not used (ABC-SMC
sets its own tolerances), but `TargetData` still resolves it: pass a fixed `epsilon` (eg. `ABC( 1.0 )`)
to skip the ε pilot. Other kernels and `n_sim > 1` throw an `ArgumentError`.

Results are reproducible for a given `rng` when the resampler is deterministic (eg. `NoResampling()`):
the other resamplers draw from the global RNG.

Returns a `NamedTuple`:
- `particles`: `npar × n_particles` matrix of the final particles (rows in the order of
  `get_active_model_params( target.model )`). Duplicates are expected.
- `distances`: length-`n_particles` vector of the final distances, all `≤ last( epsilons )`.
- `epsilons`: ε_t of each generation (non-increasing).
- `acceptance_rates`: the MCMC acceptance rate p_acc of each generation.
- `mcmc_repeats`: R_t, the number of MCMC moves per replenished particle, of each generation.
- `n_simulations`: the total number of model simulations, generation 0 included.

# Examples
```julia
target, _ = TargetData( data, summary_statistics, methods_options, model; priors = priors )   # inference_method = ABC( 1.0 )
result = abc_smc( target, ABCSMCOptions( n_particles = 2000, verbose = true ); rng = Xoshiro( 1 ) )
posterior_mean = vec( mean( result.particles, dims=2 ) )
```

See also [`ABCSMCOptions`](@ref), [`abc_rejection`](@ref), [`ABC`](@ref).
"""
function abc_smc( target, smc_options::ABCSMCOptions; rng = Random.default_rng() )
    abc = _check_abc_target( target, "abc_smc" )
    _check_smc_method( abc )
    n_alive = _check_smc_options( smc_options )

    n_particles = smc_options.n_particles
    epsilon_target = smc_options.epsilon_target
    c = smc_options.c
    max_mcmc_repeats = smc_options.max_mcmc_repeats
    verbose = smc_options.verbose
    priors = target.priors
    active_param_names = target.model.active_parameters

    # Generation 0: ABC rejection with ε = Inf, ie. prior draws and their distances
    params, summaries = _pilot_summaries( target, n_particles, rng )
    distances = _pilot_distances( abc.distance, target.obs_mean, summaries )
    log_priors = [ evaluate_log_prior( column, priors, active_param_names ) for column in eachcol( params ) ]
    particles = params
    n_simulations = n_particles

    npar = size( particles, 1 )
    proposal_buffer = zeros( npar )
    noise_buffer = zeros( npar )

    epsilons = Float64[]
    acceptance_rates = Float64[]
    mcmc_repeats = Int[]
    previous_acceptance_rate = NaN     # unknown before the first generation: a trial move estimates it

    for generation in 1:smc_options.max_generations
        order = sortperm( distances )
        particles = particles[:, order]
        distances = distances[order]
        log_priors = log_priors[order]

        # Every particle already within the target tolerance: nothing left to do
        if distances[end] <= epsilon_target
            break
        end

        epsilon = distances[n_alive]
        n_kept = n_alive
        reached_target = epsilon <= epsilon_target
        if reached_target
            epsilon = epsilon_target
            n_kept = count( distance_value -> distance_value <= epsilon_target, distances )
        end

        # Proposal from the survivors, before they are duplicated
        survivors = @view particles[:, 1:n_kept]
        proposal_factor = _proposal_cholesky_factor( survivors, smc_options.proposal_scale )

        # Replenish: dropped particles become uniform copies of survivors
        for particle_index in ( n_kept + 1 ):n_particles
            source_index = rand( rng, 1:n_kept )
            particles[:, particle_index] .= @view particles[:, source_index]
            distances[particle_index] = distances[source_index]
            log_priors[particle_index] = log_priors[source_index]
        end

        # Diversify the duplicates with R_t MH moves each
        first_moved = n_kept + 1
        n_moved = n_particles - n_kept
        if isnan( previous_acceptance_rate )
            trial_accepted, trial_simulated = _move_replenished!( particles, distances, log_priors, first_moved, 1,
                                                                  proposal_factor, epsilon, target, proposal_buffer, noise_buffer, rng )
            trial_rate = trial_accepted / n_moved
            repeats = _mcmc_repeats( trial_rate, c, max_mcmc_repeats )
            rest_accepted, rest_simulated = _move_replenished!( particles, distances, log_priors, first_moved, repeats - 1,
                                                                proposal_factor, epsilon, target, proposal_buffer, noise_buffer, rng )
            n_accepted = trial_accepted + rest_accepted
            n_simulated = trial_simulated + rest_simulated
        else
            repeats = _mcmc_repeats( previous_acceptance_rate, c, max_mcmc_repeats )
            n_accepted, n_simulated = _move_replenished!( particles, distances, log_priors, first_moved, repeats,
                                                          proposal_factor, epsilon, target, proposal_buffer, noise_buffer, rng )
        end

        acceptance_rate = n_accepted / ( repeats * n_moved )
        n_simulations += n_simulated
        previous_acceptance_rate = acceptance_rate

        push!( epsilons, epsilon )
        push!( acceptance_rates, acceptance_rate )
        push!( mcmc_repeats, repeats )

        if verbose
            println( "ABC-SMC generation $(generation): ε = $(round( epsilon, sigdigits=4 )), p_acc = $(round( acceptance_rate, digits=3 )), R = $(repeats), simulations = $(n_simulations)" )
        end

        if reached_target || acceptance_rate < smc_options.p_acc_min
            break
        end
    end

    return ( particles = particles, distances = distances, epsilons = epsilons,
             acceptance_rates = acceptance_rates, mcmc_repeats = mcmc_repeats, n_simulations = n_simulations )
end
