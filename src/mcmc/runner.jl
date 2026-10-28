#------------MCMCState struct
# current_distances: the ABC distances of the current state (n_sim values; empty for GSL and BSL),
# see _commit_current_distances!.
struct MCMCState{T, M<:AbstractMatrix{<:Real}}
    current_params::T
    ss_current::Float64
    proposal_cov::M
    current_distances::Vector{Float64}
end

#------------Results struct
# - `distance_chain`: ABC only, the n_sim distances of the stored state at each step
#   (n_sim × nsteps); 0 × nsteps for GSL and BSL. Used by `abc_post_correct`.
# - `epsilon_chain`: ABC only, the ε of each step (changes only during ε adaptation); NaN otherwise.
# - `epsilon`: ABC only, the final (adapted) ε; NaN otherwise.
struct ResultsBuffer{M<:AbstractMatrix{<:Real}, I1, I2, T}
    chain::M
    sschain::Vector{Float64}
    prop_cov::M
    current_iter::I1
    acceptance::T
    stuck_kicks::I2
    distance_chain::Matrix{Float64}
    epsilon_chain::Vector{Float64}
    epsilon::Float64
end

# Results without ABC fields (GSL and BSL layout).
function ResultsBuffer( chain, sschain, prop_cov, current_iter, acceptance, stuck_kicks )
    chain_length = size( chain, 2 )
    distance_chain = zeros( 0, chain_length )
    epsilon_chain = fill( NaN, chain_length )
    return ResultsBuffer( chain, sschain, prop_cov, current_iter, acceptance, stuck_kicks, distance_chain, epsilon_chain, NaN )
end

function _allocate_results_buffer( model, chain_length, n_stages, method )
    npar = length( get_active_model_params( model )[2] )
    chain_buffer = zeros( npar, chain_length )
    ss_buffer = zeros( chain_length )
    prop_cov = Matrix{Float64}( I, npar, npar )
    distance_chain = _allocate_distance_chain( method, chain_length )
    epsilon_chain = fill( NaN, chain_length )
    results_buffer = ResultsBuffer(
        chain_buffer,
        ss_buffer,
        prop_cov,
        Ref(0),
        zeros( Int, n_stages ),
        Ref(0),
        distance_chain,
        epsilon_chain,
        NaN,
    )
    return results_buffer
end

function _check_initial_params( model, options )
    _, active_param_values = get_active_model_params( model )
    npar_active = length( active_param_values )
    if isnothing( options.initial_params )
        current_params = active_param_values .* ( 1 .+ 0.01.*randn( npar_active ) )
    else
        current_params = deepcopy( options.initial_params )
    end
    return current_params
end

function _check_proposal_cov( model, proposal_width )
    _, active_param_values = get_active_model_params( model )
    npar_active = length( active_param_values )
    if proposal_width isa AbstractVector
        if length( proposal_width ) != npar_active
            nprop = length( proposal_width )
            msg = "proposal_width has $nprop variances for $npar_active active parameters"
            throw( ArgumentError( msg ) )
        end
        proposal_cov = diagm( proposal_width )
    else
        proposal_cov = proposal_width.*ones( npar_active ) |> diagm
    end
    return proposal_cov
end

function _print_mcmc_error( err, results, chain_length )
    acceptance_print = round.( results.acceptance.*100, digits=2 )
    stuck_print_percentage = round.( results.stuck_kicks.*100 ./ chain_length, digits=2 )

    println("\nMCMC crashed at iteration ", results.current_iter[], " due to: \n", err)
    println("Returning partial chain.")
    println("Acceptance rate: ", acceptance_print, "%")
    println( "Stuck kicks: ", stuck_print_percentage, "%\n")

    Base.showerror( stdout, err, catch_backtrace() )
    println() # Add a newline for readability
end


# The loss lives in the target's inference method; an explicit mcmc_options.loss_function is only
# accepted when it agrees with it. ABC has no loss: then any explicit loss_function throws.
_method_loss( method ) = method.loss
_method_loss( ::ABC ) = nothing

function _check_loss_function( target, mcmc_options )
    loss_function = mcmc_options.loss_function
    method = target.options.inference_method
    method_loss = _method_loss( method )
    if isnothing( loss_function )
        return nothing
    end
    if isnothing( method_loss )
        msg = "MCMCOptions.loss_function = $(loss_function) is set, but the inference method $(method) has no loss function. Leave MCMCOptions.loss_function unset."
        throw( ArgumentError( msg ) )
    end
    if loss_function != method_loss
        msg = "MCMCOptions.loss_function = $(loss_function) differs from the loss of the inference method ($(method_loss)). Set the loss in the inference method, eg. GSL(loss = ...), and leave MCMCOptions.loss_function unset."
        throw( ArgumentError( msg ) )
    end
    return nothing
end

# ABC ε adaptation is implemented in AM only.
_check_adaptation_support( algorithm, adaptation ) = nothing
function _check_adaptation_support( ::DRAM, ::EpsilonAdaptation )
    throw( ArgumentError( "ABC epsilon adaptation (adapt_epsilon) is supported only with AM, not DRAM." ) )
end

# The model is the one the target was built with, so the simulations always match the observed data.
function mcmcrun( target::TargetData, mcmc_options::MCMCOptions )

    _check_loss_function( target, mcmc_options )

    # Checked here, not in the algorithm: errors inside the algorithm are caught below
    epsilon_adaptation = _epsilon_adaptation( target.options.inference_method )
    _check_epsilon_adaptation( epsilon_adaptation, target.options.inference_method )
    _check_adaptation_support( mcmc_options.mcmc_algorithm, epsilon_adaptation )

    model = target.model

    # Setup
    MCMCRun = mcmc_options.mcmc_algorithm
    proposal_width = MCMCRun.proposal_width

    chain_length = mcmc_options.nsteps

    method = target.options.inference_method

    current_params = _check_initial_params( model, mcmc_options )
    proposal_cov = _check_proposal_cov( model, proposal_width )

    ss_current = calculate_loss( current_params, target, model, mcmc_options )
    current_distances = _allocate_current_distances( method )
    _commit_current_distances!( method, current_distances, target, current_params )

    n_stages = isa( MCMCRun, DRAM) ? MCMCRun.n_stages : 1
    results_buffers = _allocate_results_buffer( model, chain_length, n_stages, method )

    state = MCMCState( current_params, ss_current, proposal_cov, current_distances )

    try
        results, state = MCMCRun( target, model, state, mcmc_options, results_buffers )
        results = @set results.acceptance = results.acceptance ./ chain_length
        results = @set results.prop_cov = state.proposal_cov
        results = @set results.epsilon = _final_epsilon( method, results.epsilon_chain, results.current_iter[] )

        acceptance_print = round.( results.acceptance.*100, digits=2 )
        stuck_print_percentage = round.( results.stuck_kicks.*100 ./ chain_length, digits=2 )
        println("MCMC completed")
        println("Acceptance rate: ", acceptance_print, "%")
        println( "Stuck kicks: ", stuck_print_percentage, "%\n")

        return results, state
    catch err
        last_successful_step = results_buffers.current_iter[]
        partial_chain = results_buffers.chain[:, 1:last_successful_step]
        partial_sschain = results_buffers.sschain[1:last_successful_step]
        partial_distance_chain = results_buffers.distance_chain[:, 1:last_successful_step]
        partial_epsilon_chain = results_buffers.epsilon_chain[1:last_successful_step]
        final_epsilon = _final_epsilon( method, partial_epsilon_chain, last_successful_step )
        acceptance = results_buffers.acceptance

        results = ResultsBuffer(
            partial_chain,
            partial_sschain,
            state.proposal_cov,
            Ref(last_successful_step),
            acceptance ./ last_successful_step,
            results_buffers.stuck_kicks,
            partial_distance_chain,
            partial_epsilon_chain,
            final_epsilon,
            )

        _print_mcmc_error( err, results, chain_length )

        return results, state
    end
end
