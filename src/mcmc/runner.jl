#------------MCMCState struct
struct MCMCState{T, M<:AbstractMatrix{<:Real}}
    current_params::T
    ss_current::Float64
    proposal_cov::M
end

struct ResultsBuffer{M<:AbstractMatrix{<:Real}, I1, I2, T}
    chain::M
    sschain::Vector{Float64}
    prop_cov::M
    current_iter::I1
    acceptance::T
    stuck_kicks::I2
end

function _allocate_results_buffer( model, chain_length, n_stages )
    npar = length( get_active_model_params( model )[2] )
    chain_buffer = zeros( npar, chain_length )
    ss_buffer = zeros( chain_length )
    prop_cov = Matrix{Float64}( I, npar, npar )
    results_buffer = ResultsBuffer(
        chain_buffer,
        ss_buffer,
        prop_cov,
        Ref(0),
        zeros( Int, n_stages ),
        Ref(0),
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


# The model is the one the target was built with, so the simulations always match the observed data.
function mcmcrun( target::TargetData, mcmc_options::MCMCOptions )

    model = target.model

    # Setup
    MCMCRun = mcmc_options.mcmc_algorithm
    proposal_width = MCMCRun.proposal_width

    chain_length = mcmc_options.nsteps

    current_params = _check_initial_params( model, mcmc_options )
    proposal_cov = _check_proposal_cov( model, proposal_width )

    ss_current = calculate_loss( current_params, target, model, mcmc_options )

    n_stages = isa( MCMCRun, DRAM) ? MCMCRun.n_stages : 1
    results_buffers = _allocate_results_buffer( model, chain_length, n_stages )

    state = MCMCState( current_params, ss_current, proposal_cov )

    try
        results, state = MCMCRun( target, model, state, mcmc_options, results_buffers )
        results = @set results.acceptance = results.acceptance ./ chain_length
        results = @set results.prop_cov = state.proposal_cov

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
        acceptance = results_buffers.acceptance

        results = ResultsBuffer(
            partial_chain,
            partial_sschain,
            state.proposal_cov,
            Ref(last_successful_step),
            acceptance ./ last_successful_step,
            results_buffers.stuck_kicks,
            )

        _print_mcmc_error( err, results, chain_length )

        return results, state
    end
end
