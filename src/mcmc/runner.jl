#------------MCMCState struct
struct MCMCState{T, M<:AbstractMatrix{<:Real}}
    current_params::T
    ss_current::Float64
    proposal_cov::M
    accepted::Int
    stuck_kicks::Int
end

function allocate_results_buffer( npar, chain_length )
    chain_buffer = zeros( npar, chain_length )
    ss_buffer = zeros( chain_length )
    prop_cov = Matrix{Float64}( I, npar, npar )
    results_buffer = (
        chain=chain_buffer,
        sschain=ss_buffer,
        prop_cov=prop_cov,
        current_iter=Ref(0),
        acceptance=0.0, )
    return results_buffer

end

function mcmcrun( target::TargetData, model::AbstractSimulationModel, mcmc_options::MCMCOptions )

    # Setup
    MCMCRun = mcmc_options.mcmc_algorithm
    proposal_width = MCMCRun.proposal_width

    chain_length = mcmc_options.nsteps

    model_params = get_params( model )
    npar = length( model_params )

    # Set uninformative priors if not provided
    if length( target.priors ) != npar
        uninformative_priors = ntuple( _ -> nothing, Val(npar) )
        target = @set target.priors = uninformative_priors
    end

    if isnothing( mcmc_options.initial_params )
        current_params = model_params .+ 0.1 .* randn( npar )
    else
        current_params = mcmc_options.initial_params
    end

    proposal_cov = Matrix{Float64}( I, npar, npar )*proposal_width
    ss_current = calculate_loss( current_params, target, model, mcmc_options.loss_function )
    ss_current += evaluate_log_prior( current_params, target.priors )

    results_buffers = allocate_results_buffer( npar, chain_length )

    state = MCMCState( current_params, ss_current, proposal_cov, 0, 0 )

    try
        results, state = MCMCRun( target, model, state, mcmc_options, results_buffers )
        results = @set results.acceptance = state.accepted / chain_length
        results = @set results.prop_cov = state.proposal_cov

        acceptance_print = round(results.acceptance*100, digits=2)
        stuck_print_percentage = round(state.stuck_kicks*100/chain_length, digits=2)
        println("MCMC completed")
        println("Acceptance rate: ", acceptance_print, "%")
        println( "Stuck kicks: ", stuck_print_percentage, "%\n")

        return results, state
    catch e
        last_successful_step = results_buffers.current_iter[]
        partial_chain = results_buffers.chain[:, 1:last_successful_step]
        partial_sschain = results_buffers.sschain[1:last_successful_step]

        results = (
            chain=partial_chain,
            sschain=partial_sschain,
            prop_cov=state.proposal_cov,
            current_iter=last_successful_step,
            acceptance=state.accepted / last_successful_step, )

        acceptance_print = round(results.acceptance*100, digits=2)
        stuck_print_percentage = round(state.stuck_kicks*100/chain_length, digits=2)

        println("\nMCMC crashed at iteration ", last_successful_step, " due to: \n", e)
        println("Returning partial chain.")
        println("Acceptance rate: ", acceptance_print, "%")
        println( "Stuck kicks: ", stuck_print_percentage, "%\n")

        Base.showerror(stdout, e, catch_backtrace())
        println() # Add a newline for readability

        return results, state
    end
end
