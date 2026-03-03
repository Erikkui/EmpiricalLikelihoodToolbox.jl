#------------MCMCState struct
struct MCMCState{T, M<:AbstractMatrix{<:Real}}
    current_params::T
    ss_current::Float64
    proposal_cov::M
    accepted::Float64
    stuck_kicks::Int
end

Base.@kwdef struct AM
    proposal_width::Float64 = 0.01
    adaptation_interval::Int = 50
end

function (AM::AM)( target, model, state, mcmc_options, results )
    # Initialization
    adaptation_interval = AM.adaptation_interval

    chain_length = mcmc_options.nsteps
    update_interval = mcmc_options.update_interval
    loss = mcmc_options.loss_function

    npar = get_params( model ) |> length
    chain = results.chain
    sschain = results.sschain

    am_scaling_parameter = 2.38^2 / npar
    epsilon = 1e-6

    noise_buffer = zeros( npar )
    params_proposal = zeros( npar )

    # Initial Cholesky
    proposal_cov = copy( state.proposal_cov )
    proposal_cov_L = cholesky( state.proposal_cov ).L

    # Bookkeeping
    n_accepted = 0
    n_stuck = 0

    if Threads.nthreads() == 1 || Threads.threadid() == 1
        println( "Starting MCMC with AM algorithm for ", chain_length, " iterations..." )
        iter = ProgressBar( 1:chain_length, printing_delay=0.1 )
    else
        iter = 1:chain_length
    end

    for ii in iter

        # 1. Propose
        # params_proposal = state.current_params + proposal_cov_L * noise_buffer
        randn!( noise_buffer )
        mul!( params_proposal, proposal_cov_L, noise_buffer )
        params_proposal .+= state.current_params

        ss_proposal = calculate_loss( params_proposal, target, model, loss )

        # 2. Metropolis accept/reject
        log_ratio = ss_proposal - state.ss_current
        rd = log( rand() )
        accepted = rd < log_ratio

        if accepted
            copyto!( state.current_params, params_proposal )
            n_accepted += 1
            n_stuck = 0 # Reset stuck counter on move

            new_fields = ( ss_current = ss_proposal, accepted = n_accepted )
            state = setproperties( state, new_fields )
        else
            n_stuck += 1
            if n_stuck > update_interval    # If mcmc gets stuck in "too good" proposal
                ss_recalc = calculate_loss( state.current_params, target, model, loss )
                updated_fields = ( ss_current = ss_recalc, stuck_kicks = state.stuck_kicks + 1 )
                state = setproperties( state, updated_fields )
                n_stuck = 0 # Reset counter
            end
        end

        # Store
        chain[ :, ii ] = state.current_params
        sschain[ii] = state.ss_current

        # 3. Adaptive Update
        if ii > 100 && ii % adaptation_interval == 0
            proposal_cov = cov( chain[:, 1:ii]' ) .* am_scaling_parameter + epsilon * I
            proposal_cov_L = cholesky( Symmetric(proposal_cov) ).L
            state = @set state.proposal_cov = proposal_cov
        end

        description = "Acc: $(round(n_accepted/ii, digits=2)) SS: $(round(state.ss_current, digits=2))"
        set_description(iter, description)
    end

    return results, state
end
