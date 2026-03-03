function recursive_welford!( running_mean, running_cov, current_params, diff1, diff2, npar, ii )
    if ii == 1
        running_mean .= current_params
    else
        # diff1 = x_t - mean_{t-1}
        @inbounds for j in 1:npar
            diff1[j] = current_params[j] - running_mean[j]
        end

        # mean_t = mean_{t-1} + diff1 / t
        @inbounds for j in 1:npar
            running_mean[j] += diff1[j] / ii
        end

        # diff2 = x_t - mean_t
        @inbounds for j in 1:npar
            diff2[j] = current_params[j] - running_mean[j]
        end

        # C_t = ((t-2)/(t-1)) * C_{t-1} + (diff1 * diff2^T) / (t-1)
        weight1 = (ii - 2) / (ii - 1)
        weight2 = 1 / (ii - 1)
        @inbounds for col in 1:npar
            @inbounds for row in 1:npar
                running_cov[row, col] = weight1 * running_cov[row, col] + weight2 * diff1[row] * diff2[col]
            end
        end
    end
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
    current_iter = results.current_iter

    am_scaling_parameter = 2.38^2 / npar
    epsilon = 1e-6

    noise_buffer = zeros( npar )
    params_proposal = zeros( npar )

    # Buffers for the Welford covariance update
    running_mean = copy(state.current_params)
    running_cov = copy(state.proposal_cov)
    diff1 = zeros(npar)
    diff2 = zeros(npar)

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
        chain[:, ii] = state.current_params
        sschain[ii] = state.ss_current
        current_iter[] = ii

        # 3. Adaptive Update
        recursive_welford!( running_mean, running_cov, state.current_params, diff1, diff2, npar, ii )
        if ii > 100 && ii % adaptation_interval == 0
            for col in 1:npar
                for row in 1:npar
                    proposal_cov[row, col] = running_cov[row, col] * am_scaling_parameter
                end
                # Add epsilon to diagonal safely without allocating a new Matrix(I)
                proposal_cov[col, col] += epsilon
            end
            proposal_cov_L = cholesky(Symmetric(proposal_cov)).L
            state.proposal_cov .= proposal_cov
        end

        description = "Acc: $(round(n_accepted/ii, digits=2)) SS: $(round(state.ss_current, digits=2))"
        set_description(iter, description)
    end

    return results, state
end
