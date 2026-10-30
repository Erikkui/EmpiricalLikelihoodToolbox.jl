#------------Adaptive Metropolis proposal (Haario et al. 2001), shared by AM and DRAM
# Gaussian random-walk proposal whose covariance is adapted from the running covariance of the
# chain (Welford's algorithm, recursive_welford!): every `adaptation_interval` steps after step 100,
#   proposal_cov = running_cov * 2.38² / npar + 1e-6 I
# The backup copies hold the running mean/covariance of the last accepted state, for the rewind of
# the noisy-likelihood policy (the proposal covariance and its Cholesky factor are not restored).
mutable struct AdaptiveProposal
    running_mean::Vector{Float64}
    running_cov::Matrix{Float64}
    proposal_cov::Matrix{Float64}
    proposal_cov_L::LowerTriangular{Float64, Matrix{Float64}}
    scaling::Float64            # 2.38² / npar
    diagonal_epsilon::Float64   # added to the diagonal of the adapted covariance
    adaptation_interval::Int
    noise_buffer::Vector{Float64}
    diff1::Vector{Float64}      # Welford scratch buffers
    diff2::Vector{Float64}
    backup_mean::Vector{Float64}
    backup_cov::Matrix{Float64}
end

function AdaptiveProposal( initial_params::AbstractVector{<:Real}, initial_cov::AbstractMatrix{<:Real}, adaptation_interval::Integer )
    npar = length( initial_params )
    running_mean = Vector{Float64}( initial_params )
    running_cov = Matrix{Float64}( initial_cov )
    proposal_cov = Matrix{Float64}( initial_cov )
    proposal_cov_L = cholesky( Symmetric( proposal_cov ) ).L
    scaling = 2.38^2 / npar

    proposal = AdaptiveProposal(
        running_mean,
        running_cov,
        proposal_cov,
        proposal_cov_L,
        scaling,
        1e-6,
        Int( adaptation_interval ),
        zeros( npar ),
        zeros( npar ),
        zeros( npar ),
        copy( running_mean ),
        copy( running_cov ),
        )
    return proposal
end

# params_out = current_params + sqrt( scale ) * L * z, z ~ N( 0, I ): a draw from
# N( current_params, scale * proposal_cov ). DRAM's delayed-rejection stages use scale = γ_k.
function propose_parameters!( params_out, proposal::AdaptiveProposal, current_params; scale::Float64 = 1.0 )
    noise_buffer = proposal.noise_buffer
    randn!( noise_buffer )
    noise_buffer .*= sqrt( scale )
    mul!( params_out, proposal.proposal_cov_L, noise_buffer )   # params_out = L * noise_buffer
    params_out .+= current_params
    return params_out
end

# Add the state stored at step ii to the running mean/covariance, and adapt the proposal covariance
# every adaptation_interval steps after step 100. Returns true when the covariance (and its Cholesky
# factor) changed.
function update_proposal_covariance!( proposal::AdaptiveProposal, params, ii::Integer )
    running_mean = proposal.running_mean
    running_cov = proposal.running_cov
    proposal_cov = proposal.proposal_cov
    npar = length( running_mean )

    recursive_welford!( running_mean, running_cov, params, proposal.diff1, proposal.diff2, npar, ii )

    if ii > 100 && ii % proposal.adaptation_interval == 0
        for col in 1:npar
            for row in 1:npar
                proposal_cov[row, col] = running_cov[row, col] * proposal.scaling
            end
            # Add epsilon to diagonal without allocating a new Matrix(I)
            proposal_cov[col, col] += proposal.diagonal_epsilon
        end
        proposal.proposal_cov_L = cholesky( Symmetric( proposal_cov ) ).L
        return true
    end
    return false
end

# Save the running mean/covariance (the noisy-likelihood policy calls it on acceptance).
function backup_proposal_adaptation!( proposal::AdaptiveProposal )
    copyto!( proposal.backup_mean, proposal.running_mean )
    copyto!( proposal.backup_cov, proposal.running_cov )
    return nothing
end

# Restore the running mean/covariance saved by backup_proposal_adaptation! (the rewind of the noisy-likelihood policy).
function restore_proposal_adaptation!( proposal::AdaptiveProposal )
    copyto!( proposal.running_mean, proposal.backup_mean )
    copyto!( proposal.running_cov, proposal.backup_cov )
    return nothing
end
