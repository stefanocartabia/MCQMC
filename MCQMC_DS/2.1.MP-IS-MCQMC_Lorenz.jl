include("C:/Users/mussi/Documents/Manhattan/Leuven/MCQMC/Data_Libraries/1.Libraries_setup.jl")


# -----------------------------
# Simulate system trajectory
# -----------------------------
function simulate_system(f!::Function, u0, tspan::Tuple{<:Real,<:Real}, dt::Real)
    u0vec = Float64.(collect(u0))  
    prob = ODEProblem(f!, u0vec, (float(tspan[1]), float(tspan[2])))
    sol  = solve(prob, Tsit5(); saveat=float(dt))
    return Array(sol)
end

# -----------------------------
# Gaussian log-likelihood with known noise covariance
# -----------------------------
function loglik_gaussian(obs::AbstractMatrix{<:Real}, sim::AbstractMatrix{<:Real}, cov_mat::AbstractMatrix{<:Real})
    
    # Preliminary Check 
    @assert size(obs) == size(sim)

    cov_mat  = Matrix{Float64}(cov_mat)
    d, K = size(sim)
    sse = 0.0

    F = cholesky(Symmetric(cov_mat))
    inv_cov_mat = inv(Symmetric(cov_mat))

    # L2 error norm
    @inbounds for t in 1:K
        err = obs[:, t] .- sim[:, t] 
        sse += dot(err, inv_cov_mat*err)
    end

    # By cholensky decomposition: logdetΣ as 2*sum(log(diag(U)))  
    return -(K*d/2)*log(2π) - K*sum(log, diag(F.U)) - 0.5*sse
end


# -----------------------------
# Posterior log-density at x0
# -----------------------------

function build_log_posterior(f!::Function, obs, cov_mat, tspan, dt, logprior_x0::Function)

    function logpost(x0::NTuple{3,Float64})
        sim = simulate_system(f!, [x0[1], x0[2], x0[3]], tspan, dt)  
        return loglik_gaussian(obs, sim, cov_mat) + logprior_x0([x0[1], x0[2], x0[3]])
    end

    return logpost
end

# -----------------------------
# Gradient Log-posterior
# -----------------------------
function grad_logpost_fd!(grad::AbstractVector{Float64}, logpost::Function, x::NTuple{3,Float64}; h::Float64=1e-2)
    @inbounds for i in 1:3
        ei = (i==1, i==2, i==3)                                     # canonic base
        xp = (x[1] + h*ei[1], x[2] + h*ei[2], x[3] + h*ei[3])
        xm = (x[1] - h*ei[1], x[2] - h*ei[2], x[3] - h*ei[3])
        grad[i] = (logpost(xp) - logpost(xm))/(2*h)
    end
    grad 
end

# ===============================================================
# IS-MP MALA (Algorithm 6)
# ===============================================================

function MP_IS_MALA(f!::Function, x0_init::AbstractVector{<:Real}; obs::AbstractMatrix{<:Real}, cov_mat::AbstractMatrix{<:Real},
                    tspan::Tuple{<:Real,<:Real},dt::Real,
                    seq::AbstractMatrix{<:Real},
                    N_prop::Integer, N_iter::Integer, 
                    step_size::Real=0.12, logprior_x0::Function, h_der::Real=1e-3)


    @assert length(x0_init) == 3
    @assert size(obs,1) == 3
    @assert size(cov_mat) == (3,3)
    @assert size(seq,2) == 4                    "Seq must have 4 columns."
    @assert N_iter >= 1
    @assert N_prop >= 1

    # log posterior
    logpost = build_log_posterior(f!, obs, cov_mat, tspan, dt, logprior_x0)

    # Standard MALA constants
    step_size  = float(step_size)
    alpha  = (step_size^2)/2
    inv_var = 1/(step_size^2)

    # Initialisation 
    chain_x0       = Array{Float64}(undef, N_iter, 3)
    weighted_mean  = Array{Float64}(undef, N_iter, 3)
    accept_proxy   = 0.0
    weights_last   = zeros(N_prop+1)
    x_I = (float(x0_init[1]), float(x0_init[2]), float(x0_init[3]))
    grad  = zeros(3)                                         # gradient 
    proposals = Array{Float64}(undef, N_prop+1, 3, N_iter)   # current state + (N-1) proposals
    log_post  = zeros(N_prop+1)                              # Log Posterior 
    mean_MALA = zeros(N_prop+1, 3)
    logK_yi_z   = zeros(N_prop+1)                              # log K(y_i -> z)
    logK_z_yi   = zeros(N_prop+1)                              # log K(z -> y_i)
    w         = zeros(N_prop+1)
    row = 1                                                 # To select M-wcud row at each iteration
    
    for l in 1:N_iter
        wcud = seq[row:row+N_prop, :]
        row += (N_prop+1)

        ##############################################################################################################################
        ##########################              Generate auxiliary z  and N - 1 proposals           ##################################
        ##############################################################################################################################

        # 1) Parameters Langevin Kernel for auxiliary: z ~ N(mean_xI, ε^2 I)
        grad_logpost_fd!(grad, logpost, x_I; h=h_der)          # gradient update
        mean_xI = (x_I[1] + alpha*grad[1], x_I[2] + alpha*grad[2], x_I[3] + alpha*grad[3])

        z  = (mean_xI[1] + step_size*(quantile(Normal(), wcud[1,1])),
              mean_xI[2] + step_size*(quantile(Normal(), wcud[1,2])),
              mean_xI[3] + step_size*(quantile(Normal(), wcud[1,3])))

        # 2) Parameters Langevin Kernel for Proposals: y_j ~ N(mean_z, ε^2 I) with mean_z = z + α ∇logπ(z)
        grad_logpost_fd!(grad, logpost, z; h=h_der)
        mu_z = (z[1] + alpha*grad[1], z[2] + alpha*grad[2], z[3] + alpha*grad[3])

        # 3) Define complete set of proposals: XI + (N-1) proposals
        proposals[1,1,l] = x_I[1]; proposals[1,2,l] = x_I[2]; proposals[1,3,l] = x_I[3]

        for j in 2:(N_prop+1)
            proposals[j,1,l] = mu_z[1] + step_size*(quantile(Normal(), wcud[j,1]))
            proposals[j,2,l] = mu_z[2] + step_size*(quantile(Normal(), wcud[j,2]))
            proposals[j,3,l] = mu_z[3] + step_size*(quantile(Normal(), wcud[j,3]))
        end

        ##############################################################################################################################
        ##########################                    Compute Weights of IS-estimator               ##################################
        ##############################################################################################################################

        
        for i in 1:(N_prop+1)
            # Log of target distribution at proposal, log π(y_i)
            yi = (proposals[i,1,l], proposals[i,2,l], proposals[i,3,l])
            log_post[i] = logpost(yi)
            # Gradient of target distribution at proposal
            grad_logpost_fd!(grad, logpost, yi; h = h_der)
            # Mean of MALA kernel 
            mean_MALA[i,1] = yi[1] + alpha*grad[1]; mean_MALA[i,2] = yi[2] + alpha*grad[2]; mean_MALA[i,3] = yi[3] + alpha*grad[3]

            # log K(y_i -> z) for MALA kernel
            dz1 = z[1] - mean_MALA[i,1]; dz2 = z[2] - mean_MALA[i,2]; dz3 = z[3] - mean_MALA[i,3]
            logK_yi_z[i] = -0.5*inv_var*(dz1*dz1 + dz2*dz2 + dz3*dz3)

            # log K(z -> y_i) for MALA kernel
            dy1 = proposals[i,1,l] - mu_z[1]; dy2 = proposals[i,2,l] - mu_z[2]; dy3 = proposals[i,3,l] - mu_z[3]
            logK_z_yi[i] = -0.5*inv_var*(dy1*dy1 + dy2*dy2 + dy3*dy3)
        end

        # Log weights
        sum_logKzi = sum(logK_z_yi)

        @inbounds for i in 1:(N_prop+1)
            w[i] = log_post[i] + logK_yi_z[i] + (sum_logKzi - logK_z_yi[i])
        end

        # Exponentiating and normalising 
        w .-= maximum(w)
        w .= exp.(w)
        w ./= sum(w)

        # Per-iteration IS estimate of E[x₀] weighted_mean[l,1] = dot(w, proposals[:,1,l]); weighted_mean[l,2] = dot(w, proposals[:,2,l]); weighted_mean[l,3] = dot(w, proposals[:,3,l])

        # Sample next index 
        vprime = wcud[end,4]
        csum = 0.0; I_new = 1
        @inbounds for i in 1:(N_prop+1)
            csum += w[i]
            if vprime <= csum
                I_new = i; break
            end
        end
        accept_proxy += (1.0 - w[I_new])

        # Update anchor and prepare mean_xI for next iteration
        x_I = (proposals[I_new,1,l], proposals[I_new,2,l], proposals[I_new,3,l])
        chain_x0[l,1] = x_I[1]; chain_x0[l,2] = x_I[2]; chain_x0[l,3] = x_I[3]

        if l == N_iter
            weights_last = w
        end
    end


    (chain_x0 = chain_x0,
     #  weighted_mean = weighted_mean,
     accept_proxy = accept_proxy / N_iter,     
     weights_last = weights_last,
     grad = grad,
     proposals = proposals)
end




###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################



