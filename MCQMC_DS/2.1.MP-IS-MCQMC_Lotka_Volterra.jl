# -----------------------------
# Simulate system trajectory
# -----------------------------
function simulate_system(f!::Function, u0, par, tspan::Tuple{<:Real,<:Real}, dt::Real)
    prob = ODEProblem(f!, Float64.(collect(u0)) , (float(tspan[1]), float(tspan[2])), par)
    sol  = solve(prob, Tsit5(); saveat=float(dt))

    return Array(sol)
end

# Gaussian log-likelihood with known noise covariance

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

        # sse += dot(err, F \ (F' \ err))
    end

    # By cholensky decomposition: logdetΣ as 2*sum(log(diag(U)))  
    return -(K*d/2)*log(2π) - K*sum(log, diag(F.U)) - 0.5*sse
end

# -------- Log Posterior theta = log par -----------

function build_log_posterior(f!::Function, u0, obs, cov_mat, tspan, dt, logprior_par::Function)

    function logpost(theta::NTuple{4,Float64})
        par = (exp(theta[1]), exp(theta[2]), exp(theta[3]), exp(theta[4]))
        sim = simulate_system(f!, u0, par, tspan, dt)
        return loglik_gaussian(obs, sim, cov_mat) + logprior_par(par)
    end
    return logpost
end

# Gradient Log-posterior
function grad_fd!(grad::AbstractVector{Float64}, log_post::Function, x::AbstractVector{<:Real}; h::Float64=1e-3)

    @inbounds for i in 1:length(x)
        xp = collect(x); xp[i] += h
        xm = collect(x); xm[i] -= h
        grad[i] = (log_post(tuple(xp...)) - log_post(tuple(xm...))) / (2h)
    end
    return grad
end


function MP_IS_MALA_LV(f!::Function, u0, obs, cov_mat, tspan, dt, logprior_par::Function,
                       init_par::AbstractVector{<:Real}; seq::AbstractMatrix{<:Real},
                       N_prop::Integer, N_iter::Integer, step_size::Real=0.12, h_der::Real=1e-3)

    D = length(init_par)
    @assert D == 4 "This implementation expects 4 log-parameters."
    @assert size(seq,2) ≥ D+1 "seq needs ≥ D Normal-quantile columns + 1 resampling column."
    @assert size(seq,1) ≥ N_iter*(N_prop+1) "seq must have at least N_iter*(N_prop+1) rows."

    logpost = build_log_posterior(f!, u0, obs, cov_mat, tspan, dt, logprior_par)

    # MALA constants
    step  = float(step_size)
    alpha = (step^2) / 2
    inv_var = 1 / (step^2)

    # State and storage
    xI = Float64.(init_par)                 # current point in log-parameter space
    chain = Array{Float64}(undef, N_iter, D)
    accept_proxy = 0.0
    weights_last = zeros(N_prop + 1)

    z = zeros(D); grad = zeros(D); yi_vec = zeros(D)

    proposals = Array{Float64}(undef, N_prop + 1, D, N_iter)
    log_post_i = zeros(N_prop + 1)
    mu_Mala = zeros(N_prop + 1, D)
    logK_yi_z = zeros(N_prop + 1)
    logK_z_yi = zeros(N_prop + 1)
    w = Array{Float64}(undef, N_iter, N_prop + 1)

    row = 1
    for l in 1:N_iter
        wcud = seq[row:row + N_prop, :]
        row += (N_prop + 1)

        # Sample auxiliary z from MALA step at xI

        mu_x = xI .+ alpha .* grad_fd!(grad, logpost, xI; h = h_der)
        @inbounds for d in 1:D
            z[d] = mu_x[d] + step * quantile(Normal(), wcud[1, d])
        end

        # mu_z = z + alpha * grad logpi(z)

        mu_z = z .+ alpha .* grad_fd!(grad, logpost, z; h = h_der)

        # candidate 1: current state and N new proposals
        proposals[1, 1, l] = xI[1]; proposals[1, 2, l] = xI[2]
        proposals[1, 3, l] = xI[3]; proposals[1, 4, l] = xI[4]

        @inbounds for j in 2:(N_prop + 1), d in 1:D
            proposals[j, d, l] = mu_z[d] + step * quantile(Normal(), wcud[j, d])
        end

        # 3) Weights and transition kernels
        for i in 1:(N_prop + 1)

            # y_i as vector for gradient; as tuple for logpost
            @inbounds for d in 1:D
                yi_vec[d] = proposals[i, d, l]
            end

            log_post_i[i] = logpost(tuple(yi_vec...))

            grad_fd!(grad, logpost, yi_vec; h = h_der)

            mu_Mala[i, 1] = yi_vec[1] + alpha * grad[1]; mu_Mala[i, 2] = yi_vec[2] + alpha * grad[2]
            mu_Mala[i, 3] = yi_vec[3] + alpha * grad[3]; mu_Mala[i, 4] = yi_vec[4] + alpha * grad[4]   

            # log K(y_i -> z)
            s1 = 0.0
            @inbounds for d in 1:D
                s1 += (z[d] - mu_Mala[i, d])^2
            end
            logK_yi_z[i] = -0.5 * inv_var * s1

            # log K(z -> y_i)
            s2 = 0.0
            @inbounds for d in 1:D
                s2 += (proposals[i, d, l] - mu_z[d])^2
            end
            logK_z_yi[i] = -0.5 * inv_var * s2
        end

        # Stabilized normalization of weights
        sum_logKz_yi = sum(logK_z_yi)
        @inbounds for i in 1:(N_prop + 1)
            w[l,i] = log_post_i[i] + logK_yi_z[i] + (sum_logKz_yi - logK_z_yi[i])
        end

        w[l,:] .-= maximum(w[l,:])
        w[l,:] .= exp.(w[l,:])
        w[l,:] ./= sum(w[l,:])



        # Resample with column D+1
        vprime = wcud[end, D + 1]
        csum = 0.0
        I_new = 1
        @inbounds for i in 1:(N_prop + 1)
            csum += w[l,:][i]
            if vprime <= csum
                I_new = i
                break
            end
        end
        accept_proxy += (1.0 - w[l, I_new])

        # Update xI in place and save to chain
        @inbounds for d in 1:D
            xI[d] = proposals[I_new, d, l]
            chain[l, d] = xI[d]
        end


    end

    return (
            proposals = proposals,
            chain = chain,
            accept_proxy = accept_proxy / N_iter,
            grad = grad,
            weights_last = w)
end








