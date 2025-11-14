
# MP-IS-MCQMC-MALA for Lotka-Volterra model
include("C:/Users/mussi/Documents/Manhattan/Leuven/MCQMC/Data_Libraries/1.Libraries_setup.jl")

# Simulate system trajectory
function simulate_system(f!::Function, u0, par, tspan::Tuple{<:Real, <:Real}, dt::Real)
    prob = ODEProblem(f!, Float64.(collect(u0)) , (float(tspan[1]), float(tspan[2])), par)
    sol  = solve(prob, Tsit5(); saveat=float(dt))

    return Array(sol)
end

# Gaussian log-likelihood with known noise covariance (NO INVERSE)
function loglik_gaussian(obs::AbstractMatrix{<:Real}, sim::AbstractMatrix{<:Real}, cov_mat::AbstractMatrix{<:Real})

    d, K = size(sim)
    S = Symmetric(Matrix(cov_mat))      
    F = cholesky(S; check = true)        # Σ = U' * U

    sse = 0.0
    @inbounds @views for t in 1:K
        err = obs[:, t] .- sim[:, t]     
        y = F.U \ err                    #  U * y = e_t  
        sse += dot(y, y)                 # e_t' Σ^{-1} e_t = y'y
    end

    return -(K*d/2) * log(2π) -  K*sum(log, diag(F.U)) - 0.5 * sse
end

# Log Posterior theta = log par
function build_log_posterior(f!::Function, u0, obs, cov_mat, tspan, dt, logprior_par::Function)

    function logpost(theta::NTuple{4,Float64})
        par = (exp(theta[1]), exp(theta[2]), exp(theta[3]), exp(theta[4]))
        sim = simulate_system(f!, u0, par, tspan, dt)
        # logPost(θ)=logL(e^θ)+logPrior(e^θ)+∑θ
        return loglik_gaussian(obs, sim, cov_mat) + logprior_par(par) + sum(theta)
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
    # return grad
    return grad
end


# Tensor Metric for Simplified Manifold: Fisher Information in the log-parameter space
function Tensor_Metric_sm(f!, u0, theta::AbstractVector{<:Real}, tspan, dt, cov_mat::AbstractMatrix{<:Real}; λ::Float64 = 1e-6, h::Float64 = 1e-4)

    # sim = simulate_system(f!, u0, Tuple(exp.(theta)), tspan, dt)  
    d, K = 2, 501
    D = length(theta)
    chol_cov_mat = cholesky(Symmetric(Matrix(cov_mat)))
    sim_p = Array{Float64}(undef, d, K, D)    
    sim_m = Array{Float64}(undef, d, K, D)   

    for j in 1:D
        theta_p = collect(theta); theta_p[j] += h
        sim_p[:, :, j] = simulate_system(f!, u0, Tuple(exp.(theta_p)), tspan, dt)
        theta_m = collect(theta); theta_m[j] -= h
        sim_m[:, :, j] = simulate_system(f!, u0, Tuple(exp.(theta_m)), tspan, dt)

    end

    G = zeros(D, D)
    Jt = zeros(d, D)                     
    tmp = zeros(d)             
    for t in 1:K
        for j in 1:D
            tmp .= sim_p[:, t, j] .- sim_m[:, t, j]
            Jt[:, j] .= tmp ./ (2*h)
        end
        # G += Jt' * (invΣ * Jt)
        W  = chol_cov_mat \ Jt            # W = Σ^{-1} Jt
        G .+= transpose(Jt) * W
    end

    G = Symmetric(G + λ * I)
    Minv = inv(G)
    L = cholesky(Symmetric(Minv)).L

    return (FisherInfo = G, InvFisherInfo = Minv, L = L)
end

function IS_MP_sMALA_LV(f!::Function, u0, obs, cov_mat, tspan, dt, logprior_par::Function,
                       init_par::AbstractVector{<:Float64}; seq::AbstractMatrix{<:Float64},
                       N_prop::Integer, N_iter::Integer, step_size::Float64=0.12, h_der::Float64=1e-3)

    D = length(init_par)
    @assert D == 4 "This implementation expects 4 log-parameters."
    @assert size(seq,2) ≥ D+1 "seq needs ≥ D Normal-quantile columns + 1 resampling column."

    # Definition Log-posterior distribution
    f_logpost = build_log_posterior(f!, u0, obs, cov_mat, tspan, dt, logprior_par)

    # State and storage
    xI = init_par                            # Initial point in log-parameter space                          
    I_new = 1

    chain = Array{Float64}(undef, N_iter, D)
    accept_proxy = 0.0

    z = zeros(D); grad = zeros(D); yi = zeros(D)

    proposals = Array{Float64}(undef, N_prop + 1, D, N_iter)
    log_post_i = zeros(N_prop + 1)                              # current log-posterior 
    mu_yi = zeros(D)                                # mean normal Langevin dynamics
    logK_yi_z = zeros(N_prop + 1)
    logK_z_yi = zeros(N_prop + 1)
    w = Array{Float64}(undef, N_iter, N_prop + 1)
    logw = Vector{Float64}(undef, N_prop + 1)

    # Early stopping criterion
    stall_cnt = 0 
    l_effective = 0
    # max_stall  = max(5, cld(N_iter, 20))
    max_stall = cld(N_iter, 15)
    tol_move   = 1e-5

    # MALA constants 
    alpha = (step_size^2) / 2
    CovScaling = 1
   
    
    row = 1
    for l in 1:N_iter
        wcud = seq[row:row + N_prop, :]
        row += (N_prop + 1)

        # K(xi,xj) = K(xi,z)K(z,xj)
        # Sample N+1 PROPOSALS: current state and N new ones
        proposals[1, :, l] = xI
        metric = Tensor_Metric_sm(f!, u0, xI, tspan, dt, cov_mat)
        # Sample bridge variable z with MALA kernel K(z|xI)
        # Drift sm-MALA with ∇logπ(xI​)
        mu_x = xI + alpha * (metric.InvFisherInfo * grad_fd!(grad, f_logpost, xI; h = h_der))
        z = mu_x + (step_size*CovScaling) * (metric.L*quantile.(Normal(), wcud[1, 1:D]))
        # Sample N proposals with MALA kernel K(yi|z)
        # Drift sm-MALA with ∇logπ(z​)
        metric_z = Tensor_Metric_sm(f!, u0, z, tspan, dt, cov_mat)
        mu_z     = z + alpha * (metric_z.InvFisherInfo * grad_fd!(grad, f_logpost, z; h=h_der))

        @inbounds for j in 2:(N_prop + 1)
            proposals[j, :, l] = mu_z + (step_size*CovScaling) * (metric_z.L*quantile.(Normal(), wcud[j, 1:D]))
        end

        # Stationary distribution 𝑝(𝐼 = 𝑖 ∣ 𝑦_(1:𝑁+1))
        # Weights and transition kernels
        # In logarithmic scale:   Log 𝑝(𝐼 = 𝑖 ∣ 𝑦_(1:𝑁+1)) = LogPosteriors + LogKs
        #                                                 = LogPosteriors + LogKiz + sum(LogKzi) - LogKzi
        #  K(yᵢ, z) = (2π)^(-d/2) |Σ|^(-1/2) * exp( -1/2 * (z - μ(yᵢ))ᵀ * Σ^(-1) * (z - μ(yᵢ)) )

        Prec_z   = (1 / (step_size^2 * CovScaling^2)) .* metric_z.FisherInfo
        Rz       = cholesky(Symmetric(Prec_z)).U

        for i in 1:(N_prop + 1)
            yi = proposals[i, 1:D, l] 

            # Posterior log-density: log π(yi)
            log_post_i[i] = f_logpost(tuple(yi...))

            # μ(yi): Langevin drift 
            metric_y = Tensor_Metric_sm(f!, u0, yi, tspan, dt, cov_mat)
            mu_yi    = yi + alpha * (metric_y.InvFisherInfo * grad_fd!(grad, f_logpost, yi; h=h_der))

            # FORWARD: log K(z | yi) = -1/2 (z-μ(yi))' Prec_y (z-μ(yi)) + 1/2 log|Prec_y|
            Prec_y = (1 / (step_size^2 * CovScaling^2)) .* metric_y.FisherInfo
            Ry     = cholesky(Symmetric(Prec_y)).U
            d_fwd  = z .- mu_yi                 
            y1     = Ry * d_fwd
            qf     = dot(y1, y1)
            logK_yi_z[i] = -0.5*qf + sum(log, diag(Ry))  # (−D/2 log 2π è costante → si elide nei pesi)

            # BACKWARD: log K(yi | z) = -1/2 (yi-μ(z))' Prec_z (yi-μ(z))
            # metric_z = Tensor_Metric(f!, u0, z, tspan, dt, cov_mat)
            d_bwd = yi .- mu_z
            y2    = Rz * d_bwd
            qb    = dot(y2, y2)
            logK_z_yi[i] = -0.5*qb

        end

        # Importance weights w_i ∝ π(yi) K(yi,y\i)
        logw .= log_post_i .+ logK_yi_z  .+ sum(logK_z_yi) .- logK_z_yi
        logw .-= maximum(logw)
        logZ = log(sum(exp.(logw)))
        w[l, 1:(N_prop+1)] .= exp.(logw .- logZ)

        # Resampling step and Update xI
        I_new = findfirst(cumsum(w[l,:]) .>= wcud[end, D + 1])
        accept_proxy += (1.0 - w[l, I_new])
        chain[l, :] = proposals[I_new, :, l]
        xI = chain[l, :]
        
        # Early stopping criterion: norm(xI​−xl−1​)=∑​(xI,j ​− xl−1,j​)^2​
        if l > 1 && norm(xI .- chain[l-1, :]) ≤ tol_move
            stall_cnt += 1
        else
            stall_cnt = 0
        end

        l_effective = l

        if stall_cnt ≥ max_stall
            break
        end
    end

    chain = chain[1:l_effective, :]
    w     = w[1:l_effective, :]
    proposals = proposals[:, :, 1:l_effective]

    return (
            proposals = proposals,
            chain = chain,
            accept_proxy = accept_proxy / l_effective,
            grad = grad,
            weights = w
#             FisherInfo = metric_z.FisherInfo,  
#             Scaling    = metric_z.L            
)
end





