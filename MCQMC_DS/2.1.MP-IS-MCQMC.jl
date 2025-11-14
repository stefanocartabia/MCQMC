
###############################################################################################################################################
################################################        IS-MP-MCQMC Lotka–Volterra       ######################################################
###############################################################################################################################################

include("C:/Users/mussi/Documents/Manhattan/Leuven/MCQMC/Data_Libraries/1.Libraries_setup.jl")

# Simulate system trajectory
function simulate_system(f!::Function, u0, par, tspan::Tuple{<:Real, <:Real}, dt::Real)
    prob = ODEProblem(f!, Float64.(u0) , (float(tspan[1]), float(tspan[2])), par)
    sol  = solve(prob, Tsit5(); saveat=float(dt))

    return Array(sol)
end

# Gaussian log-likelihood with known noise covariance 
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

    return -0.5 * (K*d * log(2π) +  2 * K * sum(log, diag(F.U)) + sse)
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
        xp = copy(x); xp[i] += h
        xm = copy(x); xm[i] -= h
        grad[i] = (log_post(tuple(xp...)) - log_post(tuple(xm...))) / (2h)
    end
    # return grad
    return grad
end


###############################################################################################################################################
################################################        IS-MP-MCQMC Lotka–Volterra       ######################################################
################################################          Preconditioned MALA            ######################################################
###############################################################################################################################################

# Preconditioned MALA
# TO BE ADDED 


###############################################################################################################################################
################################################        IS-MP-MCQMC Lotka–Volterra          ###################################################
################################################    Simplified Manifold MALA Kernel         ###################################################
###############################################################################################################################################

# Tensor Metric for Simplified Manifold: Fisher Information in the log-parameter space
function Tensor_Metric_sm(f!, u0, theta::AbstractVector{<:Real}, tspan, dt, cov_mat::AbstractMatrix{<:Real}; λ::Float64 = 1e-6, h::Float64 = 1e-4)

    # sim = simulate_system(f!, u0, Tuple(exp.(theta)), tspan, dt)  
    d, K = 2, 501
    D = length(theta)
    chol_cov_mat = cholesky(Symmetric(Matrix(cov_mat)))
    sim_p = Array{Float64}(undef, d, K, D)    
    sim_m = Array{Float64}(undef, d, K, D)   

    for j in 1:D
        theta_p = copy(theta); theta_p[j] += h
        sim_p[:, :, j] = simulate_system(f!, u0, Tuple(exp.(theta_p)), tspan, dt)
        theta_m = copy(theta); theta_m[j] -= h
        sim_m[:, :, j] = simulate_system(f!, u0, Tuple(exp.(theta_m)), tspan, dt)

    end

    G = zeros(D, D); Jt = zeros(d, D);

    for t in 1:K
        for j in 1:D
             Jt[:, j] .= (sim_p[:, t, j] .- sim_m[:, t, j]) ./ (2*h)
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

# Simplified Manifold MALA Kernel 
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
            logK_yi_z[i] = -0.5*qf + sum(log, diag(Ry))  

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
)
end

###############################################################################################################################################
################################################        IS-MP-MCQMC Lotka–Volterra       ######################################################
################################################          Manifold MALA Kernel        ######################################################
###############################################################################################################################################


# Tensor Metric for Manifold: Fisher Information in the log-parameter space
function Tensor_Metric_m(f!, u0, θ::AbstractVector{<:Real}, tspan, dt, Σ; λ::Float64=1e-6, h_G::Float64=1e-3)

    base   = Tensor_Metric_sm(f!, u0, θ, tspan, dt, Σ; λ=λ)
    G      = base.FisherInfo
    InvG   = base.InvFisherInfo
    L      = base.L

    D = length(θ)
    dG = Vector{Matrix{Float64}}(undef, D)

    for j in 1:D
        theta_p = copy(θ); theta_p[j] += h_G
        Gp = Tensor_Metric_sm(f!, u0, theta_p, tspan, dt, Σ; λ=λ).FisherInfo
        theta_m = copy(θ); theta_m[j] -= h_G
        Gm = Tensor_Metric_sm(f!, u0, theta_m, tspan, dt, Σ; λ=λ).FisherInfo
        dG[j] = (Gp .- Gm) ./ (2h_G)
    end

    return (G=G, InvG=InvG, L=L, dG=dG)
end

# Christoffel correction for Manifold MALA
# Γi​(θ)=−j=1∑D​(G−1∂θj​​GG−1)ij​+21​j=1∑D​(G−1)ij​tr(G−1∂θj​​G).
function christoffel_correction(InvG::AbstractMatrix{<:Real},
                                dG::Vector{<:AbstractMatrix{<:Real}})
    D = size(InvG, 1)
    v1 = zeros(D)                       # - Σ_j (InvG * dG[j] * InvG)[:, j]
    v2 = zeros(D)                       #  (1/2) Σ_j InvG[:, j] * tr(InvG * dG[j])

    for j in 1:D
        Aj  = InvG * dG[j] * InvG
        v1 .+= @view Aj[:, j]
        v2 .+= (@view InvG[:, j]) .* tr(InvG * dG[j])
    end
    return -v1 .+ 0.5 .* v2
end

function IS_MP_MMALA_LV(
                        f!::Function, u0, obs, cov_mat, tspan, dt, logprior_par::Function,
                        init_par::AbstractVector{<:Float64};  seq::AbstractMatrix{<:Float64}, N_prop::Integer, 
                        N_iter::Integer,step_size::Float64=0.12, h_der::Float64=1e-3)

    D = length(init_par)

    # log-posterior
    f_logpost = build_log_posterior(f!, u0, obs, cov_mat, tspan, dt, logprior_par)

    # state and storage
    xI = copy(init_par)
    chain = Array{Float64}(undef, N_iter, D)
    accept_proxy = 0.0
    z = zeros(D); grad = zeros(D); yi = zeros(D)
    proposals  = Array{Float64}(undef, N_prop + 1, D, N_iter)
    log_post_i = zeros(N_prop + 1)
    mu_yi      = zeros(D)
    logK_yi_z  = zeros(N_prop + 1)
    logK_z_yi  = zeros(N_prop + 1)
    w          = Array{Float64}(undef, N_iter, N_prop + 1)
    logw       = Vector{Float64}(undef, N_prop + 1)

    # MALA constants
    eps = step_size

    # Gaussian log-density with precision P = eps_step^{-2} * G
    function logK(y::AbstractVector{<:Real}, μ::AbstractVector{<:Real}, G::AbstractMatrix{<:Real})
        Prec  = (1/eps^2) .* G
        chol_u = cholesky(Symmetric(Prec)).U
        v     = chol_u * (y .- μ)
        return -0.5 * (length(y)*log(2π) - 2sum(log, diag(chol_u)) + dot(v, v))
    end

    # early stopping
    stall_cnt   = 0
    l_effective = 0
    max_stall   = max(5, cld(N_iter, 20))
    tol_move    = 1e-5

    row = 1
    for l in 1:N_iter
        wcud = seq[row:row + N_prop, :]
        row += (N_prop + 1)

        # include current state among proposals
        proposals[1, :, l] = xI

        # metric and drift at current xI
        rtm_x = Tensor_Metric_m(f!, u0, xI, tspan, dt, cov_mat)
        grad_fd!(grad, f_logpost, xI; h=h_der)
        mu_x = xI .+ ((eps^2) / 2) .* (rtm_x.InvG * grad) .+ (eps^2) .* christoffel_correction(rtm_x.InvG, rtm_x.dG)

        # bridge variable z ~ K(· | xI)
        z = mu_x .+ eps .* (rtm_x.L * quantile.(Normal(), wcud[1, 1:D]))

        # metric and drift at z
        rtm_z = Tensor_Metric_m(f!, u0, z, tspan, dt, cov_mat)
        grad_fd!(grad, f_logpost, z; h=h_der)
        mu_z = z .+ ((eps^2) / 2) .* (rtm_z.InvG * grad) .+ (eps^2) .* christoffel_correction(rtm_z.InvG, rtm_z.dG)

        # generate N proposals y_i ~ K(· | z)
        for j in 2:(N_prop + 1)
            proposals[j, :, l] = mu_z .+ eps .* (rtm_z.L * quantile.(Normal(), wcud[j, 1:D]))
        end

        # weights
        for i in 1:(N_prop + 1)
            @views yi = proposals[i, :, l]

            # log π(y_i)
            log_post_i[i] = f_logpost(tuple(yi...))

            # metric and drift at y_i
            rtm_y = Tensor_Metric_m(f!, u0, yi, tspan, dt, cov_mat)
            grad_fd!(grad, f_logpost, yi; h=h_der)
            mu_yi = yi .+ ((eps^2) / 2) .* (rtm_y.InvG * grad) .+ (eps^2) .* christoffel_correction(rtm_y.InvG, rtm_y.dG)

            # forward and backward kernels
            logK_yi_z[i] = logK(z,  mu_yi,  rtm_y.G)  # log K(z | y_i)
            logK_z_yi[i] = logK(yi, mu_z,   rtm_z.G)  # log K(y_i | z)
        end

        sum_logK_z_yi = sum(logK_z_yi)
        for i in 1:(N_prop + 1)
            logw[i] = log_post_i[i] + logK_yi_z[i] + sum_logK_z_yi - logK_z_yi[i]
        end

        # stabilize and normalize
        logw .-= maximum(logw)
        logZ = log(sum(exp.(logw)))
        w[l, 1:(N_prop+1)] .= exp.(logw .- logZ)

        # resampling
        I_new = findfirst(cumsum(w[l, :]) .>= wcud[end, D + 1])
        accept_proxy += (1.0 - w[l, I_new])
        chain[l, :] = proposals[I_new, :, l]
        xI = chain[l, :]

        # early stopping
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

    chain     = chain[1:l_effective, :]
    w         = w[1:l_effective, :]
    proposals = proposals[:, :, 1:l_effective]

    return (
        proposals    = proposals,
        chain        = chain,
        accept_proxy = accept_proxy / l_effective,
        weights      = w
    )
end


###############################################################################################################################################
################################################        IS-MP-MCQMC Lorentz System       ######################################################
###############################################################################################################################################


# -----------------------------
# Simulate system trajectory
# -----------------------------
function simulate_system(f!::Function, u0, tspan::Tuple{<:Real,<:Real}, dt::Real)
    u0vec = Float64.(u0)  
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

