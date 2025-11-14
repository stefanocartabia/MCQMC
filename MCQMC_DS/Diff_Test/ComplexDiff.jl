###############################################################################################################################################
################################################        IS-MP-MCQMC Lotka–Volterra       ######################################################
###############################################################################################################################################


# Documentation 
# - https://docs.sciml.ai/SciMLSensitivity/dev/manual/direct_adjoint_sensitivities/#SciMLSensitivity.adjoint_sensitivities
# - https://github.com/ErikQQY/ComplexDiff.jl


include("C:/Users/mussi/Documents/Manhattan/Leuven/MCQMC/Data_Libraries/1.Libraries_setup.jl")



# Simulate system trajectory
function simulate_system(f!::Function, u0, par, tspan::Tuple{<:Real, <:Real}, dt::Real)
    # For complex step differentiation, we need to handle complex parameters
    if any(p -> p isa Complex, par)
        # Use dual number approach for complex step differentiation
        return simulate_system_complex(f!, u0, par, tspan, dt)
    else
        prob = ODEProblem(f!, Float64.(u0), (float(tspan[1]), float(tspan[2])), par)
        sol = solve(prob, Tsit5(); saveat=float(dt))
        return Array(sol)
    end
end

# Helper function for complex step differentiation
function simulate_system_complex(f!::Function, u0, par, tspan::Tuple{<:Real, <:Real}, dt::Real)
    # Extract real parts for simulation, but preserve complex information for gradient
    real_par = real.(par)
    prob = ODEProblem(f!, Float64.(u0), (float(tspan[1]), float(tspan[2])), real_par)
    sol = solve(prob, Tsit5(); saveat=float(dt))
    
    # Return real simulation - the complex parts will be handled by the calling function
    result = Array(sol)
    
    # If any parameter is complex, we need to return a complex result to preserve gradient info
    if any(p -> p isa Complex, par)
        return Complex{Float64}.(result)
    else
        return result
    end
end

# Gaussian log-likelihood with known noise covariance (NO INVERSE)
function loglik_gaussian(obs::AbstractMatrix{<:Real}, sim::AbstractMatrix, cov_mat::AbstractMatrix{<:Real})

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

    function logpost(theta)
        # Handle both Vector and NTuple inputs
        if theta isa AbstractVector
            par = (exp(theta[1]), exp(theta[2]), exp(theta[3]), exp(theta[4]))
        else
            par = (exp(theta[1]), exp(theta[2]), exp(theta[3]), exp(theta[4]))
        end
        sim = simulate_system(f!, u0, par, tspan, dt)
        # logPost(θ)=logL(e^θ)+logPrior(e^θ)+∑θ
        loglik_val = loglik_gaussian(obs, sim, cov_mat) 
        logprior_val = logprior_par(par)
        jacobian_val = sum(theta)
        
        return loglik_val + logprior_val + jacobian_val
    end
    return logpost
end

# Gradient Log-posterior
function grad_cpx!(grad, log_post::Function, x::AbstractVector{<:Real}; h::Float64 = 1e-200)
    # Work in complex to inject imaginary perturbations
    xc = ComplexF64.(x)
    for i in eachindex(x)
        xi = copy(xc)
        xi[i] += im*h
        val = log_post(xi)
        grad[i] = imag(val)/h
    end
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
        theta_p = copy(theta); theta_p[j] += h
        sim_p[:, :, j] = simulate_system(f!, u0, Tuple(exp.(theta_p)), tspan, dt)
        theta_m = copy(theta); theta_m[j] -= h
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


# Simplified Manifold MALA Kernel 
function IS_MP_sMALA_LV(f!::Function, u0, obs, cov_mat, tspan, dt, logprior_par::Function, 
                        init_par::AbstractVector{<:Float64}; seq::AbstractMatrix{<:Float64},
                        N_prop::Integer, N_iter::Integer, step_size::Float64=0.12)

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
        mu_x = xI + alpha * (metric.InvFisherInfo * grad_cpx!(grad, f_logpost, xI))
        z = mu_x + (step_size*CovScaling) * (metric.L*quantile.(Normal(), wcud[1, 1:D]))
        # Sample N proposals with MALA kernel K(yi|z)
        # Drift sm-MALA with ∇logπ(z​)
        metric_z = Tensor_Metric_sm(f!, u0, z, tspan, dt, cov_mat)
        mu_z     = z + alpha * (metric_z.InvFisherInfo * grad_cpx!(grad, f_logpost, z))

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
            mu_yi    = yi + alpha * (metric_y.InvFisherInfo * grad_cpx!(grad, f_logpost, yi))

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
)
end


function lotka_volterra!(du, u, p, t)
    alpha, beta, delta, gamma = p
    x, y = u
    du[1] =  alpha*x - beta*x*y
    du[2] = -gamma*y + delta*x*y
end

Theta_true = (1.5, 0.1, 0.075, 1.0); tspan = (0.0, 10.0); dt = 0.02; u0 = [5.0, 5.0]; 
lok_volt = ODEProblem(lotka_volterra!, u0, tspan, Theta_true);
sol = solve(lok_volt, Tsit5(), saveat=dt);   

# Noisy Data 
sigma_eta = 1 * I(2)              
obs_noisy = Array(sol) .+ rand(MvNormal(zeros(2), sigma_eta), size(sol, 2)); 

# Prior distributions for parameters: Gamma distributions
priors = (
          Gamma(1, 1),   # alpha
          Gamma(1, 1),   # beta 
          Gamma(1, 1),   # delta
          Gamma(1, 1)    # gamma
)

logprior_par = p -> begin
    # Handle complex parameters by taking real parts for logpdf evaluation
    p_real = real.(p)
    result = logpdf(priors[1], p_real[1]) + logpdf(priors[2], p_real[2]) + logpdf(priors[3], p_real[3]) + logpdf(priors[4], p_real[4])
    
    # If input was complex, return complex result to preserve gradient information
    if any(x -> x isa Complex, p)
        return Complex{Float64}(result)
    else
        return result
    end
end

a = [1.5, 0.1, 0.1, 2.0]
init_par = log.(a)

# function IS_MP_sMALA_LV(f!::Function, u0, obs, cov_mat, tspan, dt, logprior_par::Function, 
#                         init_par::AbstractVector{<:Float64}; seq::AbstractMatrix{<:Float64},
#                         N_prop::Integer, N_iter::Integer, step_size::Float64=0.12)

N_prop = 250 ; N_iter = 100
seq = rand(N_iter*(N_prop+1), 5)

out = IS_MP_sMALA_LV( lotka_volterra!, u0, obs_noisy, sigma_eta, tspan, dt, logprior_par, init_par; seq = seq, N_prop = N_prop, N_iter = N_iter, step_size = 2.0 );

out.chain



#-------------------------------------------------------------------------------------------------------------# 

chain = out.chain
iters = 1:size(chain, 1)

# Create trace plots
gr()
p1 = plot(iters, chain[:, 1], label="α", title="Trace of α", xlabel="Iteration", ylabel="Value")
hline!(p1, [1.5], linestyle=:dash, color=:red)
p2 = plot(iters, chain[:, 2], label="β", title="Trace of β", xlabel="Iteration", ylabel="Value")
hline!(p2, [0.1], linestyle=:dash, color=:red)
p3 = plot(iters, chain[:, 3], label="δ", title="Trace of δ", xlabel="Iteration", ylabel="Value")
hline!(p3, [0.075], linestyle=:dash, color=:red)
p4 = plot(iters, chain[:, 4], label="γ", title="Trace of γ", xlabel="Iteration", ylabel="Value")
hline!(p4, [1.0], linestyle=:dash, color=:red)

# Combine in a 4×1 layout
plot(p1, p2, p3, p4, layout=(4,1), size=(900,800), legend=false)


####--------------------------------------------------------------------------------------------####

lok_volt_new = ODEProblem(lotka_volterra!, u0, tspan, tuple(exp.(out.chain[end,:])...))
sol_new = solve(lok_volt_new, Tsit5(), saveat = dt)

# Use middle iteration instead of fixed index 500
mid_iter = max(1, div(size(out.chain, 1), 2))
lok_volt_t1 = ODEProblem(lotka_volterra!, u0, tspan, tuple(exp.(out.chain[mid_iter,:])...))
sol_new_t1 = solve(lok_volt_t1, Tsit5(), saveat = dt)

lok_volt_initial = ODEProblem(lotka_volterra!, u0, tspan, tuple(a...))
sol_initial = solve(lok_volt_initial, Tsit5(), saveat = dt)

p_overlay = plot(sol[1, :], sol[2, :],
    linewidth = 1.5, color = "blue", label = "True orbit",
    xlabel = "x", ylabel = "y",
    title = "Lotka–Volterra Orbits (Overlayed)",
    grid = true, gridalpha = 0.3, fontsize = 8)

plot!(p_overlay, sol_initial[1, :], sol_initial[2, :],
    linewidth = 1.5, color = "black", label = "Initial orbit")

plot!(p_overlay, sol_new_t1[1, :], sol_new_t1[2, :],
    linewidth = 1, color = "black", linestyle = :dashdot, label = "Mid-chain orbit")

plot!(p_overlay, obs_noisy[1, :], obs_noisy[2, :],
    linewidth = 1.5, color = "red", label = "Observed orbit")

plot!(p_overlay, sol_new[1, :], sol_new[2, :],
    linewidth = 1.5, color = "green", label = "Reconstructed orbit")

display(p_overlay)

####--------------------------------------------------------------------------------------------####