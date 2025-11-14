include("C:/Users/mussi/Documents/Manhattan/Leuven/MCQMC/Data_Libraries/1.Libraries_setup.jl")
include("C:/Users/mussi/Documents/Manhattan/Leuven/MCQMC/Data_Libraries/0.mcmc_diagnostic.jl")
using ForwardDiff
using SciMLSensitivity 
using StaticArrays 

function simulate_system(f!::Function, u0, par, tspan::Tuple, dt)
    prob = ODEProblem(f!, u0, tspan, par)
    # Optimized solver settings for performance
    sol  = solve(prob, Tsit5();
                 saveat=dt,
                 reltol=1e-6, abstol=1e-8,   
                 maxiters=10000,             
                 save_everystep=false)       
    return Array(sol)
end

# Struct to save Cholesky factorization
struct Chol_Save{T}
    chol_cov::Cholesky{T,Matrix{T}}  
end

Create_Chol_Save(cov_mat::AbstractMatrix) = Chol_Save(cholesky(Symmetric(Matrix(cov_mat))))

# Gaussian log-likelihood with known noise covariance 
function loglik_gaussian(obs::AbstractMatrix{<:Real}, sim::AbstractMatrix, chol_cov_mat)

    d, K = size(sim)
    # Use eltype of sim to handle both Float64 and ForwardDiff.Dual types
    T = eltype(sim)
    sse = zero(T)
    
    # Allocate with correct type for automatic differentiation
    err = similar(sim, d)  
    y = similar(err)
    
    @inbounds @views for t in 1:K
        err .= obs[:, t] .- sim[:, t]     
        ldiv!(y, chol_cov_mat.U, err)        
        sse += dot(y, y)                 # e_t' Σ^{-1} e_t = y'y
    end

    logdet_term = sum(log, diag(chol_cov_mat.U))
    return -(K*d/2) * log(2π) - K*logdet_term - 0.5 * sse
end

# Log-posterior
function build_log_posterior(f!::Function, u0, obs, save_chol_cov::Chol_Save, tspan, dt, logprior_par::Function)
    # Pre-compute Cholesky decomposition once
    chol_cov = save_chol_cov.chol_cov

    function logpost(theta)
        # Handle both Vector and Tuple inputs - use StaticArrays for speed
        if theta isa AbstractVector
            # Convert to SVector for faster operations
            theta_static = SVector{4}(theta)
            par = (exp(theta_static[1]), exp(theta_static[2]), exp(theta_static[3]), exp(theta_static[4]))
        else
            par = (exp(theta[1]), exp(theta[2]), exp(theta[3]), exp(theta[4]))
        end
        sim = simulate_system(f!, u0, par, tspan, dt)
        return loglik_gaussian(obs, sim, chol_cov) + logprior_par(par) + sum(theta)
    end
    return logpost
end

# Gradient via ForwardDiff
function grad_ad!(grad::AbstractVector{Float64}, log_post::Function, x::AbstractVector{<:Real})
    g = ForwardDiff.gradient(log_post, x)
    @. grad = g
    return grad
end

# Gradient via finite differences (fallback)
function grad_fd!(grad::AbstractVector{Float64}, log_post::Function, x::AbstractVector{<:Real}; h::Float64=1e-6)
    n = length(x)
    for i in 1:n
        x_plus = copy(x); x_plus[i] += h
        x_minus = copy(x); x_minus[i] -= h
        grad[i] = (log_post(x_plus) - log_post(x_minus)) / (2*h)
    end
    return grad
end

# Tensor Metric: Exotected Fisher Information Matrix 
function Tensor_Metric_sm_AD(f!, u0, theta::AbstractVector{<:Real}, tspan, dt, d::Integer, K::Integer, save_chol_cov::Chol_Save; λ::Float64 = 1e-3)

    # Dimensions from data/model
    # If obs size is known elsewhere you may pass (d,K) in; here we infer from a primal sim.
    # sim0 = simulate_system(f!, u0, Tuple(exp.(theta)), tspan, dt)   # d × K
    D = 4
    @assert D == 4 "This implementation expects 4 log-parameters."

    # Precompute Σ^{-1} via a Cholesky for stable solves
    chol_cov = save_chol_cov.chol_cov

    # Define θ -> vec(sim(θ)) in column-major order (time slices stacked)
    sim_vec = function (th::AbstractVector)
        Y = simulate_system(f!, u0, Tuple(exp.(th)), tspan, dt)  # d×K
        return vec(Y)                                            # (dK) vector
    end

    # Jacobian J = ∂ vec(sim) / ∂θ  with size (dK) × D
    # Provide a config to reduce allocations
    cfg = ForwardDiff.JacobianConfig(sim_vec, theta)
    J = ForwardDiff.jacobian(sim_vec, theta, cfg)                # (dK)×D

    # Accumulate Fisher information: G = Σ_t J_t' Σ^{-1} J_t
    G = zeros(eltype(J), D, D)

    # Work buffers to avoid repeated allocations
    # We treat each time-slice block of J (rows r:t indices) as a d×D matrix
    for t in 1:K
        r1 = (t-1)*d + 1
        r2 = t*d
        Jt = @view J[r1:r2, :]          # d×D

        # Compute Σ^{-1} * Jt via Cholesky: solve Σ * X = Jt
        # Use a temporary matrix to hold the solve result
        MJt = chol_cov \ Matrix(Jt)     # d×D

        # Rank-D update: G += Jt' * (Σ^{-1} * Jt)
        mul!(G, transpose(Jt), MJt, 1.0, 1.0)
    end

    G = Symmetric(G + λ * I)
    chol_mat = cholesky(G)
    L = chol_mat.U \ I(D)
    InvFisherInfo = chol_mat \ (chol_mat' \ I(D))
return (FisherInfo = G, InvFisherInfo = InvFisherInfo, L = L)
end

# More efficient version using ForwardDiff.jacobian directly
function Tensor_Metric_m_AD_efficient(f!, u0, θ::AbstractVector{<:Real}, tspan, dt, d::Integer, K::Integer, save_chol_cov::Chol_Save; λ::Float64=1e-6)
    
    D = length(θ)
    
    # Function that maps full parameter vector to Fisher Information Matrix (vectorized)
    function fisher_vector_function(theta_vec)
        metric = Tensor_Metric_sm_AD(f!, u0, theta_vec, tspan, dt, d, K, save_chol_cov; λ=λ)
        return vec(Matrix(metric.FisherInfo))  # Vectorize the D×D matrix to D² vector
    end
    
    # Compute base Fisher Information at θ
    base_metric = Tensor_Metric_sm_AD(f!, u0, θ, tspan, dt, d, K, save_chol_cov; λ=λ)
    G = base_metric.FisherInfo
    InvG = base_metric.InvFisherInfo  
    L = base_metric.L
    
    # Compute full Jacobian: ∂(vec(G))/∂θ with size (D²) × D
    try
        J_full = ForwardDiff.jacobian(fisher_vector_function, θ)
        
        # Extract derivatives ∂G/∂θⱼ from Jacobian
        dG = Vector{Matrix{Float64}}(undef, D)
        for j in 1:D
            dG_vec = J_full[:, j]  # j-th column contains ∂(vec(G))/∂θⱼ
            dG[j] = reshape(dG_vec, D, D)  # Reshape back to D×D matrix
        end
        
        return (G=G, InvG=InvG, L=L, dG=dG)
        
    catch e
        println("Warning: ForwardDiff failed in manifold metric computation, using finite differences fallback")
        # Fallback to finite differences if AD fails
        return Tensor_Metric_m_FD_fallback(f!, u0, θ, tspan, dt, d, K, save_chol_cov; λ=λ)
    end
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
                        f!::Function, u0, obs, d::Integer, K::Integer, cov_mat, tspan, dt, logprior_par::Function,
                        init_par::AbstractVector{<:Float64};  seq::AbstractMatrix{<:Float64}, N_prop::Integer, 
                        N_iter::Integer,step_size::Float64=0.12)

    D = length(init_par)
    save_chol_cov = Create_Chol_Save(cov_mat)
    # log-posterior
    f_logpost = build_log_posterior(f!, u0, obs, save_chol_cov, tspan, dt, logprior_par)

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
    max_stall  = ceil(0.025  * N_iter)
    tol_move   = 1e-4


    row = 1
    for l in 1:N_iter
        wcud = seq[row:row + N_prop, :]
        row += (N_prop + 1)

        # include current state among proposals
        proposals[1, :, l] = xI

        # metric and drift at current xI
        rtm_x = Tensor_Metric_m_AD_efficient(f!, u0, xI, tspan, dt, d, K, save_chol_cov)
        grad_ad!(grad, f_logpost, xI)
        mu_x = xI .+ ((eps^2) / 2) .* (rtm_x.InvG * grad) .+ (eps^2) .* christoffel_correction(rtm_x.InvG, rtm_x.dG)

        # bridge variable z ~ K(· | xI)
        z = mu_x .+ eps .* (rtm_x.L * quantile.(Normal(), wcud[1, 1:D]))

        # metric and drift at z
        rtm_z = Tensor_Metric_m_AD_efficient(f!, u0, z, tspan, dt, d, K, save_chol_cov)
        grad_ad!(grad, f_logpost, z)
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
            rtm_y = Tensor_Metric_m_AD_efficient(f!, u0, yi, tspan, dt, d, K, save_chol_cov)
            grad_ad!(grad, f_logpost, yi)
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

#----------------------------------------------------------------------------------------------------------------#
# Lotka-Volterra Model Definition
function lotka_volterra!(du, u, p, t)
    alpha, beta, delta, gamma = p
    x, y = u
    du[1] =  alpha*x - beta*x*y
    du[2] = -gamma*y + delta*x*y
end

Theta_true = (1.8,0.5,1,2.5); tspan = (0.0, 8.0); dt = 0.02; u0 = [10.0, 5.0]; 
lok_volt = ODEProblem(lotka_volterra!, u0, tspan, Theta_true);
sol = solve(lok_volt, Tsit5(), saveat=dt);   

# Noisy Data 
sigma_eta = 0.25 * I(2)              
obs_noisy = Array(sol) .+ rand(MvNormal(zeros(2), sigma_eta), size(sol, 2)); 

size(obs_noisy)

#----------------------------------------------------------------------------------------------------------------#
# p1 = plot(sol[1,:], sol[2,:], linewidth = 1.5, 
#           xlabel = "x", ylabel = "y",
#           legend = false, grid = true, gridalpha = 0.3, title = "True Orbit", fontsize =8)

# p2 = plot(obs_noisy[1,:], obs_noisy[2,:], linewidth = 1.5,
#          xlabel = "x", ylabel = "y",
#          legend = false, color = "red", grid = true, gridalpha = 0.3, title = "Observed Orbit", fontsize =8)

# lv_orbits = plot( p1, p2, layout = (1, 2), size = (700, 350), bottom_margin=5mm, top_margin=5mm)

#----------------------------------------------------------------------------------------------------------------#

priors = (
          Gamma(1, 3),   # alpha
          Gamma(1, 3),   # beta 
          Gamma(1, 3),   # delta
          Gamma(1, 3)    # gamma
)

logprior_par = p -> (logpdf(priors[1], p[1]) + logpdf(priors[2], p[2]) + logpdf(priors[3], p[3]) + logpdf(priors[4], p[4]))

N_prop = 25 ; N_iter =  10000 
seq = rand(N_iter*(N_prop+1), 5)
a = [1.,1.,1.,1.]
init_par = log.(a)

############################################ Step size tuning  ################################################
tuning_time = @elapsed pre_run = IS_MP_MMALA_LV(lotka_volterra!, u0, obs_noisy, size(obs_noisy, 1), size(obs_noisy, 2), sigma_eta, tspan, dt, logprior_par,
                                                init_par; seq=seq, N_prop=N_prop, N_iter=1, step_size=0.15);

println("Execution time: $(tuning_time) sec")

p_weights_start = bar(pre_run.weights[1,:], legend=false, xlabel="Index", ylabel="Weights", title="Weights - Start (Iter 1)", grid=true, ylim=(0, maximum(pre_run.weights)*1.1))
########################################## Run sm-MALA IS-MCQMC ###############################################
mcqmc_time = @elapsed out = IS_MP_MMALA_LV(lotka_volterra!, u0, obs_noisy, size(obs_noisy, 1), size(obs_noisy, 2), sigma_eta, tspan, dt, logprior_par,
                                                init_par; seq=seq, N_prop=N_prop, N_iter=N_iter, step_size=0.05);

println("Execution time: $(mcqmc_time) sec")
exp.(out.chain)

# @save raw"C:\Users\mussi\Documents\Manhattan\Leuven\MCQMC\MCQMC_DS\Results\LV_Mmala_mcqmc.jld2" out     
##-------------------------------------------------------------------------------------------------------------#
p_weights_start = bar(out.weights[1,:], legend=false, xlabel="Index", ylabel="Weights", 
                     title="Weights - Start (Iter 1)", grid=true, ylim=(0, maximum(out.weights)*1.1))

len = length(out.weights[:,1])
mid_iter = max(1, div(len, 2))
p_weights_mid = bar(out.weights[mid_iter,:], legend=false, xlabel="Index", ylabel="Weights", 
                   title="Weights - Middle (Iter $mid_iter)", grid=true, ylim=(0, maximum(out.weights)*1.1))

p_weights_end = bar(out.weights[end,:], legend=false, xlabel="Index", ylabel="Weights", 
                   title="Weights - End (Iter $(len))", grid=true, ylim=(0, maximum(out.weights)*1.1))

# Combine weight plots
p_weights_evolution = plot(p_weights_start, p_weights_mid, p_weights_end, layout=(1,3), size=(1200,400))
display(p_weights_evolution)
#-------------------------------------------------------------------------------------------------------------# 
rand_num = rand(1:100) 

chain = exp.(out.chain)
iters = 1:size(chain, 1)

p1 = plot(iters, chain[:, 1], label="α", title="Trace of α", xlabel="Iteration", ylabel="Value")
hline!(p1, [Theta_true[1]], linestyle=:dash, color=:red)
p2 = plot(iters, chain[:, 2], label="β", title="Trace of β", xlabel="Iteration", ylabel="Value")
hline!(p2, [Theta_true[2]], linestyle=:dash, color=:red)
p3 = plot(iters, chain[:, 3], label="δ", title="Trace of δ", xlabel="Iteration", ylabel="Value")
hline!(p3, [Theta_true[3]], linestyle=:dash, color=:red)
p4 = plot(iters, chain[:, 4], label="γ", title="Trace of γ", xlabel="Iteration", ylabel="Value")
hline!(p4, [Theta_true[4]], linestyle=:dash, color=:red)

p = plot(p1, p2, p3, p4, layout=(4,1), size=(900,800), legend=false)
savefig(p, raw"C:\Users\mussi\Documents\Manhattan\Leuven\MCQMC\Plots\SM-MCQMC\LV_TracePlots" * string(rand_num)* ".png")

####--------------------------------------------------------------------------------------------####

lok_volt_new = ODEProblem(lotka_volterra!, u0, tspan, tuple(exp.(out.chain[end,:])...))
sol_new = solve(lok_volt_new, Tsit5(), saveat = dt)

mid_iter = max(1, div(size(out.chain, 1), 2))
lok_volt_t1 = ODEProblem(lotka_volterra!, u0, tspan, tuple(exp.(out.chain[mid_iter,:])...))
sol_new_t1 = solve(lok_volt_t1, Tsit5(), saveat = dt)

lok_volt_initial = ODEProblem(lotka_volterra!, u0, tspan, tuple(a...))
sol_initial = solve(lok_volt_initial, Tsit5(), saveat = dt)

p_overlay = plot(sol[1, :], sol[2, :], linewidth = 1.5, color = "blue", label = "True orbit", xlabel = "x", ylabel = "y", title = "Lotka–Volterra Orbits (Overlayed)", grid = true, gridalpha = 0.3, fontsize = 8)
plot!(p_overlay, sol_initial[1, :], sol_initial[2, :], linewidth = 1.5, color = "black", label = "Initial orbit")
plot!(p_overlay, sol_new_t1[1, :], sol_new_t1[2, :], linewidth = 1, color = "black", linestyle = :dashdot, label = "Mid-chain orbit")
plot!(p_overlay, obs_noisy[1, :], obs_noisy[2, :], linewidth = 1.5, color = "red", label = "Observed orbit")
plot!(p_overlay, sol_new[1, :], sol_new[2, :], linewidth = 1.5, color = "green", label = "Reconstructed orbit")
savefig(p_overlay, raw"C:\Users\mussi\Documents\Manhattan\Leuven\MCQMC\Plots\SM-MCQMC\LV_Orbits" * string(rand_num)* ".png")

####--------------------------------------------------------------------------------------------####
lv_orbits_sm = plot(sol.t, obs_noisy[1,:], lw=2, label="x true",
    grid=true, gridalpha=0.3, legendfontsize=9,
    legend=:outertop, legendcolumns=2, legendborder=false,
    legend_foreground_color=:transparent,
    title="Lotka-Volterra Dynamics (sm-MALA)", fontsize=10)
plot!(sol.t, obs_noisy[2,:], lw=2, label="y true")
plot!(sol.t, sol_new[1,:], lw=2, ls=:solid, label="x sm-MALA", color=:blue)
plot!(sol.t, sol_new[2,:], lw=2, ls=:solid, label="y sm-MALA", color=:red)
xlabel!("Time")

savefig(lv_orbits_sm, raw"C:\Users\mussi\Documents\Manhattan\Leuven\MCQMC\Plots\SM-MCQMC\LV_TimeSeries" * string(rand_num)* ".png")

####--------------------------------------------------------------------------------------------####

p = exp.(out.chain[end,:])
# p = Theta_true
sse = 0.0; resid = zeros(size(obs_noisy, 1)); 
lv_end = ODEProblem(lotka_volterra!, u0, tspan, p);
sol = solve(lv_end, Tsit5(), saveat=dt);
F = cholesky(Symmetric(sigma_eta))
for t in 1:size(obs_noisy, 2)
    @views resid .= obs_noisy[:, t] .- sol[:, t]
    y = F.U \ resid
    sse += dot(y, y)
end
sse


####--------------------------------------------------------------------------------------------####

for i in 1:4
    println("ESS $(i): Parameters = ", ess_ips(out.chain[:, i]))
end


