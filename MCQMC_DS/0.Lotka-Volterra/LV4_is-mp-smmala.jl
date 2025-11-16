include("C:/Users/mussi/Documents/Manhattan/Leuven/MCQMC/Data_Libraries/1.Libraries_setup.jl")
include("C:/Users/mussi/Documents/Manhattan/Leuven/MCQMC/Data_Libraries/0.mcmc_diagnostic.jl")
using ForwardDiff
using SciMLSensitivity 
using StaticArrays
using Random 

#####################################################################################################################
########################################         IS-MP-smMALA (MC)        ###########################################
############################            Automatic differentiation Black Box        ##################################
##########################    Lotka-Volterra 4D (alpha, beta, delta and gamma)        ###############################
#####################################################################################################################

# ODE Solver: Tsitouras 5/4 Runge–Kutta
function simulate_system(f!::Function, u0, par, tspan::Tuple, dt)
    prob = ODEProblem(f!, u0, tspan, par)
    sol = solve(prob, Tsit5(); saveat=dt, reltol=1e-6, abstol=1e-8, maxiters=10000, save_everystep=false)     
    return Array(sol)
end

# Struct to save Cholesky factorization
struct Chol_Save{T}
    chol_cov::Cholesky{T,Matrix{T}} 
    logdet :: T 
end

function CholSave(cov::AbstractMatrix{<:Real})
    F = cholesky(Symmetric(Matrix(cov)))
    logdet = 2*sum(log, diag(F.U))   
    return Chol_Save(F, logdet)
end

# Gaussian log-likelihood: (n*K/2)*log(2π) - (n/2)*log(det(V)) - (1/2)*Σ_{j=1}^n (x_j - μ)' * inv(V) * (x_j - μ)
function loglik_gaussian(obs::AbstractMatrix{<:Real}, sim::AbstractMatrix, chol_save::Chol_Save, d::Integer, K::Integer)

    @assert size(obs) == size(sim)

    E = obs .- sim                       
    Y = similar(sim)                     
    Y = chol_save.chol_cov.L \ E
    sse = sum(abs2, Y)

    return -0.5 * (K*d*log(2π) + K*chol_save.logdet + sse)
end

# Log-posterior in log-parameter space (Jacobian adjustment included)
function build_log_posterior(f!::Function, u0, obs, save_chol_cov::Chol_Save, tspan, dt, logprior_par::Function, d::Integer, K::Integer)
    
    function logpost(theta::AbstractVector)
        par = exp.(theta)
        sim = simulate_system(f!, u0, par, tspan, dt)
        return loglik_gaussian(obs, sim, save_chol_cov, d, K) + logprior_par(par) + sum(theta)
    end
    return logpost
end

# Gradient: Automatic Differentiation 
# The solver implicitly integrates the sensitivity equation, this is the reason of the Dual: u(t) and S_j(t)=∂u(t)/∂θ_j
function grad!(grad::AbstractVector{Float64}, log_post::Function, x::AbstractVector{<:Real}, cfg)
    ForwardDiff.gradient!(grad, log_post, x, cfg)                   
    return grad
end

# Riemannian Metric: Expected Fisher Information Matrix
function Tensor_Metric_sm( f!, u0, theta::AbstractVector{<:Real}, tspan, dt, d::Integer, K::Integer, save_chol_cov::Chol_Save, priors; λ::Float64 = 1e-6)

    D = length(theta)
    chol_cov = save_chol_cov.chol_cov

    function sim_vec(theta_local::AbstractVector)
        return vec(simulate_system(f!, u0, exp.(theta_local), tspan, dt))
    end

    cfg = ForwardDiff.JacobianConfig(sim_vec, theta, ForwardDiff.Chunk{4}())
    J = ForwardDiff.jacobian(sim_vec, theta, cfg)   

    G = zeros(eltype(J), D, D)

    for t in 1:K
    r1 = (t-1)*d + 1; r2 = t*d
    Jt  = @view J[r1:r2, :]     
    Jt_L   = chol_cov.L \ Jt       
    G  += transpose(Jt_L) * Jt_L
    end

    par = exp.(theta)              
    prior_diag = similar(par)

    for i in eachindex(par)                  
        scale = Distributions.scale(priors[i])  # Gamma(location, scale)
        prior_diag[i] = par[i] / scale          # -d²/dθ² log prior(e^θ)+θ
    end

    G .+= Diagonal(prior_diag)

    G = Symmetric(G + λ * I)
    chol_mat = cholesky(G)
    L = chol_mat.L \ I
    InvG  = chol_mat \ I

    return (FisherInfo = G, InvFisherInfo = InvG, L = L)
end

# Simplified Manifold MALA Kernel 
function IS_MP_sMALA_LV(f!::Function, u0, obs, d::Integer, K::Integer, cov_mat, tspan, dt, priors, init_par; seq::AbstractMatrix{<:Float64}, N_prop::Integer, N_iter::Integer, step_size::Float64)

    D = length(init_par)
    @assert D == 4 "This implementation expects 4 log-parameters."
    @assert size(seq,2) ≥ D+1 "seq needs ≥ D Normal-quantile columns + 1 resampling column."

    # Saving Cholesky Decomposition
    save = CholSave(cov_mat)

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
    max_stall  = ceil(0.1  * N_iter)
    tol_move   = 1e-4

    # MALA constants 
    alpha = (step_size^2) / 2
    CovScaling = 1

    # Log-prior function
    logprior_par = p -> (logpdf(priors[1], p[1]) + logpdf(priors[2], p[2]) + logpdf(priors[3], p[3]) + logpdf(priors[4], p[4]))

    # Definition Log-posterior distribution
    f_logpost = build_log_posterior(f!, u0, obs, save, tspan, dt, logprior_par, d, K)
    cfg_1 = ForwardDiff.GradientConfig(f_logpost, init_par, ForwardDiff.Chunk{D}())
   
    row = 1
    for l in 1:N_iter
        wcud = seq[row:row + N_prop, :]
        row += (N_prop + 1)
        # K(xi,xj) = K(xi,z)K(z,xj)
        # Sample N+1 PROPOSALS: current state and N new ones
        proposals[1, :, l] = xI
        metric = Tensor_Metric_sm(f!, u0, xI, tspan, dt, d, K, save, priors)
        # Sample bridge variable z with MALA kernel K(z|xI)
        # Drift sm-MALA with ∇logπ(xI​)
        mu_x = xI + alpha * (metric.InvFisherInfo * grad!(grad, f_logpost, xI, cfg_1))
        z = mu_x + (step_size*CovScaling) * (metric.L*quantile.(Normal(), wcud[1, 1:D]))
        # Sample N proposals with MALA kernel K(yi|z)
        # Drift sm-MALA with ∇logπ(z​)
        metric_z = Tensor_Metric_sm(f!, u0, z, tspan, dt, d, K, save, priors)
        mu_z     = z + alpha * (metric_z.InvFisherInfo * grad!(grad, f_logpost, z, cfg_1))

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
            log_post_i[i] = f_logpost(yi)

            # μ(yi): Langevin drift 
            metric_y = Tensor_Metric_sm(f!, u0, yi, tspan, dt, d, K, save, priors)
            mu_yi    = yi + alpha * (metric_y.InvFisherInfo * grad!(grad, f_logpost, yi, cfg_1))

            # FORWARD: log K(z | yi) = -1/2 (z-μ(yi))' Prec_y (z-μ(yi)) + 1/2 log|Prec_y|
            Prec_y = (1 / (step_size^2 * CovScaling^2)) .* metric_y.FisherInfo
            Ry     = cholesky(Symmetric(Prec_y)).U
            diff_fwd  = z .- mu_yi                 
            y1     = Ry * diff_fwd
            qf     = dot(y1, y1)
            logK_yi_z[i] = -0.5*qf + sum(log, diag(Ry)) 

            # BACKWARD: log K(yi | z) = -1/2 (yi-μ(z))' Prec_z (yi-μ(z))
            # metric_z = Tensor_Metric(f!, u0, z, tspan, dt, cov_mat)
            diff_bwd = yi .- mu_z
            y2    = Rz * diff_bwd
            qb    = dot(y2, y2)
            logK_z_yi[i] = -0.5*qb

        end

        # Importance weights w_i ∝ π(yi) K(yi,y\i)
        logw .= log_post_i .+ logK_yi_z  .+ sum(logK_z_yi) .- logK_z_yi
        logw .-= maximum(logw)
        logZ = log(sum(exp.(logw)))
        w[l, 1:(N_prop+1)] .= exp.(logw .- logZ)

        # Resampling step and Update xI
        # for j in 1:(N_prop + 1)
        #     I_new = findfirst(cumsum(w[l,:]) .>= wcud[j, D + 1])
        # end 

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
            weights = w)      

end

#####################################################################################################################################################
#####################################################################################################################################################

# Lotka-Volterra Model Definition
function lotka_volterra!(du, u, p, t)
    alpha, beta, delta, gamma = p
    x, y = u
    du[1] =  alpha*x - beta*x*y
    du[2] = -delta*y + gamma*x*y
end

# Simulate True Data
Theta_true = (1.5, 0.1, 1.0, 0.075); tspan = (0.0, 8.0); dt = 0.02; u0 = [5.0, 5.0]; 
lok_volt = ODEProblem(lotka_volterra!, u0, tspan, Theta_true);
sol = solve(lok_volt, Tsit5(), saveat=dt);   

# Noisy Data 
sigma_eta = 0.25 * I(2)              
obs_noisy = Array(sol) .+ rand(MvNormal(zeros(2), sigma_eta), size(sol, 2)); 

size(obs_noisy)

#----------------------------------------------------------------------------------------------------------------#
p1 = plot(sol[1,:], sol[2,:], linewidth = 1.5, xlabel = "x", ylabel = "y", legend = false, grid = true, gridalpha = 0.3, title = "True Orbit", fontsize =8)
p2 = plot(obs_noisy[1,:], obs_noisy[2,:], linewidth = 1.5, xlabel = "x", ylabel = "y", legend = false, color = "red", grid = true, gridalpha = 0.3, title = "Observed Orbit", fontsize =8)
lv_orbits = plot( p1, p2, layout = (1, 2), size = (700, 350), bottom_margin=5mm, top_margin=5mm, left_margin=5mm)

#----------------------------------------------------------------------------------------------------------------#

priors = (
          Gamma(2, 1),    # alpha
          Gamma(1, 0.2),  # beta 
#         Gamma(2, 1),    # beta
          Gamma(2, 1),    # delta
          Gamma(1, 0.2)   # gamma
#         Gamma(2, 1),    # gamma          
)

N_prop = 25 ; N_iter =  1000 
Random.seed!(1234)
seq = rand(N_iter*(N_prop+1), 5)
a = [2,0.5,2,0.5]
init_par = log.(a)

############################################ Step size tuning  ################################################
step_size = [0.35, 0.2, 0.1, 0.05, 0.025, 0.01]
pre_runs = Any[]; plots    = Any[]

for eps in step_size
    tuning_time = @elapsed pre_run = IS_MP_sMALA_LV( lotka_volterra!, u0, obs_noisy, size(obs_noisy, 1), size(obs_noisy, 2), 
                                    sigma_eta, tspan, dt, priors, init_par; seq = seq, N_prop = N_prop, N_iter = 1, step_size = eps )

    println("Step_size = $(eps);  Execution time = $(tuning_time) sec")
    push!(pre_runs, pre_run)

    p = bar(pre_run.weights[1, :], legend = false, xlabel = "Index", ylabel = "Weights", title  = "Starting Weights \n ε = $(eps)",
            grid   = true, ylim   = (0, maximum(pre_run.weights) * 1.1) )

    fname = "weights_start_eps_$(replace(string(eps), "." => "_")).png"
    savefig(p, fname); push!(plots, p)
end
plot_weight_start = plot(plots...; layout = (2, 3), size = (1200, 800), top_margin = 3mm, left_margin = 3mm, bottom_margin = 3mm, plot_title = "Step Size Tuning")
savefig(plot_weight_start, raw"C:\Users\mussi\Documents\Manhattan\Leuven\MCQMC\Plots\is-mp-smmala_ad\tuning_eps.png")

########################################## Run sm-MALA IS-MCQMC ###############################################

mcqmc_time = @elapsed out = IS_MP_sMALA_LV(lotka_volterra!, u0, obs_noisy, size(obs_noisy, 1), size(obs_noisy, 2), sigma_eta, tspan, dt, 
                                           priors, init_par; seq=seq, N_prop=N_prop, N_iter=N_iter, step_size=0.05);

println("Execution time: $(mcqmc_time) sec")
exp.(out.chain)

acf_log = autocor(exp.(out.chain[:, 2] ), 1:100)         

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












# function simulate_system(f!::Function, u0, par, tspan::Tuple, dt)
#     prob = ODEProblem(f!, u0, tspan, par)
#     # Optimized solver settings for performance
#     sol  = solve(prob, Tsit5(); saveat=dt, reltol=1e-6, abstol=1e-8, maxiters=10000, save_everystep=false)       
#     return Array(sol)
# end

# # Struct to save Cholesky factorization
# struct Chol_Save{T}
#     chol_cov::Cholesky{T,Matrix{T}} 
#     logdet :: T 
# end

# function CholSave(cov::AbstractMatrix{<:Real})
#     F = cholesky(Symmetric(Matrix(cov)))
#     logdet = 2*sum(log, diag(F.U))   
#     return Chol_Save(F, logdet)
# end

# # Gaussian log-likelihood with known noise covariance 
# # function loglik_gaussian(obs::AbstractMatrix{<:Real}, sim::AbstractMatrix, chol_cov_mat)

# #     d, K = size(sim)
# #     # Use eltype of sim to handle both Float64 and ForwardDiff.Dual types
# #     T = eltype(sim)
# #     sse = zero(T)
    
# #     # Allocate with correct type for automatic differentiation
# #     err = similar(sim, d)  
# #     y = similar(err)
    
# #     @inbounds @views for t in 1:K
# #         err .= obs[:, t] .- sim[:, t]     
# #         ldiv!(y, chol_cov_mat.U, err)        
# #         sse += dot(y, y)                 # e_t' Σ^{-1} e_t = y'y
# #     end

# #     logdet_term = sum(log, diag(chol_cov_mat.U))
# #     return -(K*d/2) * log(2π) - K*logdet_term - 0.5 * sse
# # end

# function loglik_gaussian(obs::AbstractMatrix{<:Real}, sim::AbstractMatrix, chol_save::Chol_Save, d::Integer, K::Integer)

#     @assert size(obs) == size(sim)

#     E = obs .- sim                       
#     Y = similar(sim)                     
#     Y = chol_save.chol_cov.L \ E
#     sse = sum(abs2, Y)

#     return -0.5 * (K*d*log(2π) + K*chol_save.logdet + sse)
# end

# # Log-posterior
# function build_log_posterior(f!::Function, u0, obs, save_chol_cov::Chol_Save, tspan, dt, logprior_par::Function, d::Integer, K::Integer)
    
#     function logpost(theta::AbstractVector)
#         par = exp.(theta)
#         sim = simulate_system(f!, u0, par, tspan, dt)
#         return loglik_gaussian(obs, sim, save_chol_cov, d, K) + logprior_par(par) + sum(theta)
#     end
#     return logpost
# end

# # Gradient via ForwardDiff
# function grad!(grad::AbstractVector{Float64}, log_post::Function, x::AbstractVector{<:Real})
#     cfg = ForwardDiff.GradientConfig(log_post, x)
#     ForwardDiff.gradient!(grad, log_post, x, cfg)
#     return grad
# end

# function Tensor_Metric_sm(f!, u0, theta::AbstractVector{<:Real}, tspan, dt, d::Integer, K::Integer, save_chol_cov::Chol_Save; λ::Float64 = 1e-1)
#     D = 4
#     chol_cov = save_chol_cov.chol_cov                               # Precompute Cholesky Σ^{-1} 

#     sim_vec = function (th::AbstractVector)
#         Y =  simulate_system(f!, u0, Tuple(exp.(th)), tspan, dt)
#         return vec(Y)                                                       
#     end

#     cfg = ForwardDiff.JacobianConfig(sim_vec, theta, ForwardDiff.Chunk{4}())
#     J = ForwardDiff.jacobian(sim_vec, theta, cfg)                           

#     G = zeros(eltype(J), D, D)
#     for t in 1:K
#         r1 = (t-1)*d + 1; r2 = t*d
#         Jt = @view J[r1:r2, :]                                              
#         MJt = copy(Jt)                                                      
#         ldiv!(chol_cov, MJt)                                                   
#         mul!(G, transpose(Jt), MJt, 1.0, 1.0)                               
#     end

#     G = Symmetric(G + λ * I)
#     chol_mat = cholesky(G)
#     L = chol_mat.L \ I
#     InvFisherInfo = chol_mat\I

#     return (FisherInfo = G, InvFisherInfo = InvFisherInfo, L = L)
# end