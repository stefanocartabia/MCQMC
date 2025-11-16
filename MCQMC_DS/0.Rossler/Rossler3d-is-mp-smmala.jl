# https://en.wikipedia.org/wiki/R%C3%B6ssler_attractor
# http://www.scholarpedia.org/article/Hyperchaos
# https://journals.aps.org/pre/abstract/10.1103/PhysRevE.56.5069

include("C:/Users/mussi/Documents/Manhattan/Leuven/MCQMC/Data_Libraries/1.Libraries_setup.jl")
include("C:/Users/mussi/Documents/Manhattan/Leuven/MCQMC/Data_Libraries/0.mcmc_diagnostic.jl")
using ForwardDiff
using SciMLSensitivity 
using StaticArrays
using Random 

###############################################################################################################################################
######################################################         IS-MP-smMALA (QMC)        ######################################################
#############################################            Automatic differentiation Black Box        ###########################################
###################################################          3D Rossler System (a,b,c)         ################################################
###############################################################################################################################################

# ODE Solver: Rodas5P (stiff-aware) with fallback to increase maxiters
function simulate_system(f!::Function, u0, par, tspan::Tuple, dt)
    prob = ODEProblem(f!, u0, tspan, par)
    # Use Rodas5P for potentially stiff systems
    sol = solve(prob, Rodas5P(); saveat=dt, reltol=1e-6, abstol=1e-8, maxiters=100000, save_everystep=false)
    
    # Check if solution succeeded
    if sol.retcode != :Success
        @warn "ODE solver failed with retcode: $(sol.retcode)"
        return fill(NaN, length(u0), length(sol.t))
    end
    
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

    @assert size(obs) == size(sim) "Size mismatch: obs=$(size(obs)), sim=$(size(sim))"
    
    # Check for NaN values (failed simulation)
    if any(isnan, sim)
        return -Inf
    end

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
function grad!(grad::AbstractVector{Float64}, log_post::Function, x::AbstractVector{<:Real}, cfg)
    ForwardDiff.gradient!(grad, log_post, x, cfg)                   
    return grad
end

# Riemannian Metric: Expected Fisher Information Matrix w.r.t. parameters
function Tensor_Metric_sm(f!, u0, theta::AbstractVector{<:Real}, tspan, dt, d::Integer, K::Integer, save_chol_cov::Chol_Save, priors; λ::Float64 = 1e-6)

    D = length(theta)
    chol_cov = save_chol_cov.chol_cov

    function sim_vec(theta_local::AbstractVector)
        return vec(simulate_system(f!, u0, exp.(theta_local), tspan, dt))
    end

    cfg = ForwardDiff.JacobianConfig(sim_vec, theta, ForwardDiff.Chunk{3}())
    J = ForwardDiff.jacobian(sim_vec, theta, cfg)
    
    # Check dimensions match expectation
    expected_size = d * K
    if size(J, 1) != expected_size
        @warn "Jacobian size mismatch: got $(size(J,1)) rows, expected $(expected_size). Returning identity metric."
        # Return identity-like metric to avoid crash
        return Diagonal(ones(D)), Diagonal(ones(D)), Matrix{Float64}(I, D, D)
    end
    
    # Check for NaN values
    if any(isnan, J)
        @warn "NaN values detected in Jacobian. Returning identity metric."
        return Diagonal(ones(D)), Diagonal(ones(D)), Matrix{Float64}(I, D, D)
    end

    G = zeros(eltype(J), D, D)

    for t in 1:K
        r1 = (t-1)*d + 1
        r2 = t*d
        Jt  = @view J[r1:r2, :]     
        Jt_L   = chol_cov.L \ Jt       
        G  += transpose(Jt_L) * Jt_L
    end

    # Prior precision for Gamma distributions in log-parameter space
    par = exp.(theta)              
    prior_diag = similar(par)

    for i in eachindex(par)                  
        scale = Distributions.scale(priors[i])  # Gamma(shape, scale)
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
function IS_MP_sMALA(f!::Function, u0, obs, d::Integer, K::Integer, cov_mat, tspan, dt, priors, init_par; seq::AbstractMatrix{<:Float64}, N_prop::Integer, N_iter::Integer, step_size::Float64)

    D = length(init_par)
    @assert D == 3 "This implementation expects 3 log-parameters."
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
    log_post_i = zeros(N_prop + 1)                              
    mu_yi = zeros(D)                                
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
    logprior_par = p -> (logpdf(priors[1], p[1]) + logpdf(priors[2], p[2]) + logpdf(priors[3], p[3]))

    # Definition Log-posterior distribution
    f_logpost = build_log_posterior(f!, u0, obs, save, tspan, dt, logprior_par, d, K)
    cfg_1 = ForwardDiff.GradientConfig(f_logpost, init_par, ForwardDiff.Chunk{D}())
   
    row = 1
    for l in 1:N_iter
        wcud = seq[row:row + N_prop, :]
        row += (N_prop + 1)
        
        # Sample N+1 PROPOSALS: current state and N new ones
        proposals[1, :, l] = xI
        metric = Tensor_Metric_sm(f!, u0, xI, tspan, dt, d, K, save, priors)
        
        # Sample bridge variable z with MALA kernel K(z|xI)
        mu_x = xI + alpha * (metric.InvFisherInfo * grad!(grad, f_logpost, xI, cfg_1))
        z = mu_x + (step_size*CovScaling) * (metric.L*quantile.(Normal(), wcud[1, 1:D]))
        
        # Sample N proposals with MALA kernel K(yi|z)
        metric_z = Tensor_Metric_sm(f!, u0, z, tspan, dt, d, K, save, priors)
        mu_z     = z + alpha * (metric_z.InvFisherInfo * grad!(grad, f_logpost, z, cfg_1))

        @inbounds for j in 2:(N_prop + 1)
            proposals[j, :, l] = mu_z + (step_size*CovScaling) * (metric_z.L*quantile.(Normal(), wcud[j, 1:D]))
        end

        # Compute importance weights
        Prec_z   = (1 / (step_size^2 * CovScaling^2)) .* metric_z.FisherInfo
        Rz       = cholesky(Symmetric(Prec_z)).U

        for i in 1:(N_prop + 1)
            yi = proposals[i, 1:D, l] 

            # Posterior log-density: log π(yi)
            log_post_i[i] = f_logpost(yi)

            # μ(yi): Langevin drift 
            metric_y = Tensor_Metric_sm(f!, u0, yi, tspan, dt, d, K, save, priors)
            mu_yi    = yi + alpha * (metric_y.InvFisherInfo * grad!(grad, f_logpost, yi, cfg_1))

            # FORWARD: log K(z | yi)
            Prec_y = (1 / (step_size^2 * CovScaling^2)) .* metric_y.FisherInfo
            Ry     = cholesky(Symmetric(Prec_y)).U
            diff_fwd  = z .- mu_yi                 
            y1     = Ry * diff_fwd
            qf     = dot(y1, y1)
            logK_yi_z[i] = -0.5*qf + sum(log, diag(Ry)) 

            # BACKWARD: log K(yi | z)
            diff_bwd = yi .- mu_z
            y2    = Rz * diff_bwd
            qb    = dot(y2, y2)
            logK_z_yi[i] = -0.5*qb + sum(log, diag(Rz))
        end

        # Log weights
        sum_logKzi = sum(logK_z_yi)

        @inbounds for i in 1:(N_prop + 1)
            logw[i] = log_post_i[i] + logK_yi_z[i] + (sum_logKzi - logK_z_yi[i])
        end

        # Exponentiate and normalize 
        logw .-= maximum(logw)
        w[l,:] .= exp.(logw)
        w[l,:] ./= sum(w[l,:])

        # Resampling
        vprime = wcud[end, D+1]
        csum = 0.0; I_new = 1
        @inbounds for i in 1:(N_prop + 1)
            csum += w[l,i]
            if vprime <= csum
                I_new = i; break
            end
        end
        accept_proxy += (1.0 - w[l,I_new])

        # Update state
        xI = proposals[I_new, :, l]
        chain[l,:] = xI

        # Early stopping check
        if l > 1
            diff_move = norm(chain[l,:] .- chain[l-1,:])
            if diff_move < tol_move
                stall_cnt += 1
                if stall_cnt ≥ max_stall
                    l_effective = l
                    break
                end
            else
                stall_cnt = 0
            end
        end
    end

    if l_effective == 0
        l_effective = N_iter
    end

    chain = chain[1:l_effective, :]
    w     = w[1:l_effective, :]
    proposals = proposals[:, :, 1:l_effective]

    return (
            proposals = proposals,
            chain = chain,
            accept_proxy = accept_proxy / l_effective,
            grad = grad,
            length = l_effective,
            weights = w)      

end

#####################################################################################################################################################
#####################################################################################################################################################

# Rossler system with parameters (a, b, c)
function rossler!(du, u, p, t)
    a, b, c = p
    x, y, z = u
    du[1] = -y - z
    du[2] =  x + a*y
    du[3] =  b + z*(x - c)
end

# ---------------------------------------------------------------------------------------------------------------------------------------------------#
# # Find suitable time window for inference
# tspan = (0.0, 20.0); u0 = [1.0, 1.0, 1.0]; dt = 0.1
# par_true = (.1, 0.1, 14)  # a, b, c
# rossler_sys = ODEProblem(rossler!, u0, tspan, par_true)
# sol = solve(rossler_sys, Tsit5(), saveat=dt)   

# idx = findall(t -> 15.0 <= t <= 16.0, sol.t)
# t_sub = sol.t[idx]
# x_sub = sol[1, idx]
# y_sub = sol[2, idx]
# z_sub = sol[3, idx]

# (x_sub[1], y_sub[1], z_sub[1])

#----------------------------------------------------------- Data Generation -------------------------------------------------------------------------#
# Simulate True Data
tspan = (1, 20.0); u0 = [1.0, 1.0, 1.0]; dt = 0.02
par_true = (.1, .1, 14)  # a, b, c
rossler_sys = ODEProblem(rossler!, u0, tspan, par_true)
sol = solve(rossler_sys, Tsit5(), saveat=dt)

# Noisy Data 
sigma_eta = 0.001 * I(3)              
obs_noisy = Array(sol) .+ rand(MvNormal(zeros(3), sigma_eta), size(sol, 2));

#----------------------------------------------------------- Orbits  ---------------------------------------------------------------------------------#
p1 = plot(sol[1,:], sol[2,:], sol[3,:], linewidth = 1.5, title = "\nTrue Rossler Attractor", 
          xlabel = "x", ylabel = "y", zlabel = "z", legend = false, grid = true, gridalpha = 0.3 )
p2 = plot(obs_noisy[1,:], obs_noisy[2,:], obs_noisy[3,:], linewidth = 1.5, title = "\nObserved Rossler Attractor", 
          xlabel = "x", ylabel = "y", zlabel = "z", legend = false, color = "red", grid = true, gridalpha = 0.3)
plot_traj = plot( p1, p2, layout = (1, 2), size = (1100, 400), top_margin = 3mm)
savefig(plot_traj, raw"C:\Users\mussi\Documents\Manhattan\Leuven\MCQMC\Plots\is-mp-smmala_ad\Rossler-Par\MC_trajectory.png")

#----------------------------------------------------------- Time Series  ---------------------------------------------------------------------------------#
p1 = scatter(sol.t, obs_noisy[1,:], markersize=3, alpha=0.5, label="x obs", color=:lightblue,
             grid=true, gridalpha=0.3, ylabel="x", legend=:outertop)
plot!(p1, sol.t, sol[1,:], lw=2, label="x true", color=:blue)

p2 = scatter(sol.t, obs_noisy[2,:], markersize=3, alpha=0.5, label="y obs", color=:lightcoral,
             grid=true, gridalpha=0.3, ylabel="y", legend=:outertop)
plot!(p2, sol.t, sol[2,:], lw=2, label="y true", color=:red)

p3 = scatter(sol.t, obs_noisy[3,:], markersize=3, alpha=0.5, label="z obs", color=:lightgreen,
             grid=true, gridalpha=0.3, ylabel="z", xlabel="Time", legend=:outertop)
plot!(p3, sol.t, sol[3,:], lw=2, label="z true", color=:green)

Rossler_ts = plot(p1, p2, p3, layout=(3,1), size=(1000, 800), 
                plot_title="\n Rossler System Dynamics", 
                left_margin=5mm, bottom_margin=5mm, top_margin=5mm, legendcolumns=2,
                legendborder=false, legend_foreground_color=:transparent)

savefig(Rossler_ts, raw"C:\Users\mussi\Documents\Manhattan\Leuven\MCQMC\Plots\is-mp-smmala_ad\Rossler-Par\MC_trajectory_ts.png")

#-------------------------------------------------------------------------------------------------------------------------------------------------------#

# Priors for parameters (Gamma distributions)
priors = (
          Gamma(0.5, 1.0),    
          Gamma(0.5, 1.0),     
          Gamma(20, 0.5)           
)

N_prop = 50 ; N_iter = 500
Random.seed!(1234)
wcud = rand(N_iter*(N_prop+1), 4)
a = [5.0, 5.0, 5.0]  
init_par = log.(a)

############################################################## Step size tuning  ########################################################################

step_size = [0.025, 0.001]
pre_runs = Any[]; plots = Any[]

for eps in step_size
    tuning_time = @elapsed pre_run = IS_MP_sMALA(rossler!, u0, obs_noisy, size(obs_noisy, 1), size(obs_noisy, 2), 
                                    sigma_eta, tspan, dt, priors, init_par; seq = wcud, N_prop = N_prop, N_iter = 1, step_size = eps)

    println("Step_size = $(eps);  Execution time = $(tuning_time) sec")
    push!(pre_runs, pre_run)

    p = bar(pre_run.weights[1, :], legend = false, xlabel = "Index", ylabel = "Weights", title  = "Starting Weights \n ε = $(eps)",
            grid   = true, ylim   = (0, maximum(pre_run.weights) * 1.1) )

    push!(plots, p)
end
plot_weight_start = plot(plots...; layout = (2, 3), size = (1200, 800), top_margin = 3mm, left_margin = 3mm, bottom_margin = 3mm, plot_title = "Step Size Tuning")
savefig(plot_weight_start, raw"C:\Users\mussi\Documents\Manhattan\Leuven\MCQMC\Plots\is-mp-smmala_ad\Rossler-Par\MC_tuning_eps.png")

for eps in step_size
    fname = "C:\\Users\\mussi\\Documents\\Manhattan\\Leuven\\MCQMC\\Plots\\is-mp-smmala_ad\\Rossler-Par\\weights_start_eps_$(replace(string(eps), "." => "_")).png"
    isfile(fname) && rm(fname)
end

############################################################## Run IS-MP-smMALA Rossler System ########################################################################

println("\n Running IS-MP-smMALA for Rossler System Parameter Estimation... \n")
mcqmc_time = @elapsed out = IS_MP_sMALA(rossler!, u0, obs_noisy, size(obs_noisy, 1), size(obs_noisy, 2), sigma_eta,
                                                tspan, dt, priors, init_par, seq=wcud, N_prop=N_prop, N_iter=N_iter, step_size=0.60)

println("Execution time: $(mcqmc_time) sec")
println("Chain stops after $(out.length) of $(N_iter).")
par_est = vec(mean(exp.(out.chain)[out.length÷2:end, :], dims=1))

##-------------------------------------------------------------------------------------------------------------#
# Weights evolution plots
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
savefig(p_weights_evolution, raw"C:\Users\mussi\Documents\Manhattan\Leuven\MCQMC\Plots\is-mp-smmala_ad\Rossler-Par\MC_weights.png")
#-------------------------------------------------------------------------------------------------------------# 
#  Trace Plots for parameters
chain = exp.(out.chain)
iters = 1:size(chain, 1)

p1 = plot(iters, chain[:, 1], label="", title="Trace of σ", xlabel="Iteration", ylabel="Value")
hline!(p1, [par_true[1]], linestyle=:dash, color=:red)
p2 = plot(iters, chain[:, 2], label="", title="Trace of ρ", xlabel="Iteration", ylabel="Value")
hline!(p2, [par_true[2]], linestyle=:dash, color=:red)
p3 = plot(iters, chain[:, 3], label="", title="Trace of β", xlabel="Iteration", ylabel="Value")
hline!(p3, [par_true[3]], linestyle=:dash, color=:red)
p = plot(p1, p2, p3, layout=(3,1), size=(900,800), legend=false)
savefig(p, raw"C:\Users\mussi\Documents\Manhattan\Leuven\MCQMC\Plots\is-mp-smmala_ad\Rossler-Par\MC_TracePlots.png")

#-------------------------------------------------------------------------------------------------------------# 
# #  Gradient Trace Plots for parameters
# chain = (out.grad)
# iters = 1:size(chain, 1)

# p1 = plot(iters, chain[:, 1], label="", title="Gradient Trace of σ", xlabel="Iteration", ylabel="Value")
# hline!(p1, [0], linestyle=:dash, color=:red)
# p2 = plot(iters, chain[:, 2], label="", title="Gradient Trace of ρ", xlabel="Iteration", ylabel="Value")
# hline!(p2, [0], linestyle=:dash, color=:red)
# p3 = plot(iters, chain[:, 3], label="", title="Gradient Trace of β", xlabel="Iteration", ylabel="Value")
# hline!(p3, [0], linestyle=:dash, color=:red)
# p = plot(p1, p2, p3, layout=(3,1), size=(900,800), legend=false)
# savefig(p, raw"C:\Users\mussi\Documents\Manhattan\Leuven\MCQMC\Plots\is-mp-smmala_ad\Rossler-Par\MC_GradientTracePlots.png")

####--------------------------------------------------------------------------------------------####
# Reconstruct trajectories with estimated parameters
lor_sys_new = ODEProblem(rossler!, u0, tspan, tuple(par_est...))
sol_new = solve(lor_sys_new, Rodas5P(), saveat = dt)

p_overlay = plot(sol[1, :], sol[2, :], sol[3, :], linewidth = 1.5, color = "blue", label = "True orbit",
                 xlabel = "x", ylabel = "y", zlabel = "z", title = "Rossler Attractor (Overlayed)", grid = true, gridalpha = 0.3, fontsize = 8)
scatter!(p_overlay, obs_noisy[1, :], obs_noisy[2, :], obs_noisy[3, :], markersize = 2, color = "red", alpha = 0.3, label = "Observations")
plot!(p_overlay, sol_new[1, :], sol_new[2, :], sol_new[3, :], linewidth = 1.5, color = "green", label = "Reconstructed orbit")

savefig(p_overlay, raw"C:\Users\mussi\Documents\Manhattan\Leuven\MCQMC\Plots\is-mp-smmala_ad\Rossler-Par\MC_Rossler_Orbits.png")

####--------------------------------------------------------------------------------------------####
# Time series plot 
p1 = scatter(sol.t, obs_noisy[1,:], markersize=3, alpha=0.5, label="x Observed", color=:lightblue,
             grid=true, gridalpha=0.3, ylabel="x", legend=:outertop)
plot!(p1, sol.t, sol_new[1,:], lw=2, label="x Reconstructed", color=:blue)

p2 = scatter(sol.t, obs_noisy[2,:], markersize=3, alpha=0.5, label="y Observed", color=:lightcoral,
             grid=true, gridalpha=0.3, ylabel="y", legend=:outertop)
plot!(p2, sol.t, sol_new[2,:], lw=2, label="y Reconstructed", color=:red)

p3 = scatter(sol.t, obs_noisy[3,:], markersize=3, alpha=0.5, label="z Observed", color=:lightgreen,
             grid=true, gridalpha=0.3, ylabel="z", xlabel="Time", legend=:outertop)
plot!(p3, sol.t, sol_new[3,:], lw=2, label="z Reconstructed", color=:green)

Rossler_ts = plot(p1, p2, p3, layout=(3,1), size=(1000, 800), 
                plot_title="Rossler Dynamics (IS-MP-smMALA)", 
                left_margin=5mm, bottom_margin=5mm, top_margin=5mm, legendcolumns=2,
                legendborder=false, legend_foreground_color=:transparent, legend=:outertop)

savefig(Rossler_ts, raw"C:\Users\mussi\Documents\Manhattan\Leuven\MCQMC\Plots\is-mp-smmala_ad\Rossler-Par\MC_Rossler_TimeSeries.png")

####--------------------------------------------------------------------------------------------####
# SSE function for parameters
function f_sse(p_test)
    sse = 0.0
    resid = zeros(size(obs_noisy, 1))
    lor_end = ODEProblem(rossler!, u0, tspan, p_test)
    sol_end = Array(solve(lor_end, Rodas5P(); saveat=dt, reltol=1e-6, abstol=1e-8, maxiters=100000, save_everystep=false))
    
    F = cholesky(Symmetric(sigma_eta))
    for t in 1:size(obs_noisy, 2)
        @views resid .= obs_noisy[:, t] .- sol_end[:, t]
        y = F.U \ resid
        sse += dot(y, y)
    end
    return sse
end

f_sse(a)
f_sse(par_est)
f_sse(par_true)

##-------------------------------------------- Marginal Likelihood for Rossler Parameters ---------------------------------------------------#

n_points = 2000

# Parameter ranges for σ, ρ, β
param_ranges = [
    range(0.0, 15.0, length=n_points),   # σ
    range(20.0, 35.0, length=n_points),  # ρ 
    range(0, 5.0, length=n_points)     # β
]

loglik_matrix = Array{Float64}(undef, n_points, 3)

theta_fixed = log.(collect(par_true))  # Start with true parameters in log space

# Compute profile likelihood for each parameter
for param_idx in 1:3
    for (i, param_val) in enumerate(param_ranges[param_idx])
        theta_test = copy(theta_fixed)
        theta_test[param_idx] = log(param_val)  
        sim = simulate_system(rossler!, u0, exp.(theta_test), tspan, dt)
        loglik_matrix[i, param_idx] = loglik_gaussian(obs_noisy, sim, CholSave(sigma_eta), size(obs_noisy, 1), size(obs_noisy, 2))
    end
end

p1 = plot(param_ranges[1], loglik_matrix[:, 1], 
    linewidth=2, xlabel= "a", ylabel="Log-Likelihood", 
    title="a", grid=true, gridalpha=0.3, label="")
vline!(p1, [par_true[1]], linestyle=:dash, color=:red, label="True", linewidth=2)
vline!(p1, [par_est[1]], linestyle=:dashdot, color=:blue, label="Estimated", linewidth=2)

p2 = plot(param_ranges[2], loglik_matrix[:, 2], 
    linewidth=2, xlabel= "b", ylabel="Log-Likelihood", 
    title="b", grid=true, gridalpha=0.3, label="")
vline!(p2, [par_true[2]], linestyle=:dash, color=:red, label="True", linewidth=2)
vline!(p2, [par_est[2]], linestyle=:dashdot, color=:blue, label="Estimated", linewidth=2)

p3 = plot(param_ranges[3], loglik_matrix[:, 3], 
    linewidth=2, xlabel= "c", ylabel="Log-Likelihood", 
    title="c", grid=true, gridalpha=0.3, label="")
vline!(p3, [par_true[3]], linestyle=:dash, color=:red, label="True", linewidth=2)
vline!(p3, [par_est[3]], linestyle=:dashdot, color=:blue, label="Estimated", linewidth=2)

p_profiles = plot(p1, p2, p3, layout=(1,3), size=(1500, 600), plot_title="Marginal Log-Likelihoods - Rossler Parameters", 
                  left_margin=6mm, bottom_margin=5mm, legend=:outertop, legendcolumns=2, legendfontsize=11, 
                  legendborder=false, legend_foreground_color=:transparent)
display(p_profiles)

savefig(p_profiles, raw"C:\Users\mussi\Documents\Manhattan\Leuven\MCQMC\Plots\is-mp-smmala_ad\Rossler-Par\MC_Rossler_Marginal_Likelihoods.png")

##-------------------------------------------------- Marginal Posterior for Rossler Parameters --------------------------------------------------------#

logpost_matrix = Array{Float64}(undef, n_points, 3)
for param_idx in 1:3
    for (i, param_val) in enumerate(param_ranges[param_idx])
        theta_test = copy(theta_fixed)
        theta_test[param_idx] = log(param_val)  
        par_test = exp.(theta_test)
        
        sim = simulate_system(rossler!, u0, par_test, tspan, dt)
        loglik = loglik_gaussian(obs_noisy, sim, CholSave(sigma_eta), size(obs_noisy, 1), size(obs_noisy, 2))
        logprior = logpdf(priors[1], par_test[1]) + logpdf(priors[2], par_test[2]) + logpdf(priors[3], par_test[3])
        jacobian_adjustment = sum(theta_test)  
        
        logpost_matrix[i, param_idx] = loglik + logprior + jacobian_adjustment
    end
end

pp1 = plot(param_ranges[1], logpost_matrix[:, 1], 
    linewidth=2, xlabel="a", ylabel="Log-Posterior", 
    title="a", grid=true, gridalpha=0.3, label="")
vline!(pp1, [par_true[1]], linestyle=:dash, color=:red, label="True", linewidth=2)
vline!(pp1, [par_est[1]], linestyle=:dashdot, color=:blue, label="Estimated", linewidth=2)

pp2 = plot(param_ranges[2], logpost_matrix[:, 2], 
    linewidth=2, xlabel="b", ylabel="Log-Posterior", 
    title="b", grid=true, gridalpha=0.3, label="")
vline!(pp2, [par_true[2]], linestyle=:dash, color=:red, label="True", linewidth=2)
vline!(pp2, [par_est[2]], linestyle=:dashdot, color=:blue, label="Estimated", linewidth=2)

pp3 = plot(param_ranges[3], logpost_matrix[:, 3], 
    linewidth=2, xlabel="c", ylabel="Log-Posterior", 
    title="c", grid=true, gridalpha=0.3, label="")
vline!(pp3, [par_true[3]], linestyle=:dash, color=:red, label="True", linewidth=2)
vline!(pp3, [par_est[3]], linestyle=:dashdot, color=:blue, label="Estimated", linewidth=2)

p_posteriors = plot(pp1, pp2, pp3, layout=(1,3), size=(1500, 600), 
    plot_title="Marginal Log-Posterior - Rossler Parameters", left_margin=6mm, bottom_margin=5mm,
    legend=:outertop, legendcolumns=2, legendfontsize=11, legendborder=false, legend_foreground_color=:transparent)
display(p_posteriors)

savefig(p_posteriors, raw"C:\Users\mussi\Documents\Manhattan\Leuven\MCQMC\Plots\is-mp-smmala_ad\Rossler-Par\MC_Rossler_Marginal_Log-Posteriors.png")


