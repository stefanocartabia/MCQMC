include("C:/Users/mussi/Documents/Manhattan/Leuven/MCQMC/Data_Libraries/1.Libraries_setup.jl")
include("C:/Users/mussi/Documents/Manhattan/Leuven/MCQMC/Data_Libraries/0.mcmc_diagnostic.jl")
using ForwardDiff
using SciMLSensitivity
using Zygote
# using ReverseDiff

#####################################################################################################################
########################################         sm-MALA MCMC        ################################################
#############################################   Adjoint Method   ####################################################
#####################################################################################################################

# ODE Solver: Tsitouras 5/4 Runge–Kutta + EXPLICIT SENSITIVITY
function simulate_system(f!::Function, u0, par, tspan::Tuple, dt)
    prob = ODEForwardSensitivityProblem(f!, u0, tspan, par)
    sol = solve(prob, Tsit5(); saveat=dt, reltol=1e-6, abstol=1e-8, sensealg = ForwardSensitivity())
    x, dp = extract_local_sensitivities(sol)   

    n_params = length(dp)
    n_states, n_times = size(x)
    sens = Array{eltype(x)}(undef, n_states, n_times, n_params)
    for j in 1:n_params
        sens[:, :, j] = Array(dp[j])
    end
    
    return Array(x), sens, sol.t
end

# Struct to save Cholesky factorization
struct Chol_Save{T}
    chol_cov::Cholesky{T,Matrix{T}} 
    logdet :: T 
end

# Cholesky factorisation
function CholSave(cov::AbstractMatrix{<:Real})
    F = cholesky(Symmetric(Matrix(cov)))
    logdet = 2*sum(log, diag(F.U))   
    return Chol_Save(F, logdet)
end

# Log-posterior and gradient 
function build_log_posterior_and_grad(f!, u0, obs, save_chol_cov::Chol_Save,
                                      tspan, dt, logprior_par::Function,
                                      d::Integer, K::Integer)

    function log_lik_adj(par)
        prob = ODEProblem(f!, u0, tspan, par)
        sol = solve(prob, Tsit5(); saveat = dt, reltol = 1e-6, abstol = 1e-8, sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true)))

        sim = Array(sol)
        E   = obs .- sim
        Y   = save_chol_cov.chol_cov.L \ E
        sse = sum(abs2, Y)

        return -0.5 * (K * d * log(2π) + K * save_chol_cov.logdet + sse)
    end

    function logpost_and_grad(theta)
        par = exp.(theta)

        loglik = log_lik_adj(par)
        logprior = logprior_par(par)

        _, back = Zygote.pullback(log_lik_adj, par)
        grad_lik_p = back(1.0)[1]
        grad_prior_p = ForwardDiff.gradient(logprior_par, par)

        grad_theta = (grad_lik_p .+ grad_prior_p) .* par .+ ones(eltype(par), length(par))

        logpost = loglik + logprior + sum(theta)

        return logpost, grad_theta
    end

    return logpost_and_grad
end


# Riemannian Metric: Expected Fisher Information Matrix
function Tensor_Metric(f!, u0, theta::AbstractVector{<:Real}, tspan, dt, d::Integer, K::Integer, save_chol_cov::Chol_Save; λ::Float64 = 1e-1)
    D = length(theta)
    L = save_chol_cov.chol_cov.L
    par = collect(exp.(theta))
    
    x, sen, _ = simulate_system(f!, u0, par, tspan, dt) 

    # Build Jacobian J: (d*K) x D
    n_states, n_times = size(x)
    J = zeros(eltype(x), d*K, D)
    for j in 1:D
        Sj = sen[:, :, j]
        for t in 1:K
            r1 = (t-1)*d + 1; r2 = t*d
            J[r1:r2, j] = Sj[1:d, t]
        end
    end

    # Fisher information: G = Σ_t J_t' Σ^{-1} J_t
    G = zeros(eltype(J), D, D)
    for t in 1:K
        r1 = (t-1)*d + 1; r2 = t*d
        Jt = @view J[r1:r2, :]
        MJt = similar(Jt)
        mul!(MJt, L, Jt)
        mul!(G, transpose(MJt), MJt, 1.0, 1.0)
    end

    G = Symmetric(G + λ * I)
    chol_mat = cholesky(G)
    L_inv = chol_mat.L \ I
    InvFisherInfo = chol_mat \ I

    return (FisherInfo = G, InvFisherInfo = InvFisherInfo, L = L_inv)
end

# smMALA 
function rmala_lv(f!::Function, u0, obs, cov_mat, tspan, dt, logprior_par::Function, init_par; N_iter::Integer=10_000, step_size::Float64)

    D = length(init_par); d, K = size(obs)
    save = CholSave(cov_mat)
    # Log-posterior and gradient function using ADJOINT
    f_logpost_grad = build_log_posterior_and_grad(f!, u0, obs, save, tspan, dt, logprior_par, d, K)

    # Inital Conditions
    theta_cur = collect(init_par)
    logpost_cur, grad_cur = f_logpost_grad(theta_cur)

    # Riemann Metric 
    metric = Tensor_Metric(f!, u0, theta_cur, tspan, dt, d, K, save)
    G_cur    = metric.FisherInfo; InvG_cur = metric.InvFisherInfo; L_cur = metric.L
  
    # MALA constants 
    s = step_size
    alpha = (s^2)/2

    # Storage
    chain_log = Array{Float64}(undef, N_iter, D)
    chain_par = similar(chain_log)
    logpost_vec = Vector{Float64}(undef, N_iter)
    grad_rec = Array{Float64}(undef, N_iter, D)
    acc = 0

    function log_q(y::AbstractVector{<:Real}, mu::AbstractVector{<:Real}, G::AbstractMatrix{<:Real})
        Prec = (1/s^2) .* G
        R = cholesky(Symmetric(Prec)).U
        v = R * (y .- mu )
        return -0.5*(length(y)*log(2π) - 2sum(log, diag(R)) + dot(v, v))
    end

    # Stopping criterion
    stall_cnt   = 0; l_effective = 0
    max_stall  = ceil(.025  * N_iter); tol_move   = 1e-4

    for t in 1:N_iter

        # Current Theta: Compute gradient and new proposal
        mu_mala_cur = theta_cur .+ alpha .* (InvG_cur * grad_cur)
        grad_rec[t, :] .= grad_cur
        theta_prop = mu_mala_cur .+ s .* (L_cur * randn(D))
        
        # Evaluate proposed state's log-posterior and gradient
        logpost_prop, grad_prop = f_logpost_grad(theta_prop)

        # Compute proposal metric
        metric_p = Tensor_Metric(f!, u0, theta_prop, tspan, dt, d, K, save)
        G_prop = metric_p.FisherInfo; InvG_prop = metric_p.InvFisherInfo; L_prop = metric_p.L
        mu_mala_prop = theta_prop .+ alpha .* (InvG_prop * grad_prop)

        # Accept/Rejection Rule
        # Remember the proposal is non-symmetric: 𝑞 ( 𝜃 ′ ∣ 𝜃 𝑡 ) ≠ 𝑞 ( 𝜃 𝑡 ∣ 𝜃 ′ ) 
        # logu< logπ(θ′)−logπ(θt​)​​+ logq(θt​∣θ′)−logq(θ′∣θt​)​​

        if log(rand()) < ((logpost_prop - logpost_cur) + (log_q(theta_cur,  mu_mala_prop, G_prop) - log_q(theta_prop, mu_mala_cur,  G_cur)))

            theta_cur = theta_prop
            logpost_cur = logpost_prop
            grad_cur = grad_prop
            G_cur, InvG_cur, L_cur = G_prop, InvG_prop, L_prop

            acc += 1
        end

        chain_log[t, :] .= theta_cur
        @. chain_par[t, :] = exp(chain_log[t, :])
        logpost_vec[t] = logpost_cur

        # early stopping
        if t > 1 && norm(chain_par[t, :] .- chain_par[t-1, :]) ≤ tol_move
            stall_cnt += 1
        else
            stall_cnt = 0
        end
        l_effective = t
        if stall_cnt ≥ max_stall
            break
        end
    end

    return (
            chain_theta = chain_log[1:l_effective, :],
            chain_par   = chain_par[1:l_effective, :],
            logpost     = logpost_vec[1:l_effective],
            acc_rate    = acc / l_effective,
            last_metric = (FisherInfo = G_cur, InvFisherInfo = InvG_cur, L = L_cur),
            grad_record = grad_rec[1:l_effective, :]
    )
end


######################################################################################################################################################
######################################################################################################################################################

function lotka_volterra!(du, u, p, t)
    alpha, beta, delta, gamma = p
    x, y = u
    du[1] = alpha*x - beta*x*y
    du[2] = -gamma*y + delta*x*y
end

Theta_true = (1.5, 0.1, 0.075, 1.0); tspan = (0.0, 10.0); dt = 0.02; u0 = [5.0, 5.0]; 
# Theta_true = (1.5, 0.1, 0.075, 1.0); tspan = (0.0, 1.0); dt = 0.05; u0 = [1.0, 1.0]; 
lok_volt = ODEProblem(lotka_volterra!, u0, tspan, Theta_true);
sol = solve(lok_volt, Tsit5(), saveat=dt);   

# Noisy Data 
sigma_eta = .01 * I(2)              
obs_noisy = Array(sol) .+ rand(MvNormal(zeros(2), sigma_eta), size(sol, 2)); 

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
          Gamma(2, 1),   # alpha
          Gamma(2, 1),   # beta 
          Gamma(2, 1),   # delta
          Gamma(2, 1)    # gamma
)

logprior_par = p -> (logpdf(priors[1], p[1]) + logpdf(priors[2], p[2]) + logpdf(priors[3], p[3]) + logpdf(priors[4], p[4]))

N_iter = 2_000; 
a = [ 1.1,  1.1,  1.1,  1.1]
init_par = log.(a)

println("Running sm-MALA MCMC with Adjoint Sensitivity...")
mcqmc_time = @elapsed out = rmala_lv(lotka_volterra!, u0, obs_noisy, sigma_eta, tspan, dt, logprior_par, init_par, N_iter=N_iter, step_size= 0.03);
println("Execution time: $(mcqmc_time) sec")


out.chain_par
out.grad_record
out.acc_rate
# ------------------------------------------------------------------------------------------------------------------#

for i in 1:4
    println("ESS $(i): Parameters = ", ess_ips(out.chain_par[:, i]))
end

#------------------------------------------------------------------------------------------------------------------# 

chain = out.chain_par
iters = 1:size(chain, 1)

# Create trace plots
gr()
p1 = plot(iters, chain[:, 1], label="α", title="Trace of α", xlabel="Iteration", ylabel="Value")
hline!(p1, [Theta_true[1]], linestyle=:dash, color=:red)
p2 = plot(iters, chain[:, 2], label="β", title="Trace of β", xlabel="Iteration", ylabel="Value")
hline!(p2, [Theta_true[2]], linestyle=:dash, color=:red)
p3 = plot(iters, chain[:, 3], label="δ", title="Trace of δ", xlabel="Iteration", ylabel="Value")
hline!(p3, [Theta_true[3]], linestyle=:dash, color=:red)
p4 = plot(iters, chain[:, 4], label="γ", title="Trace of γ", xlabel="Iteration", ylabel="Value")
hline!(p4, [Theta_true[4]], linestyle=:dash, color=:red)
plot(p1, p2, p3, p4, layout=(4,1), size=(900,800), legend=false)

####--------------------------------------------------------------------------------------------####


chain= out.grad_record
iters = 1:size(chain, 1)

# Create trace plots
gr()
p1 = plot(iters, chain[:, 1], label="α", title="Trace of α", xlabel="Iteration", ylabel="Value")
hline!(p1, [0], linestyle=:dash, color=:red)
p2 = plot(iters, chain[:, 2], label="β", title="Trace of β", xlabel="Iteration", ylabel="Value")
hline!(p2, [0], linestyle=:dash, color=:red)
p3 = plot(iters, chain[:, 3], label="δ", title="Trace of δ", xlabel="Iteration", ylabel="Value")
hline!(p3, [0], linestyle=:dash, color=:red)
p4 = plot(iters, chain[:, 4], label="γ", title="Trace of γ", xlabel="Iteration", ylabel="Value")
hline!(p4, [0], linestyle=:dash, color=:red)
# Combine in a 4×1 layout
plot(p1, p2, p3, p4, layout=(4,1), size=(900,800), legend=false, suptitle = "Gradient Trace Plots")

#--------------------------------------------------------------------------------------------####


lok_volt_new = ODEProblem(lotka_volterra!, u0, tspan, tuple(exp.(out.chain_par[end,:])...))
sol_new = solve(lok_volt_new, Tsit5(), saveat = dt)

mid_iter = max(1, div(size(out.chain_par, 1), 2))
lok_volt_t1 = ODEProblem(lotka_volterra!, u0, tspan, tuple(exp.(out.chain_par[mid_iter,:])...))
sol_new_t1 = solve(lok_volt_t1, Tsit5(), saveat = dt)

lok_volt_initial = ODEProblem(lotka_volterra!, u0, tspan, tuple(a...))
sol_initial = solve(lok_volt_initial, Tsit5(), saveat = dt)

p_overlay = plot(sol[1, :], sol[2, :], linewidth = 1.5, color = "blue", label = "True orbit", xlabel = "x", ylabel = "y", title = "Lotka–Volterra Orbits (Overlayed)", grid = true, gridalpha = 0.3, fontsize = 8)
plot!(p_overlay, sol_initial[1, :], sol_initial[2, :], linewidth = 1.5, color = "black", label = "Initial orbit")
plot!(p_overlay, sol_new_t1[1, :], sol_new_t1[2, :], linewidth = 1, color = "black", linestyle = :dashdot, label = "Mid-chain orbit")
plot!(p_overlay, obs_noisy[1, :], obs_noisy[2, :], linewidth = 1.5, color = "red", label = "Observed orbit")
plot!(p_overlay, sol_new[1, :], sol_new[2, :], linewidth = 1.5, color = "green", label = "Reconstructed orbit")

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

####--------------------------------------------------------------------------------------------####

function f_sse(p)
    sse = 0.0; resid = zeros(size(obs_noisy, 1)); 
    lv_end = ODEProblem(lotka_volterra!, u0, tspan, p);
    sol = solve(lv_end, Tsit5(), saveat=dt);
    F = cholesky(Symmetric(sigma_eta))
    for t in 1:size(obs_noisy, 2)
        @views resid .= obs_noisy[:, t] .- sol[:, t]
        y = F.U \ resid
        sse += dot(y, y)
    end
    return sse
end

f_sse(a)
f_sse(exp.(out.chain_par[end,:]))
f_sse(Theta_true)