include("C:/Users/mussi/Documents/Manhattan/Leuven/MCQMC/Data_Libraries/1.Libraries_setup.jl")
include("C:/Users/mussi/Documents/Manhattan/Leuven/MCQMC/Data_Libraries/0.mcmc_diagnostic.jl")
using ForwardDiff
using SciMLSensitivity

#####################################################################################################################
########################################         sm-MALA MCMC        ################################################
##################################  Automatic differentiation Black Box #############################################
################################     Lotka-Volterra 2D (alpha, and gamma)    ########################################
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

# Riemann Metric Tensor: Fisher Information + Prior Hessian
function Posterior_Tensor_Metric( f!, u0, theta::AbstractVector{<:Real}, tspan, dt, d::Integer, K::Integer, save_chol_cov::Chol_Save, priors; λ::Float64 = 1e-6)

    D = length(theta)
    chol_cov = save_chol_cov.chol_cov

    function sim_vec(theta_local::AbstractVector)
        return vec(simulate_system(f!, u0, exp.(theta_local), tspan, dt))
    end

    cfg = ForwardDiff.JacobianConfig(sim_vec, theta, ForwardDiff.Chunk{2}())
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

# smMALA
function rmala_lv(f!::Function, u0, obs, cov_mat, tspan, dt, priors, logprior_par::Function, init_par; N_iter::Integer=10_000, step_size::Float64)

    D = length(init_par); d, K = size(obs)
    save = CholSave(cov_mat)

    # Log-posterior 
    f_logpost = build_log_posterior(f!, u0, obs, save, tspan, dt, logprior_par, d, K)

    # Inital Conditions
    theta_cur = collect(init_par)
    logpost_cur = f_logpost(theta_cur)
    cfg_1 = ForwardDiff.GradientConfig(f_logpost, theta_cur, ForwardDiff.Chunk{D}())

    # Riemann Metric 
    # cfg_2 = ForwardDiff.JacobianConfig(sim_vec, theta_cur, ForwardDiff.Chunk{4}())
    metric = Posterior_Tensor_Metric(f!, u0, theta_cur, tspan, dt, d, K, save, priors)
    G_cur    = metric.FisherInfo; InvG_cur = metric.InvFisherInfo; L_cur = metric.L
  
    # MALA constants 
    s = step_size
    alpha = (s^2)/2

    # Storage
    chain_log = Array{Float64}(undef, N_iter, D)
    chain_par = similar(chain_log)
    logpost_vec = Vector{Float64}(undef, N_iter)
    grad = zeros(D)
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
        grad!(grad, f_logpost, theta_cur, cfg_1)
        mu_mala_cur = theta_cur .+ alpha .* (InvG_cur * grad)
        grad_rec[t, :] .= grad
        theta_prop = mu_mala_cur .+ s .* (L_cur * randn(D))
        # Evaluate proposed state's log-posterior
        logpost_prop = f_logpost(theta_prop)

        # Compute proposal metric
        metric_p = Posterior_Tensor_Metric(f!, u0, theta_prop, tspan, dt, d, K, save, priors)
        G_prop = metric_p.FisherInfo; InvG_prop = metric_p.InvFisherInfo; L_prop = metric_p.L
        grad!(grad, f_logpost, theta_prop, cfg_1)
        mu_mala_prop = theta_prop .+ alpha .* (InvG_prop * grad)

        # Accept/Rejection Rule
        # Remember the proposal is non-symmetric: 𝑞 ( 𝜃 ′ ∣ 𝜃 𝑡 ) ≠ 𝑞 ( 𝜃 𝑡 ∣ 𝜃 ′ ) 
        # logu< logπ(θ′)−logπ(θt​)​​+ logq(θt​∣θ′)−logq(θ′∣θt​)​​

        if log(rand()) < ((logpost_prop - logpost_cur) + (log_q(theta_cur,  mu_mala_prop, G_prop) - log_q(theta_prop, mu_mala_cur,  G_cur)))

        theta_cur = theta_prop
        logpost_cur = logpost_prop
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
    alpha, gamma = p
    x, y = u
    du[1] = alpha*x - 0.1*x*y
    du[2] = -1.0*y + gamma*x*y
end

Theta_true = (1.5, 0.075); tspan = (0.0, 8.0); dt = 0.02; u0 = [5.0, 5.0]; 
# Theta_true = (1.5, 0.1, 0.075, 1.0); tspan = (0.0, 1.0); dt = 0.05; u0 = [1.0, 1.0]; 

lok_volt = ODEProblem(lotka_volterra!, u0, tspan, Theta_true);
sol = solve(lok_volt, Tsit5(), saveat=dt);   

# Noisy Data 
# sigma_eta = .01 * I(2)       
sigma_eta = 1 * I(2)          
obs_noisy = Array(sol) .+ rand(MvNormal(zeros(2), sigma_eta), size(sol, 2)); 

#----------------------------------------------------------------------------------------------------------------#

p1 = plot(sol[1,:], sol[2,:], linewidth = 1.5, xlabel = "x", ylabel = "y", legend = false, grid = true, gridalpha = 0.3, title = "True Orbit", fontsize =8)
p2 = plot(obs_noisy[1,:], obs_noisy[2,:], linewidth = 1.5, xlabel = "x", ylabel = "y", legend = false, color = "red", grid = true, gridalpha = 0.3, title = "Observed Orbit", fontsize =8)
lv_orbits = plot( p1, p2, layout = (1, 2), size = (700, 350), bottom_margin=5mm, top_margin=5mm)

lv_orbits_2 = plot(sol.t, sol[1,:], lw=2, label="x true", grid=:true, gridalpha=0.3, legendfontsize=9, 
      legend=:outertop, legendcolumns=2, legendborder=false,legend_foreground_color=:transparent,
      title = "2D Lotka-Volterra Dynamics", fontsize =10) 
plot!(sol.t, sol[2,:], lw=2, label = "y true")
scatter!(sol.t, obs_noisy[1,:], ms=2.2, label= "x noisy")
scatter!(sol.t, obs_noisy[2,:], ms=2.2, label= "y noisy")
xlabel!("Time") 

#----------------------------------------------------------------------------------------------------------------#


priors = (
          Gamma(2, 1),   # alpha
          Gamma(1, 0.2)  # gamma
)

logprior_par = p -> (logpdf(priors[1], p[1]) + logpdf(priors[2], p[2]))
N_iter = 4000; 
a = (3.,  1.); init_par = log.(a)

println("Running sm-MALA MCMC...")
Random.seed!(1234);
mcqmc_time = @elapsed out = rmala_lv(lotka_volterra!, u0, obs_noisy, sigma_eta, tspan, dt, priors, logprior_par, init_par, N_iter=N_iter, step_size= 0.15);
println("Execution time: $(mcqmc_time) sec")


par_est =  mean(out.chain_par[N_iter÷2:end, :], dims=1)
out.grad_record
out.acc_rate
# ------------------------------------------------------------------------------------------------------------------#

for i in 1:2
    println("ESS $(i): Parameters = ", ess_ips(out.chain_par[:, i]))
end

#------------------------------------------------------------------------------------------------------------------# 

chain = out.chain_par
iters = 1:size(chain, 1)

p1 = plot(iters, chain[:, 1], label="α", title="Trace of α", xlabel="Iteration", ylabel="Value")
hline!(p1, [Theta_true[1]], linestyle=:dash, color=:red)
p2 = plot(iters, chain[:, 2], label="γ", title="Trace of γ", xlabel="Iteration", ylabel="Value")
hline!(p2, [Theta_true[2]], linestyle=:dash, color=:red)
plot1 = plot(p1, p2, layout=(2,1), size=(900,800), legend=false, plot_title="Trace Plots (sm-MALA)", left_margin=5mm)

savefig(plot1, raw"C:\Users\mussi\Documents\Manhattan\Leuven\MCQMC\Plots\sm-mala\Trace_Plots.png")


####------------------------------------------------------------------------------------------------

chain= out.grad_record
iters = 1:size(chain, 1)

p1 = plot(iters, chain[:, 1], label="α", title="Trace of α", xlabel="Iteration", ylabel="Value")
hline!(p1, [Theta_true[1]], linestyle=:dash, color=:red)
p2 = plot(iters, chain[:, 2], label="γ", title="Trace of γ", xlabel="Iteration", ylabel="Value")
hline!(p2, [Theta_true[2]], linestyle=:dash, color=:red)
plot2 = plot(p1, p2, layout=(2,1), size=(900,800), legend=false, plot_title="Gradient Trace Plots (sm-MALA)", left_margin=3mm)

savefig(plot2, raw"C:\Users\mussi\Documents\Manhattan\Leuven\MCQMC\Plots\sm-mala\Gradient_Trace_Plots.png")

#--------------------------------------------------------------------------------------------####


lok_volt_new = ODEProblem(lotka_volterra!, u0, tspan, tuple(par_est...))
sol_new = solve(lok_volt_new, Tsit5(), saveat = dt)

mid_iter = max(1, div(size(out.chain_theta, 1), 2))
lok_volt_t1 = ODEProblem(lotka_volterra!, u0, tspan, tuple(exp.(out.chain_theta[500,:])...))
sol_new_t1 = solve(lok_volt_t1, Tsit5(), saveat = dt)

lok_volt_initial = ODEProblem(lotka_volterra!, u0, tspan, tuple(a...))
sol_initial = solve(lok_volt_initial, Tsit5(), saveat = dt)

p_overlay = plot(sol[1, :], sol[2, :], linewidth = 1.5, color = "blue", label = "True orbit", xlabel = "x", ylabel = "y", title = "2D Lotka–Volterra Orbits (sm-MALA)", grid = true, gridalpha = 0.3, fontsize = 8, aspect_ratio = :equal)
plot!(p_overlay, sol_initial[1, :], sol_initial[2, :], linewidth = 1.5, color = "black", label = "Initial orbit")
plot!(p_overlay, sol_new_t1[1, :], sol_new_t1[2, :], linewidth = 1, color = "black", linestyle = :dashdot, label = "Mid-chain orbit")
plot!(p_overlay, obs_noisy[1, :], obs_noisy[2, :], linewidth = 1.5, color = "red", label = "Observed orbit")
plot3 = plot!(p_overlay, sol_new[1, :], sol_new[2, :], linewidth = 1.5, color = "green", label = "Reconstructed orbit")

savefig(plot3, raw"C:\Users\mussi\Documents\Manhattan\Leuven\MCQMC\Plots\sm-mala\Overlay_Orbits.png")

####--------------------------------------------------------------------------------------------####

lv_orbits_sm = plot(sol.t, sol[1,:], lw=2, label="x true", grid=true, gridalpha=0.3, legendfontsize=9, 
                    legend=:outertop, legendcolumns=2, legendborder=false, legend_foreground_color=:transparent, 
                    title="2D Lotka-Volterra Dynamics (sm-MALA)", fontsize=10)
plot!(sol.t, sol[2,:], lw=2, label="y true")
scatter!(sol.t, obs_noisy[1,:], ms=2.2, label="x noisy", alpha=0.6)
scatter!(sol.t, obs_noisy[2,:], ms=2.2, label="y noisy", alpha=0.6)
plot!(sol.t, sol_new[1,:], lw=2, ls=:dash, label="x sm-MALA", color=:blue)
plot!(sol.t, sol_new[2,:], lw=2, ls=:dash, label="y sm-MALA", color=:red)
xlabel!("Time")
savefig(lv_orbits_sm, raw"C:\Users\mussi\Documents\Manhattan\Leuven\MCQMC\Plots\sm-mala\LV_Dynamics_sm-MALA.png")


####--------------------------------------------------------------------------------------------####

function f_sse(p)
    sse = 0.0; resid = zeros(size(obs_noisy, 1)); 
    lv_end = ODEProblem(lotka_volterra!, u0, tspan, p);
    sol = Array(solve(lv_end, Tsit5(); saveat=dt, reltol=1e-6, abstol=1e-8, maxiters=10000, save_everystep=false))

    
    F = cholesky(Symmetric(sigma_eta))
    for t in 1:size(obs_noisy, 2)
        @views resid .= obs_noisy[:, t] .- sol[:, t]
        y = F.U \ resid
        sse += dot(y, y)
    end
    return sse
end

f_sse(a)
f_sse(par_est)
f_sse(Theta_true)


##-------------------------------------------- Marginal Likelihood ---------------------------------------------------#

n_points = 2000

# Parameter ranges
param_ranges = [
    range(0.01, 3.5, length=n_points),  # alpha
    range(0.01, 2., length=n_points)   # gamma
]

loglik_matrix = Array{Float64}(undef, n_points, 2)

theta_fixed = log.(collect(Theta_true))  # Start with true parameters in log space

# Compute profile likelihood for each parameter
for param_idx in 1:2
    for (i, param_val) in enumerate(param_ranges[param_idx])
        theta_test = copy(theta_fixed)
        theta_test[param_idx] = log(param_val)  
        sim = simulate_system(lotka_volterra!, u0, exp.(theta_test), tspan, dt)
        loglik_matrix[i, param_idx] = loglik_gaussian(obs_noisy, sim, CholSave(sigma_eta), size(obs_noisy, 1), size(obs_noisy, 2))
    end
end

p1 = plot(param_ranges[1], loglik_matrix[:, 1], 
    linewidth=2, xlabel= "Sample Value", ylabel="Log-Likelihood", 
    title="α", grid=true, gridalpha=0.3, label="")
vline!(p1, [Theta_true[1]], linestyle=:dash, color=:red, label="", linewidth=2)

p2 = plot(param_ranges[2], loglik_matrix[:, 2], 
    linewidth=2, xlabel= "Sample Value", ylabel="Log-Likelihood", 
    title="γ", grid=true, gridalpha=0.3, label="")
vline!(p2, [Theta_true[2]], linestyle=:dash, color=:red, label="", linewidth=2)

p_profiles = plot(p1, p2, layout=(1,2), size=(1300, 800),  plot_title="Marginal Likelihoods", left_margin=6mm, bottom_margin=5mm)
display(p_profiles)

savefig(p_profiles, raw"C:\Users\mussi\Documents\Manhattan\Leuven\MCQMC\Plots\sm-mala\Marginal_Likelihoods.png")

##-------------------------------------------------- Marginal Posterior --------------------------------------------------------#

logpost_matrix = Array{Float64}(undef, n_points, 2)

for param_idx in 1:2
    for (i, param_val) in enumerate(param_ranges[param_idx])
        theta_test = copy(theta_fixed)
        theta_test[param_idx] = log(param_val)  
        par_test = exp.(theta_test)
        
        sim = simulate_system(lotka_volterra!, u0, par_test, tspan, dt)
        loglik = loglik_gaussian(obs_noisy, sim, CholSave(sigma_eta), size(obs_noisy, 1), size(obs_noisy, 2))
        logprior = logprior_par(par_test)
        jacobian_adjustment = sum(theta_test)  
        
        logpost_matrix[i, param_idx] = loglik + logprior + jacobian_adjustment
    end
end

    pp1 = plot(param_ranges[1], logpost_matrix[:, 1], 
        linewidth=2, xlabel= "Sample Value", ylabel="Density", 
        title="α", grid=true, gridalpha=0.3, label="")
    vline!(pp1, [Theta_true[1]], linestyle=:dash, color=:red, label="", linewidth=2)

    pp2 = plot(param_ranges[2], logpost_matrix[:, 2], 
        linewidth=2, xlabel= "Sample Value", ylabel="Density", 
        title="γ", grid=true, gridalpha=0.3, label="")
    vline!(pp2, [Theta_true[2]], linestyle=:dash, color=:red, label="", linewidth=2)

    p_posteriors = plot(pp1, pp2, layout=(1,2), size=(1300, 800), 
        plot_title="Marginal Posteriors", left_margin=6mm, bottom_margin=5mm)
    display(p_posteriors)

    savefig(p_posteriors, raw"C:\Users\mussi\Documents\Manhattan\Leuven\MCQMC\Plots\sm-mala\Marginal_Posteriors.png")