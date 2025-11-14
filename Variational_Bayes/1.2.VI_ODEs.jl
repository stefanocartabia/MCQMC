include("C:/Users/mussi/Documents/Manhattan/Leuven/MCQMC/Data_Libraries/1.Libraries_setup.jl")

using Random
using Turing
using AdvancedVI
using FillArrays
using RDatasets
using Optimisers
using QuasiMonteCarlo
using DiffResults
using DynamicPPL  
using Zygote    
using ADTypes
using LogDensityProblems
using Distributions
using DataFrames
using StatsFuns
using Bijectors: Bijectors
using LogDensityProblemsAD
using ForwardDiff
Pkg.status("AdvancedVI")


function simulate_system(f!::Function, u0, par, tspan::Tuple, dt)
    prob = ODEProblem(f!, u0, tspan, par)
    sol = solve(prob, Tsit5(); saveat = dt, reltol = 1e-6, abstol = 1e-8, maxiters = 10000, save_everystep = false)
    return Array(sol)
end

struct Chol_Save{T}
    chol_cov::Cholesky{T,Matrix{T}}
    logdet::T
end

function CholSave(cov::AbstractMatrix{<:Real})
    F = cholesky(Symmetric(Matrix(cov)))
    logdet = 2 * sum(log, diag(F.U))
    return Chol_Save(F, logdet)
end

function loglik_gaussian(obs::AbstractMatrix{<:Real}, sim::AbstractMatrix, chol_save::Chol_Save, d::Integer, K::Integer)
    @assert size(obs) == size(sim)
    E = obs .- sim
    Y = chol_save.chol_cov.L \ E
    sse = sum(abs2, Y)
    return -0.5 * (K*d*log(2π) + K*chol_save.logdet + sse)
end

function make_logpost_par(f!, u0, obs, save_chol_cov::Chol_Save, tspan, dt, logprior_par)
    function logpost(par::AbstractVector)
        sim = simulate_system(f!, u0, par, tspan, dt)
        lp = loglik_gaussian(obs, sim, save_chol_cov, size(obs,1), size(obs,2))
        return lp + logprior_par(par)   
    end
    return logpost
end

struct ODELogPosterior{F,B}
    logpost_par::F         
    b::B                    # bijector: unconstrained ℝ^m -> positive constrained space
    dim::Int
end

function LogDensityProblems.logdensity(m::ODELogPosterior, θ::AbstractVector)
    # From unconstrained θ to constrained parameter space
    par = m.b(θ)
    # Log-posterior in original parameter space
    lp_param = m.logpost(par)

    return lp_param + Bijectors.logabsdetjac(m.b, θ)
end

LogDensityProblems.dimension(m::ODELogPosterior) = m.dim
LogDensityProblems.capabilities(::Type{<:ODELogPosterior}) = LogDensityProblems.LogDensityOrder{0}()


# --------------------- Noisy Data ----------------------- #

function lotka_volterra!(du, u, p, t)
    alpha, beta, delta, gamma = p
    x, y = u
    du[1] = alpha*x - beta*x*y
    du[2] = -gamma*y + delta*x*y
end

Theta_true = (1.5, 0.1, 0.075, 1.0); tspan = (0.0, 10.0); dt = 0.02; u0 = [5.0, 5.0]; 
lok_volt = ODEProblem(lotka_volterra!, u0, tspan, Theta_true);
sol = solve(lok_volt, Tsit5(), saveat=dt);   
sigma_eta = 1 * I(2)              
obs_noisy = Array(sol) .+ rand(MvNormal(zeros(2), sigma_eta), size(sol, 2)); 


priors = (
    Gamma(2, 1),   # alpha
    Gamma(2, 1),   # beta 
    Gamma(2, 1),   # delta
    Gamma(2, 1)    # gamma
)

logprior_par = p -> sum(logpdf.(priors, p))

d_par = 4 
b_forward = Bijectors.elementwise(exp)  

save_chol_cov = CholSave(sigma_eta)  
logpost_par_fn = make_logpost_par(lotka_volterra!, u0, obs_noisy, save_chol_cov, tspan, dt, logprior_par)
model = ODELogPosterior(logpost_par_fn, b_forward, d_par)


init_par = [1.0, 1.0, 1.0, 1.0]  
μ0 = log.(init_par)                 
q0 = MeanFieldGaussian(μ0, Diagonal(fill(0.1^2, d_par)))

# ----------------- VI Optimisation --------------------------------------- # 
n_max_iter = 10000
n_montecarlo = 10
objective = RepGradELBO(n_montecarlo)

q_avg, q_last, stats, state = AdvancedVI.optimize( model, objective, q0, n_max_iter; show_progress = true, adtype = AutoForwardDiff(), optimizer = Optimisers.Adam(1e-3), operator = ClipScale() );
nothing

plot([i.elbo for i in stats], xlabel="Iterations", ylabel="ELBO", label="ELBO over iterations", title="ELBO Optimisation")


z_unconstrained = rand(q_last, 10^4)  
z_constrained = b_forward(z_unconstrained) 
avg = vec(mean(z_constrained; dims=2))

println("Estimated parameters (mean): ", avg)
println("True parameters: ", collect(Theta_true))

#-----------------------------------------------------------------------#

lok_volt_new = ODEProblem(lotka_volterra!, u0, tspan, tuple(avg...))
sol_new = solve(lok_volt_new, Tsit5(), saveat = dt)

lok_volt_initial = ODEProblem(lotka_volterra!, u0, tspan, tuple(init_par...))
sol_initial = solve(lok_volt_initial, Tsit5(), saveat = dt)

p_overlay = plot(sol[1, :], sol[2, :], linewidth = 1.5, color = "blue", label = "True orbit", xlabel = "x", ylabel = "y", title = "Lotka–Volterra Orbits (Overlayed)", grid = true, gridalpha = 0.3, fontsize = 8)
plot!(p_overlay, sol_initial[1, :], sol_initial[2, :], linewidth = 1.5, color = "black", label = "Initial orbit")
plot!(p_overlay, obs_noisy[1, :], obs_noisy[2, :], linewidth = 1.5, color = "red", label = "Observed orbit")
plot!(p_overlay, sol_new[1, :], sol_new[2, :], linewidth = 1.5, color = "green", label = "Reconstructed orbit")

####--------------------------------------------------------------------------------------------####

lv_orbits_sm = plot(sol.t, obs_noisy[1,:], lw=2, label="x true",
    grid=true, gridalpha=0.3, legendfontsize=9,
    legend=:outertop, legendcolumns=2, legendborder=false,
    legend_foreground_color=:transparent,
    title="Lotka-Volterra Dynamics (VI)", fontsize=10)
plot!(sol.t, obs_noisy[2,:], lw=2, label="y true")
plot!(sol.t, sol_new[1,:], lw=2, ls=:solid, label="x VI", color=:blue)
plot!(sol.t, sol_new[2,:], lw=2, ls=:solid, label="y VI", color=:red)
xlabel!("Time")


# ----------------- VI Optimisation 2 ------------------------- #

Theta_true_2 = (1.8, 0.5, 1.0, 2.5); tspan = (0.0, 8.0); dt = 0.02; u0 = [10.0, 5.0]; 
lok_volt = ODEProblem(lotka_volterra!, u0, tspan, Theta_true_2);
sol = solve(lok_volt, Tsit5(), saveat=dt);   
sigma_eta = 0.25 * I(2)              
obs_noisy = Array(sol) .+ rand(MvNormal(zeros(2), sigma_eta), size(sol, 2)); 


logprior_par = p -> sum(logpdf.(priors, p))
d_par = 4 
b_forward = Bijectors.elementwise(exp)  
save_chol_cov = CholSave(sigma_eta)  
logpost_par_fn = make_logpost_par(lotka_volterra!, u0, obs_noisy, save_chol_cov, tspan, dt, logprior_par)
model = ODELogPosterior(logpost_par_fn, b_forward, d_par)

init_par = [1.0, 1.0, 1.0, 1.0]  
μ0 = log.(init_par)                 
L_init = Diagonal(fill(0.01, d_par))         
q0 = FullRankGaussian(μ0, cholesky(Matrix(L_init)).L)

n_max_iter = 10000
n_montecarlo = 20
objective = RepGradELBO(n_montecarlo)

q_avg, q_last, stats, state = AdvancedVI.optimize( model, objective, q0, n_max_iter; show_progress = true, adtype = AutoForwardDiff(), operator = ClipScale() );
nothing

plot([i.elbo for i in stats], xlabel="Iterations", ylabel="ELBO", label="ELBO over iterations", title="ELBO Optimisation")


z_unconstrained = rand(q_last, 10^4)  
z_constrained = b_forward(z_unconstrained) 
avg = vec(mean(z_constrained; dims=2))

println("Estimated parameters (mean): ", avg)
println("True parameters: ", collect(Theta_true_2))


#-----------------------------------------------------------------------#

lok_volt_new = ODEProblem(lotka_volterra!, u0, tspan, tuple(avg...))
sol_new = solve(lok_volt_new, Tsit5(), saveat = dt)

lok_volt_initial = ODEProblem(lotka_volterra!, u0, tspan, tuple(init_par...))
sol_initial = solve(lok_volt_initial, Tsit5(), saveat = dt)

p_overlay = plot(sol[1, :], sol[2, :], linewidth = 1.5, color = "blue", label = "True orbit", xlabel = "x", ylabel = "y", title = "Lotka–Volterra Orbits (Overlayed)", grid = true, gridalpha = 0.3, fontsize = 8)
plot!(p_overlay, sol_initial[1, :], sol_initial[2, :], linewidth = 1.5, color = "black", label = "Initial orbit")
plot!(p_overlay, obs_noisy[1, :], obs_noisy[2, :], linewidth = 1.5, color = "red", label = "Observed orbit")
plot!(p_overlay, sol_new[1, :], sol_new[2, :], linewidth = 1.5, color = "green", label = "Reconstructed orbit")

####--------------------------------------------------------------------------------------------####

lv_orbits_sm = plot(sol.t, obs_noisy[1,:], lw=2, label="x true", grid=true, gridalpha=0.3, legendfontsize=9, legend=:outertop, legendcolumns=2, legendborder=false, legend_foreground_color=:transparent, title="Lotka-Volterra Dynamics (VI)", fontsize=10)
plot!(sol.t, obs_noisy[2,:], lw=2, label="y true")
plot!(sol.t, sol_new[1,:], lw=2, ls=:solid, label="x VI", color=:blue)
plot!(sol.t, sol_new[2,:], lw=2, ls=:solid, label="y VI", color=:red)
xlabel!("Time")


