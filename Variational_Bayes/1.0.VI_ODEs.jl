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
    sol = solve(prob, Tsit5() ;saveat=dt, reltol=1e-6, abstol=1e-8, maxiters=10000, save_everystep=false)     
    return Array(sol)
end

struct Chol_Save{T}
    chol_cov::Cholesky{T,Matrix{T}} 
    logdet :: T 
end

function CholSave(cov::AbstractMatrix{<:Real})
    F = cholesky(Symmetric(Matrix(cov)))
    logdet = 2*sum(log, diag(F.U))   
    return Chol_Save(F, logdet)
end

function build_log_posterior(f!::Function, u0, obs, save_chol_cov::Chol_Save, tspan, dt, logprior_par::Function, d::Integer, K::Integer)
    
    function logpost(theta::AbstractVector)
        par = exp.(theta)
        sim = simulate_system(f!, u0, par, tspan, dt)
        return loglik_gaussian(obs, sim, save_chol_cov, d, K) + logprior_par(par) + sum(theta)
    end
    return logpost
end

struct ODELogPosterior
    logpost::Function
    dim::Int
end

LogDensityProblems.logdensity(m::ODELogPosterior, θ::AbstractVector) = m.logpost(θ)
LogDensityProblems.dimension(m::ODELogPosterior) = m.dim

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

#
init_par = log.([ 1.1,  1.1,  1.1,  1.1])
d_theta = length(init_par)       
save_chol_cov = CholSave(sigma_eta)  
logpost = build_log_posterior(lotka_volterra!, u0, obs_noisy, save_chol_cov, tspan, dt, logprior_par, size(obs_noisy,1), size(obs_noisy,2))
model = ODELogPosterior(logpost, d_theta)

# ----------------- Variational Family ------------------------ # 
μ0 = log.(init_par)
q0 = MeanFieldGaussian(μ0, Diagonal(fill(0.1^2, d_theta)));

n_max_iter = 2000
n_montecarlo = 50
objective = RepGradELBO(n_montecarlo)

q_avg, q_last, stats, state = AdvancedVI.optimize(model, objective, q0, n_max_iter; show_progress = true, adtype = AutoForwardDiff(), optimizer = AdvancedVI.DoWG(), operator =  ClipScale()); 
nothing

plot([i.elbo for i in stats], xlabel="Iterations", ylabel="ELBO", label="ELBO over iterations", title="ELBO Optimisation", ylims=(-150000, Inf))


z = rand(q_last, 10^4);
avg = exp.(vec(mean(z; dims=2)))







