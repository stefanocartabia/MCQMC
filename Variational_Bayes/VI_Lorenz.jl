include("C:/Users/mussi/Documents/Manhattan/Leuven/MCQMC/Data_Libraries/1.Libraries_setup.jl")
using Random, Turing, AdvancedVI, FillArrays, RDatasets, Optimisers, QuasiMonteCarlo, DiffResults, 
      DynamicPPL, Zygote, ADTypes, LogDensityProblems, Distributions, DataFrames, StatsFuns
using Bijectors: Bijectors
Pkg.status("AdvancedVI")

###############################################################################################################################################
#####################################################         Variational Inference           #################################################
#####################################################    Lorenz System (Initial Conditions)   #################################################
###############################################################################################################################################

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
    b::B                    # bijector: unconstrained ℝ^m -> to positive constrained space
    dim::Int
end

function LogDensityProblems.logdensity(m::ODELogPosterior, θ::AbstractVector)
    par = m.b(θ)                         # unconstrained → constrained
    lp_param = m.logpost_par(par)        
    return lp_param + Bijectors.logabsdetjac(m.b, θ)
end

LogDensityProblems.dimension(m::ODELogPosterior) = m.dim
LogDensityProblems.capabilities(::Type{<:ODELogPosterior}) = LogDensityProblems.LogDensityOrder{0}()

#----------------------------------------------------------- Data Generation -------------------------------------------------------------------------#
# Simulate True Data
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    x, y, z = u
    du[1] = σ * (y - x)
    du[2] = x * (ρ - z) - y
    du[3] = x*y - β*z
end

tspan = (15.0, 16.0); u0 = [4.19, 0.93, 27.0]; dt = 0.02
par_true = (10.0, 28.0, 8/3)  # σ, ρ, β
lor_sys = ODEProblem(lorenz!, u0, tspan, par_true)
sol = solve(lor_sys, Tsit5(), saveat=dt)

# Noisy Data 
sigma_eta = 1 * I(3)              
obs_noisy = Array(sol) .+ rand(MvNormal(zeros(3), sigma_eta), size(sol, 2));

#----------------------------------------------------------- Orbits  ---------------------------------------------------------------------------------#
p1 = plot(sol[1,:], sol[2,:], sol[3,:], linewidth = 1.5, title = "\nTrue Lorenz Attractor", 
          xlabel = "x", ylabel = "y", zlabel = "z", legend = false, grid = true, gridalpha = 0.3 )
p2 = plot(obs_noisy[1,:], obs_noisy[2,:], obs_noisy[3,:], linewidth = 1.5, title = "\nObserved Lorenz Attractor", 
          xlabel = "x", ylabel = "y", zlabel = "z", legend = false, color = "red", grid = true, gridalpha = 0.3)
plot_traj = plot( p1, p2, layout = (1, 2), size = (1100, 400), top_margin = 3mm)

#----------------------------------------------------------- Time Series  -----------------------------------------------------------------------------#
p1 = scatter(sol.t, obs_noisy[1,:], markersize=3, alpha=0.5, label="x obs", color=:lightblue,
             grid=true, gridalpha=0.3, ylabel="x", legend=:outertop)
plot!(p1, sol.t, sol[1,:], lw=2, label="x true", color=:blue)

p2 = scatter(sol.t, obs_noisy[2,:], markersize=3, alpha=0.5, label="y obs", color=:lightcoral,
             grid=true, gridalpha=0.3, ylabel="y", legend=:outertop)
plot!(p2, sol.t, sol[2,:], lw=2, label="y true", color=:red)

p3 = scatter(sol.t, obs_noisy[3,:], markersize=3, alpha=0.5, label="z obs", color=:lightgreen,
             grid=true, gridalpha=0.3, ylabel="z", xlabel="Time", legend=:outertop)
plot!(p3, sol.t, sol[3,:], lw=2, label="z true", color=:green)

lorenz_ts = plot(p1, p2, p3, layout=(3,1), size=(1000, 800), 
                plot_title="\n Lorenz System Dynamics", 
                left_margin=5mm, bottom_margin=5mm, top_margin=5mm, legendcolumns=2,
                legendborder=false, legend_foreground_color=:transparent)


# ------------------------------------------------ VI Optimisation -----------------------------------------------------------# 

priors = (
          Gamma(2, 6),    # σ ~ Gamma(2, 6)
          Gamma(2, 15),   # ρ ~ Gamma(2, 15)  
          Gamma(2, 2)     # β ~ Gamma(2, 2)       
)

logprior_par = p -> sum(logpdf.(priors, p))

d_par = 3
b_forward = Bijectors.elementwise(exp)  

save_chol_cov = CholSave(sigma_eta)  
logpost_par_fn = make_logpost_par(lorenz!, u0, obs_noisy, save_chol_cov, tspan, dt, logprior_par)
model = ODELogPosterior(logpost_par_fn, b_forward, d_par)


init_par = [5.0, 25.0, 1.0]  
μ0 = log.(init_par)                 
q0 = MeanFieldGaussian(μ0, Diagonal(fill(0.1^2, d_par)))


n_max_iter = 10000
n_montecarlo = 20
objective = RepGradELBO(n_montecarlo)

q_avg, q_last, stats, state = AdvancedVI.optimize(model, objective, q0, n_max_iter; show_progress = true, adtype = AutoForwardDiff(), 
                                                  optimizer = Optimisers.Adam(1e-3), operator = ClipScale() );
nothing

z_unconstrained = rand(q_last, 10^4)  
z_constrained = b_forward(z_unconstrained) 
avg = vec(mean(z_constrained; dims=2))

println("Estimated parameters (mean): ", avg)
println("True parameters: ", collect(par_true))

#-----------------------------------------------------------------------------------------------------------------------------#  
# Elbo Optimisatation Plot

VI_6000_10 = plot([i.elbo for i in stats], xlabel="Iterations", ylabel="ELBO", label="ELBO over iterations", 
                   title=" ELBO Optimisation Lorenz System ",  legend=:outertop, legendborder=false, legend_foreground_color=:transparent) 
savefig(VI_6000_10, raw"C:\Users\mussi\Documents\Manhattan\Leuven\MCQMC\Plots\VI\Lorenz_15_16_ELBO.png")

# Time series plot 
lor_sys_new = ODEProblem(lorenz!, u0, tspan, tuple(avg...))
sol_new = solve(lor_sys_new, Tsit5(), saveat = dt)

p1 = scatter(sol.t, obs_noisy[1,:], markersize=3, alpha=0.5, label="x Observed", color=:lightblue,
             grid=true, gridalpha=0.3, ylabel="x", legend=:outertop)
plot!(p1, sol.t, sol_new[1,:], lw=2, label="x Reconstructed", color=:blue)

p2 = scatter(sol.t, obs_noisy[2,:], markersize=3, alpha=0.5, label="y Observed", color=:lightcoral,
             grid=true, gridalpha=0.3, ylabel="y", legend=:outertop)
plot!(p2, sol.t, sol_new[2,:], lw=2, label="y Reconstructed", color=:red)

p3 = scatter(sol.t, obs_noisy[3,:], markersize=3, alpha=0.5, label="z Observed", color=:lightgreen,
             grid=true, gridalpha=0.3, ylabel="z", xlabel="Time", legend=:outertop)
plot!(p3, sol.t, sol_new[3,:], lw=2, label="z Reconstructed", color=:green)

lorenz_ts = plot(p1, p2, p3, layout=(3,1), size=(1000, 800), 
                plot_title="Lorenz Dynamics (VI Reconstruction)", 
                left_margin=5mm, bottom_margin=5mm, top_margin=5mm, legendcolumns=2,
                legendborder=false, legend_foreground_color=:transparent, legend=:outertop)

savefig(lorenz_ts, raw"C:\Users\mussi\Documents\Manhattan\Leuven\MCQMC\Plots\VI\Lorenz_15_16_VIReconstruction.png")               
#-----------------------------------------------------------------------------------------------------------------------------#