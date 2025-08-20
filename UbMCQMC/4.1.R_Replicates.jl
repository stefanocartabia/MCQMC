include("3.F_k_m.jl")


######################################## E[F_k_m] and Var[F_k_m] #################################################
########################################     Unbiased    #########################################################
# This script aims to estimate the unbiased F_k_m over R indipendent chains to derive the mean and variance 

function R_UbFkm(R::Int, m::Int,seq,mdl,k,f)

    p = mdl.p
    R_Fkm = Matrix{Float64}(p,R)
    for r in R
        R_Fkm[:,r] .= UbFkm(m,seq,mdl,k,f)
    end

    mu_pool = mean(R_Fkm; dim=2)
    sigma_pool = sum((R_Fkm .- mu_pool).^2; dims=2) ./ (R*(R-1))

    return (
        mu_pool = mu_pool,
        sigma_pool = sigma_pool
    )
    
end


########################################### E[F_k_m] and Var[F_k_m] #################################################
#############################################       Biased       ####################################################


function R_Fkm(R::Int, m::Int,seq,mdl,k,f)

    p = mdl.p
    R_Fkm = Matrix{Float64}(p,R)
    for r in R
        R_Fkm[:,r] .= Fkm(m,seq,mdl,k,f)
    end

    mu_pool = mean(R_Fkm; dim=2)
    sigma_pool = sum((R_Fkm .- mu_pool).^2; dims=2) ./ (R*(R-1))

    return (
        mu_pool = mu_pool,
        sigma_pool = sigma_pool
    )
    
end