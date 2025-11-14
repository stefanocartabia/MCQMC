
using FFTW
function acovf(x::AbstractVector{<:Real})
    n = length(x)
    m = mean(x)
    y = x .- m
    m2 = 2^ceil(Int, log2(2n))          # padding a potenza di 2
    fy = rfft(vcat(y, zeros(m2 - n)))
    ac = irfft(abs.(fy).^2, m2)[1:n]
    ac ./= (n:-1:1)                      # correzione per fine serie
    return ac
end

# ESS con initial positive sequence (Geyer)
function ess_ips(x::AbstractVector{<:Real})
    n = length(x)
    ac = acovf(x)
    γ0 = ac[1]
    ρ =  ac[2:end] ./ γ0            # autocorrelazioni ai lag ≥ 1
    s = 0.0
    k = 1
    while k < length(ρ)
        s_pair = ρ[k] + (k+1 <= length(ρ) ? ρ[k+1] : 0.0)
        if s_pair <= 0
            break
        end
        s += s_pair
        k += 2
    end
    τ = 1 + 2s                           # time-series variance factor
    return n / τ
end

# ESS per più catene: concateno dopo burn-in (assumendo stazionarietà/mescolamento simile)
function ess_ips_multichain(chains::Vector{<:AbstractVector{<:Real}})
    x = vcat(chains...)
    return ess_ips(x)
end