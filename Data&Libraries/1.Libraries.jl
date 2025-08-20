import Pkg

for lib in ["Plots", "LinearAlgebra","CSV", "DataFrames","Distributions", "GLM", "StatsBase", "Statistics"]
    try
        Base.find_package(lib) == nothing && Pkg.add(lib)
    catch
        Pkg.add(lib)
    end
end

# Pkg.add(url="https://github.com/VMLS-book/VMLS.jl"); using VMLS

using DifferentialEquations
using Plots
using LaTeXStrings
using Latexify
using ModelingToolkit
using ParameterizedFunctions
# using VMLS
using LinearAlgebra
using CSV
using DataFrames
using Distributions
using GLM
using StatsBase
using Statistics

