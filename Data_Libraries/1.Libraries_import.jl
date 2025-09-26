import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.add("ForwardDiff")
Pkg.precompile()