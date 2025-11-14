import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.precompile()
Pkg.add("ForwardDiff")