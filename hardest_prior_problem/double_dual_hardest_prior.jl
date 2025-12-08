#=
double_dual_hardest_prior:
- Julia version: 1.10.5
- Author: ChernyshovaP
- Date: 2025-12-01
=#

using JuMP, MosekTools, IsApprox, Random, Ket
import LinearAlgebra
import Dualization
import SCS
import Mosek

module FixedPrior
    include("fixed_prior.jl")
end
module Distribs
    include("distribs.jl")
end
module Measures
    include("measures.jl")
end

# function probs_lp(vals, full_sum)
#     #= 
#       finding appropriate values for probabilities given a double-dual problem solution
#       in fact the double-dual solution doesn't depend on the prior so the distribution doesn't matter
#     =#
#     N = size(vals)
#     model = Model(Ipopt.Optimizer)
#     set_silent(model)
#     @variable(model, p[1:N])
#     @constraint(model, full_sum == vals * p)
#     @constraint(model, sum(p) == 1)
#     [@constraint(model, p[i] >= 0) for i in 1:N]
#     @objective(model, Min, sum(p.^2))
#     optimize!(model)
#     return value.(p)
# end

function sdp_solver()
    N, d = 3, 2

    ρ = [random_state(d) for i in 1:N]
    # ρ = [[1 0 ; 0 0], [0.5 0.5 ; 0.5 0.5]] # |0> and |+>
    # ρ = [[1 0 ; 0 0], [0 0 ; 0 1]]
    # ρ = [[1 0 ; 0 0], [0.5 0.5 ; 0.5 0.5], [0 0 ; 0 1]]
    # ρ = [[1 0 ; 0 0], [0.81 0.39 ; 0.39 0.19], [0.64 0.48 ; 0.48 0.36]]
    # ρ = [[1 0 ; 0 0], [0 0 ; 0 1], [0.5 0.5 ; 0.5 0.5], [0.5 -0.5 ; -0.5 0.5]]
    # ρ = [[1 0 ; 0 0], [0.1464466094067263 0.35355339059327384 ; 0.35355339059327384 0.8535533905932737], [0.1464466094067263 -0.35355339059327384 ; -0.35355339059327384 0.8535533905932737]] #this almost saturates the bound, 1.975 -> 2=d

    dual_value, dual_solution = FixedPrior.double_dual_model(N, d, ρ)
    println("Double dual: ", dual_value, "\n", dual_solution)

    println(dual_value * N)


    #= 
        so far for what i've checked, PGM gives better probability
        (for a different initial problem setting nevertheless)
        so the POVMs i get are not PGM
    =#
    # pgm_value, pgm_solution = Measures.PGM(ρ)
    # println("PGM: ", pgm_value, "\n", pgm_solution)

    #= 
        this should be checked (analitically) but usually all the values are the same
        and equal to the objective of SDP problem by definition
    =#
    # products = [0. for i in 1:N]
    # for idx in 1:N
    #     val = real(LinearAlgebra.tr(ρ[idx] * dual_solution[idx]))
    #     println(val)
    #     products[idx] = val
    # end
    # println(products)

end

sdp_solver()