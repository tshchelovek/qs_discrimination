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

module SDPModels
    include("sdp_models.jl")
end
module Distribs
    include("distribs.jl")
end
module Measures
    include("measures.jl")
end
module Fio
    include("fio.jl")
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
# end

function simulator_mixed(d, N)

    ρ = [Ket.random_state(d) for i in 1:N]
    # println("Generated states:\n", ρ)
    println(LinearAlgebra.tr(ρ[1] * ρ[1]))
    
    dual_value, dual_solution = SDPModels.double_dual_model(d, N, ρ)
    # println("Double dual: ", dual_value, "\n", dual_solution)

    # println("Sanity check: 2 >= ", dual_value * N)

    return ρ, dual_value, dual_solution
end

function simulator_pure(d, N)

    ρ = [Ket.random_state(d, 1) for i in 1:N]
    # println("Generated states:\n", ρ)
    println(LinearAlgebra.tr(ρ[1] * ρ[1]))
    
    dual_value, dual_solution = SDPModels.double_dual_model(d, N, ρ)
    println("Double dual: ", dual_value, "\n", dual_solution)

    println("Sanity check: 2 >= ", dual_value * N)

    return ρ, dual_value, dual_solution
end

function simulator_classical(d, N)

    ρ = [Matrix{Float64}(LinearAlgebra.diagm(LinearAlgebra.diag(Ket.random_state(d)))) for i in 1:N]
    # println("Generated states:\n", ρ)
    # println(LinearAlgebra.tr(ρ[1] * ρ[1]))
    
    dual_value, dual_solution = SDPModels.double_dual_model(d, N, ρ)
    # println("Double dual: ", dual_value, "\n", dual_solution)

    # println("Sanity check: 2 >= ", dual_value * N)

    return ρ, dual_value, dual_solution
end

function main()
    d = 2
    N = 4

    #=
        before running check to not overwrite data!
    =#

    for i in 1:100
        rho, objective_value, objective_solution = simulator_mixed(d, N)
        # Fio.write_txt("data/double_dual/mixed", rho, objective_value, objective_solution)
    end

    for i in 1:100
        rho, objective_value, objective_solution = simulator_pure(d, N)
        # Fio.write_txt("data/double_dual/pure", rho, objective_value, objective_solution)
    end

    for i in 1:100
        rho, objective_value, objective_solution = simulator_classical(d, N)
        # Fio.write_txt("data/double_dual/classical", rho, objective_value, objective_solution)
    end
end

main()