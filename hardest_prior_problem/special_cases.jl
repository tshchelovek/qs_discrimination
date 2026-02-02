#=
double_dual_hardest_prior:
- Julia version: 1.10.5
- Author: ChernyshovaP
- Date: 2025-12-17
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

function simulator_mixed(N, d)
    # all the same random state N times

    state = Ket.random_state(d)
    ρ = [state for i in 1:N]
    println("Generated states:\n", ρ)
    # println(LinearAlgebra.tr(ρ[1] * ρ[1]))
    
    # dual_value, dual_solution = SDPModels.double_dual_model(N, d, ρ)
    # println("Double dual: ", dual_value, "\n", dual_solution)

    # println("Sanity check: 2 >= ", dual_value * N)

    return ρ, dual_value, dual_solution
end

function simulator_mixed(N, d)
    # "normalized" - the first state is put to fully mixed one

    ρ = [Ket.random_state(d) for i in 1:N]
    ρ[1] = Ket.pauli(0) / N
    println("Generated states:\n", ρ)
    # println(LinearAlgebra.tr(ρ[1] * ρ[1]))
    
    # dual_value, dual_solution = SDPModels.double_dual_model(N, d, ρ)
    # println("Double dual: ", dual_value, "\n", dual_solution)

    # println("Sanity check: 2 >= ", dual_value * N)

    return ρ, dual_value, dual_solution
end

function simulation_triangle_ineq(file_name, N = 0, d = 0)
    #= 
    testing if solution to double dual forms a well-defined distance on the set of states
    since the other three conditions can be easily achieved by renormalizing - d(x,x)=0, d(x,y)>0, d(x,y)=d(y,x)
    actually for any number of states

    okay it works analitically but it's still not a metric
    =#

    data = Fio.read_txt(file_name, N, d)
    data[1]
end

function main()
    N = 2
    d = 2

    #=
        before running check to not overwrite data!
    =#

    for i in 1:100
        rho, objective_value, objective_solution = simulation_triangle_ineq(N, d)
        # Fio.write_txt("data/special_cases/tri_ineq", rho, objective_value, objective_solution)
    end

end

simulation_triangle_ineq("data/double_dual/mixed_2_2.txt")