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

function simulator_mixed(d, N)
    # all the same random state N times

    state = Ket.random_state(d)
    ρ = [state for i in 1:N]
    println("Generated states:\n", ρ)
    # println(LinearAlgebra.tr(ρ[1] * ρ[1]))
    
    # dual_value, dual_solution = SDPModels.double_dual_model(ρ)
    # println("Double dual: ", dual_value, "\n", dual_solution)

    # println("Sanity check: 2 >= ", dual_value * N)

    return ρ, dual_value, dual_solution
end

function simulator_mixed_covariant(d, N)
    # all the same random state N times

    state = Ket.random_state(d)
    unitary = Ket.random_unitary(d)
    ρ = [state for i in 1:N]
    for i in 2:N
        ρ[i] = Hermitian(unitary^i * state * unitary'^i)
    end
    println("Generated states:\n", ρ)
    # println(LinearAlgebra.tr(ρ[1] * ρ[1]))
    
    dual_value, dual_solution, dual_prior = SDPModels.dual_model(ρ)
    # println("Double dual: ", dual_value, "\n", dual_solution)

    # println("Sanity check: 2 >= ", dual_value * N)

    return ρ, dual_value, dual_solution, dual_prior
end

function simulator_mixed(d, N)
    # "normalized" - the first state is put to fully mixed one

    ρ = [Ket.random_state(d) for i in 1:N]
    ρ[1] = Ket.pauli(0) / N
    println("Generated states:\n", ρ)
    # println(LinearAlgebra.tr(ρ[1] * ρ[1]))
    
    # dual_value, dual_solution = SDPModels.double_dual_model(ρ)
    # println("Double dual: ", dual_value, "\n", dual_solution)

    # println("Sanity check: 2 >= ", dual_value * N)

    return ρ, dual_value, dual_solution
end

function simulation_triangle_ineq(file_name, d = 0, N = 0)
    #= 
    testing if solution to double dual forms a well-defined distance on the set of states
    since the other three conditions can be easily achieved by renormalizing - d(x,x)=0, d(x,y)>0, d(x,y)=d(y,x)
    actually for any number of states

    okay it works analitically but it's still not a metric
    =#

    data = Fio.read_txt(file_name, d, N)
    data[1]
end

function simulator_spherical_design()
    des321 = [[0.5 + 0.0im 0.5 + 0.0im; 0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im -0.5 - 0.0im; -0.5 + 0.0im 0.5 + 0.0im]]
    des331 = [[0.5 + 0.0im 0.5 + 0.0im; 0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im -0.25 - 0.4330127018922193im; -0.25 + 0.4330127018922193im 0.5 + 0.0im], [0.5 + 0.0im -0.25 + 0.4330127018922193im; -0.25 - 0.4330127018922193im 0.5 + 0.0im]]
    des342 = [[0.7886751345948129 + 0.0im 0.28867513459481287 - 0.28867513459481287im; 0.28867513459481287 + 0.28867513459481287im 0.21132486540518705 + 0.0im], [0.21132486540518716 + 0.0im 0.28867513459481287 + 0.28867513459481287im; 0.28867513459481287 - 0.28867513459481287im 0.7886751345948125 + 0.0im], [0.21132486540518716 + 0.0im -0.28867513459481287 - 0.28867513459481287im; -0.28867513459481287 + 0.28867513459481287im 0.7886751345948125 + 0.0im], [0.7886751345948129 + 0.0im -0.28867513459481287 + 0.28867513459481287im; -0.28867513459481287 - 0.28867513459481287im 0.21132486540518705 + 0.0im]]
    des351 = [[0.5 + 0.0im 0.5 + 0.0im; 0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im -0.5 - 0.0im; -0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im 0.5 + 0.0im; 0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im -0.25 - 0.4330127018922193im; -0.25 + 0.4330127018922193im 0.5 + 0.0im], [0.5 + 0.0im -0.25 + 0.4330127018922193im; -0.25 - 0.4330127018922193im 0.5 + 0.0im]]
    des363 = [[0.5 + 0.0im 0.5 + 0.0im; 0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im -0.5 - 0.0im; -0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im 0.0 - 0.5im; 0.0 + 0.5im 0.5 + 0.0im], [0.5 + 0.0im 0.0 + 0.5im; 0.0 - 0.5im 0.5 + 0.0im], [1.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im], [0 0; 0 1]]
    des372 = [[1.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im], [0.19618739074461747 + 0.0im 0.3971119470091982 + 0.0im; 0.3971119470091982 + 0.0im 0.8038126092553826 + 0.0im], [0.19618739074461747 + 0.0im -0.1985559735045991 - 0.34390903425626546im; -0.1985559735045991 + 0.34390903425626546im 0.8038126092553826 + 0.0im], [0.19618739074461747 + 0.0im -0.1985559735045991 + 0.34390903425626546im; -0.1985559735045991 - 0.34390903425626546im 0.8038126092553826 + 0.0im], [0.6371459425887157 + 0.0im 0.4808232423993796 + 0.0im; 0.4808232423993796 + 0.0im 0.3628540574112841 + 0.0im], [0.6371459425887157 + 0.0im -0.2404116211996898 - 0.4164051426478659im; -0.2404116211996898 + 0.4164051426478659im 0.36285405741128424 + 0.0im], [0.6371459425887157 + 0.0im -0.2404116211996898 + 0.4164051426478659im; -0.2404116211996898 - 0.4164051426478659im 0.36285405741128424 + 0.0im]]
    des383 = [[0.7886751345948129 + 0.0im 0.28867513459481287 - 0.28867513459481287im; 0.28867513459481287 + 0.28867513459481287im 0.21132486540518705 + 0.0im], [0.21132486540518716 + 0.0im 0.28867513459481287 - 0.28867513459481287im; 0.28867513459481287 + 0.28867513459481287im 0.7886751345948125 + 0.0im], [0.7886751345948129 + 0.0im 0.28867513459481287 + 0.28867513459481287im; 0.28867513459481287 - 0.28867513459481287im 0.21132486540518705 + 0.0im], [0.7886751345948129 + 0.0im -0.28867513459481287 - 0.28867513459481287im; -0.28867513459481287 + 0.28867513459481287im 0.21132486540518705 + 0.0im], [0.21132486540518716 + 0.0im 0.28867513459481287 + 0.28867513459481287im; 0.28867513459481287 - 0.28867513459481287im 0.7886751345948125 + 0.0im], [0.21132486540518716 + 0.0im -0.28867513459481287 - 0.28867513459481287im; -0.28867513459481287 + 0.28867513459481287im 0.7886751345948125 + 0.0im], [0.7886751345948129 + 0.0im -0.28867513459481287 + 0.28867513459481287im; -0.28867513459481287 - 0.28867513459481287im 0.21132486540518705 + 0.0im], [0.21132486540518716 + 0.0im -0.28867513459481287 + 0.28867513459481287im; -0.28867513459481287 - 0.28867513459481287im 0.7886751345948125 + 0.0im]]

    d = 2
    N = 2
    ρ = des321
    
    precision = 4

    dual_value, dual_solution, dual_prior = SDPModels.dual_model(ρ)
    println(round(dual_value, digits = precision), ' ', round.(dual_prior, digits = precision))
    primal_value, primal_solution = SDPModels.primal_model(ρ, dual_prior)
    # println("Primal POVM:\n", [round.(primal_solution[i], digits = precision) for i in 1:size(primal_solution)[1]])
    # double_dual_value, double_dual_solution = SDPModels.double_dual_model(ρ)
    # println("Double dual POVM:\n", [round.(double_dual_solution[i], digits = precision) for i in 1:size(double_dual_solution)[1]])
    println(Distribs.posterior_distr(primal_solution, ρ))
end

function simulator_sphere_id()
    des2_1 = [[1/2 0; 0 1/2], [0.5 + 0.0im 0.5 + 0.0im; 0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im -0.5 - 0.0im; -0.5 + 0.0im 0.5 + 0.0im]]
    des3_1 = [[1/2 0; 0 1/2], [0.5 + 0.0im 0.5 + 0.0im; 0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im -0.25 - 0.4330127018922193im; -0.25 + 0.4330127018922193im 0.5 + 0.0im], [0.5 + 0.0im -0.25 + 0.4330127018922193im; -0.25 - 0.4330127018922193im 0.5 + 0.0im]]
    des4_1 = [[1/2 0; 0 1/2], [0.7886751345948129 + 0.0im 0.28867513459481287 - 0.28867513459481287im; 0.28867513459481287 + 0.28867513459481287im 0.21132486540518705 + 0.0im], [0.21132486540518716 + 0.0im 0.28867513459481287 + 0.28867513459481287im; 0.28867513459481287 - 0.28867513459481287im 0.7886751345948125 + 0.0im], [0.21132486540518716 + 0.0im -0.28867513459481287 - 0.28867513459481287im; -0.28867513459481287 + 0.28867513459481287im 0.7886751345948125 + 0.0im], [0.7886751345948129 + 0.0im -0.28867513459481287 + 0.28867513459481287im; -0.28867513459481287 - 0.28867513459481287im 0.21132486540518705 + 0.0im]]
    des5_1 = [[1/2 0; 0 1/2], [0.5 + 0.0im 0.5 + 0.0im; 0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im -0.5 - 0.0im; -0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im 0.5 + 0.0im; 0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im -0.25 - 0.4330127018922193im; -0.25 + 0.4330127018922193im 0.5 + 0.0im], [0.5 + 0.0im -0.25 + 0.4330127018922193im; -0.25 - 0.4330127018922193im 0.5 + 0.0im]]
    des6_1 = [[1/2 0; 0 1/2], [0.5 + 0.0im 0.5 + 0.0im; 0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im -0.5 - 0.0im; -0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im 0.0 - 0.5im; 0.0 + 0.5im 0.5 + 0.0im], [0.5 + 0.0im 0.0 + 0.5im; 0.0 - 0.5im 0.5 + 0.0im], [1.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im], [0 0; 0 1]]
    des7_1 = [[1/2 0; 0 1/2], [1.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im], [0.19618739074461747 + 0.0im 0.3971119470091982 + 0.0im; 0.3971119470091982 + 0.0im 0.8038126092553826 + 0.0im], [0.19618739074461747 + 0.0im -0.1985559735045991 - 0.34390903425626546im; -0.1985559735045991 + 0.34390903425626546im 0.8038126092553826 + 0.0im], [0.19618739074461747 + 0.0im -0.1985559735045991 + 0.34390903425626546im; -0.1985559735045991 - 0.34390903425626546im 0.8038126092553826 + 0.0im], [0.6371459425887157 + 0.0im 0.4808232423993796 + 0.0im; 0.4808232423993796 + 0.0im 0.3628540574112841 + 0.0im], [0.6371459425887157 + 0.0im -0.2404116211996898 - 0.4164051426478659im; -0.2404116211996898 + 0.4164051426478659im 0.36285405741128424 + 0.0im], [0.6371459425887157 + 0.0im -0.2404116211996898 + 0.4164051426478659im; -0.2404116211996898 - 0.4164051426478659im 0.36285405741128424 + 0.0im]]
    des8_1 = [[1/2 0; 0 1/2], [0.7886751345948129 + 0.0im 0.28867513459481287 - 0.28867513459481287im; 0.28867513459481287 + 0.28867513459481287im 0.21132486540518705 + 0.0im], [0.21132486540518716 + 0.0im 0.28867513459481287 - 0.28867513459481287im; 0.28867513459481287 + 0.28867513459481287im 0.7886751345948125 + 0.0im], [0.7886751345948129 + 0.0im 0.28867513459481287 + 0.28867513459481287im; 0.28867513459481287 - 0.28867513459481287im 0.21132486540518705 + 0.0im], [0.7886751345948129 + 0.0im -0.28867513459481287 - 0.28867513459481287im; -0.28867513459481287 + 0.28867513459481287im 0.21132486540518705 + 0.0im], [0.21132486540518716 + 0.0im 0.28867513459481287 + 0.28867513459481287im; 0.28867513459481287 - 0.28867513459481287im 0.7886751345948125 + 0.0im], [0.21132486540518716 + 0.0im -0.28867513459481287 - 0.28867513459481287im; -0.28867513459481287 + 0.28867513459481287im 0.7886751345948125 + 0.0im], [0.7886751345948129 + 0.0im -0.28867513459481287 + 0.28867513459481287im; -0.28867513459481287 - 0.28867513459481287im 0.21132486540518705 + 0.0im], [0.21132486540518716 + 0.0im -0.28867513459481287 + 0.28867513459481287im; -0.28867513459481287 - 0.28867513459481287im 0.7886751345948125 + 0.0im]]

    ρ = des2_1
    d = 2
    N = length(ρ)
    
    precision = 4

    dual_value, dual_solution, dual_prior = SDPModels.dual_model(ρ)
    println(round(dual_value, digits = precision), ' ', round.(dual_prior, digits = precision))
    primal_value, primal_solution = SDPModels.primal_model(ρ, dual_prior)
    # println("Primal POVM:\n", [round.(primal_solution[i], digits = precision) for i in 1:size(primal_solution)[1]])
    # double_dual_value, double_dual_solution = SDPModels.double_dual_model(ρ)
    # println("Double dual POVM:\n", [round.(double_dual_solution[i], digits = precision) for i in 1:size(double_dual_solution)[1]])
    println(Distribs.posterior_distr(primal_solution, ρ))
end

function main()
    d = 2
    N = 4

    #=
        before running check to not overwrite data!
    =#

    for i in 1:100
        rho, objective_value, objective_solution, objective_prior = simulator_mixed_covariant(d, N)

        # Fio.write_txt("data/special_cases/mixed_covariant", rho, objective_value, objective_solution, objective_prior)
    end

end

# main()

simulator_sphere_id()