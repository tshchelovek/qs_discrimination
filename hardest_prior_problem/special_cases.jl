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

function kron_k_times(density_matrix, k)
    tensor_product = density_matrix
    for i in 1:k-1
        tensor_product = [LinearAlgebra.kron(tensor_product[i], density_matrix[i]) for i in eachindex(density_matrix)]
    end
    return tensor_product
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

function simulator_spherical_design(N = 2, k = 1)
    d = 2
    sets = [[[0.5 + 0.0im 0.5 + 0.0im; 0.5 + 0.0im 0.5 + 0.0im]],
        [[0.5 + 0.0im 0.5 + 0.0im; 0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im -0.5 - 0.0im; -0.5 + 0.0im 0.5 + 0.0im]],
        [[0.5 + 0.0im 0.5 + 0.0im; 0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im -0.25 - 0.4330127018922193im; -0.25 + 0.4330127018922193im 0.5 + 0.0im], [0.5 + 0.0im -0.25 + 0.4330127018922193im; -0.25 - 0.4330127018922193im 0.5 + 0.0im]],
        [[0.7886751345948129 + 0.0im 0.28867513459481287 - 0.28867513459481287im; 0.28867513459481287 + 0.28867513459481287im 0.21132486540518705 + 0.0im], [0.21132486540518716 + 0.0im 0.28867513459481287 + 0.28867513459481287im; 0.28867513459481287 - 0.28867513459481287im 0.7886751345948125 + 0.0im], [0.21132486540518716 + 0.0im -0.28867513459481287 - 0.28867513459481287im; -0.28867513459481287 + 0.28867513459481287im 0.7886751345948125 + 0.0im], [0.7886751345948129 + 0.0im -0.28867513459481287 + 0.28867513459481287im; -0.28867513459481287 - 0.28867513459481287im 0.21132486540518705 + 0.0im]],
        [[0.5 + 0.0im 0.5 + 0.0im; 0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im -0.5 - 0.0im; -0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im 0.5 + 0.0im; 0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im -0.25 - 0.4330127018922193im; -0.25 + 0.4330127018922193im 0.5 + 0.0im], [0.5 + 0.0im -0.25 + 0.4330127018922193im; -0.25 - 0.4330127018922193im 0.5 + 0.0im]],
        [[0.5 + 0.0im 0.5 + 0.0im; 0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im -0.5 - 0.0im; -0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im 0.0 - 0.5im; 0.0 + 0.5im 0.5 + 0.0im], [0.5 + 0.0im 0.0 + 0.5im; 0.0 - 0.5im 0.5 + 0.0im], [1.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im], [0 0; 0 1]],
        [[1.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im], [0.19618739074461747 + 0.0im 0.3971119470091982 + 0.0im; 0.3971119470091982 + 0.0im 0.8038126092553826 + 0.0im], [0.19618739074461747 + 0.0im -0.1985559735045991 - 0.34390903425626546im; -0.1985559735045991 + 0.34390903425626546im 0.8038126092553826 + 0.0im], [0.19618739074461747 + 0.0im -0.1985559735045991 + 0.34390903425626546im; -0.1985559735045991 - 0.34390903425626546im 0.8038126092553826 + 0.0im], [0.6371459425887157 + 0.0im 0.4808232423993796 + 0.0im; 0.4808232423993796 + 0.0im 0.3628540574112841 + 0.0im], [0.6371459425887157 + 0.0im -0.2404116211996898 - 0.4164051426478659im; -0.2404116211996898 + 0.4164051426478659im 0.36285405741128424 + 0.0im], [0.6371459425887157 + 0.0im -0.2404116211996898 + 0.4164051426478659im; -0.2404116211996898 - 0.4164051426478659im 0.36285405741128424 + 0.0im]],
        [[0.7886751345948129 + 0.0im 0.28867513459481287 - 0.28867513459481287im; 0.28867513459481287 + 0.28867513459481287im 0.21132486540518705 + 0.0im], [0.21132486540518716 + 0.0im 0.28867513459481287 - 0.28867513459481287im; 0.28867513459481287 + 0.28867513459481287im 0.7886751345948125 + 0.0im], [0.7886751345948129 + 0.0im 0.28867513459481287 + 0.28867513459481287im; 0.28867513459481287 - 0.28867513459481287im 0.21132486540518705 + 0.0im], [0.7886751345948129 + 0.0im -0.28867513459481287 - 0.28867513459481287im; -0.28867513459481287 + 0.28867513459481287im 0.21132486540518705 + 0.0im], [0.21132486540518716 + 0.0im 0.28867513459481287 + 0.28867513459481287im; 0.28867513459481287 - 0.28867513459481287im 0.7886751345948125 + 0.0im], [0.21132486540518716 + 0.0im -0.28867513459481287 - 0.28867513459481287im; -0.28867513459481287 + 0.28867513459481287im 0.7886751345948125 + 0.0im], [0.7886751345948129 + 0.0im -0.28867513459481287 + 0.28867513459481287im; -0.28867513459481287 - 0.28867513459481287im 0.21132486540518705 + 0.0im], [0.21132486540518716 + 0.0im -0.28867513459481287 + 0.28867513459481287im; -0.28867513459481287 - 0.28867513459481287im 0.7886751345948125 + 0.0im]]]

    # ρ = kron_k_times(sets[N] * d / N, k) - sum(rho)=identity
    ρ = kron_k_times(sets[N], k) # sum(rho) = identity * N/d, tr(rho[i]) = 1
    # ρ = des383
    precision = 5
    for i in 1:N
        println("purity?: ", round(real(LinearAlgebra.tr(ρ[i]^2)), digits = precision))
    end

    dual_value, dual_solution, dual_prior = SDPModels.dual_model(ρ)
    println(round(dual_value, digits = precision), ' ', round.(dual_prior, digits = precision))
    primal_value, primal_solution = SDPModels.primal_model(ρ, dual_prior)
    # println("Primal POVM:\n", [round.(primal_solution[i], digits = precision) for i in 1:size(primal_solution)[1]])
    double_dual_value, double_dual_povm, double_dual_q = SDPModels.double_dual_model_with_q(ρ)
    # for i in 1:N
    #     for j in 1:N
    #         print(round(real(LinearAlgebra.tr(ρ[i] * double_dual_povm[j])), digits = precision), " ")
    #     end
    #     print("\n")
    # end
    println("Double dual POVM:\n", [round.(double_dual_povm[i], digits = precision) for i in 1:size(double_dual_povm)[1]])
    # pgm_povm = Measures.PGM_observables(ρ)
    # println("PGM POVM:\n", [round.(pgm_povm[i], digits = precision) for i in 1:size(pgm_povm)[1]])
    # println("states:\n", [round.(ρ[i], digits = precision) for i in 1:size(ρ)[1]])
    # println("sum of states: ", sum(ρ))
    # println(Distribs.posterior_distr(double_dual_povm, ρ))
end

function simulator_sphere_id()
    des2_1 = [[1/2 0; 0 1/2], [0.5 + 0.0im 0.5 + 0.0im; 0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im -0.5 - 0.0im; -0.5 + 0.0im 0.5 + 0.0im]]
    des3_1 = [[1/2 0; 0 1/2], [0.5 + 0.0im 0.5 + 0.0im; 0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im -0.25 - 0.4330127018922193im; -0.25 + 0.4330127018922193im 0.5 + 0.0im], [0.5 + 0.0im -0.25 + 0.4330127018922193im; -0.25 - 0.4330127018922193im 0.5 + 0.0im]]
    des4_1 = [[1/2 0; 0 1/2], [0.7886751345948129 + 0.0im 0.28867513459481287 - 0.28867513459481287im; 0.28867513459481287 + 0.28867513459481287im 0.21132486540518705 + 0.0im], [0.21132486540518716 + 0.0im 0.28867513459481287 + 0.28867513459481287im; 0.28867513459481287 - 0.28867513459481287im 0.7886751345948125 + 0.0im], [0.21132486540518716 + 0.0im -0.28867513459481287 - 0.28867513459481287im; -0.28867513459481287 + 0.28867513459481287im 0.7886751345948125 + 0.0im], [0.7886751345948129 + 0.0im -0.28867513459481287 + 0.28867513459481287im; -0.28867513459481287 - 0.28867513459481287im 0.21132486540518705 + 0.0im]]
    des5_1 = [[1/2 0; 0 1/2], [0.5 + 0.0im 0.5 + 0.0im; 0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im -0.5 - 0.0im; -0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im 0.5 + 0.0im; 0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im -0.25 - 0.4330127018922193im; -0.25 + 0.4330127018922193im 0.5 + 0.0im], [0.5 + 0.0im -0.25 + 0.4330127018922193im; -0.25 - 0.4330127018922193im 0.5 + 0.0im]]
    des6_1 = [[1/2 0; 0 1/2], [0.5 + 0.0im 0.5 + 0.0im; 0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im -0.5 - 0.0im; -0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im 0.0 - 0.5im; 0.0 + 0.5im 0.5 + 0.0im], [0.5 + 0.0im 0.0 + 0.5im; 0.0 - 0.5im 0.5 + 0.0im], [1.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im], [0 0; 0 1]]
    des7_1 = [[1/2 0; 0 1/2], [1.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im], [0.19618739074461747 + 0.0im 0.3971119470091982 + 0.0im; 0.3971119470091982 + 0.0im 0.8038126092553826 + 0.0im], [0.19618739074461747 + 0.0im -0.1985559735045991 - 0.34390903425626546im; -0.1985559735045991 + 0.34390903425626546im 0.8038126092553826 + 0.0im], [0.19618739074461747 + 0.0im -0.1985559735045991 + 0.34390903425626546im; -0.1985559735045991 - 0.34390903425626546im 0.8038126092553826 + 0.0im], [0.6371459425887157 + 0.0im 0.4808232423993796 + 0.0im; 0.4808232423993796 + 0.0im 0.3628540574112841 + 0.0im], [0.6371459425887157 + 0.0im -0.2404116211996898 - 0.4164051426478659im; -0.2404116211996898 + 0.4164051426478659im 0.36285405741128424 + 0.0im], [0.6371459425887157 + 0.0im -0.2404116211996898 + 0.4164051426478659im; -0.2404116211996898 - 0.4164051426478659im 0.36285405741128424 + 0.0im]]
    des8_1 = [[1/2 0; 0 1/2], [0.7886751345948129 + 0.0im 0.28867513459481287 - 0.28867513459481287im; 0.28867513459481287 + 0.28867513459481287im 0.21132486540518705 + 0.0im], [0.21132486540518716 + 0.0im 0.28867513459481287 - 0.28867513459481287im; 0.28867513459481287 + 0.28867513459481287im 0.7886751345948125 + 0.0im], [0.7886751345948129 + 0.0im 0.28867513459481287 + 0.28867513459481287im; 0.28867513459481287 - 0.28867513459481287im 0.21132486540518705 + 0.0im], [0.7886751345948129 + 0.0im -0.28867513459481287 - 0.28867513459481287im; -0.28867513459481287 + 0.28867513459481287im 0.21132486540518705 + 0.0im], [0.21132486540518716 + 0.0im 0.28867513459481287 + 0.28867513459481287im; 0.28867513459481287 - 0.28867513459481287im 0.7886751345948125 + 0.0im], [0.21132486540518716 + 0.0im -0.28867513459481287 - 0.28867513459481287im; -0.28867513459481287 + 0.28867513459481287im 0.7886751345948125 + 0.0im], [0.7886751345948129 + 0.0im -0.28867513459481287 + 0.28867513459481287im; -0.28867513459481287 - 0.28867513459481287im 0.21132486540518705 + 0.0im], [0.21132486540518716 + 0.0im -0.28867513459481287 + 0.28867513459481287im; -0.28867513459481287 - 0.28867513459481287im 0.7886751345948125 + 0.0im]]

    ρ = des3_1
    precision = 4

    dual_value, dual_solution, dual_prior = SDPModels.dual_model(ρ)
    println(round(dual_value, digits = precision), ' ', round.(dual_prior, digits = precision))
    primal_value, primal_solution = SDPModels.primal_model(ρ, dual_prior)
    # println("Primal POVM:\n", [round.(primal_solution[i], digits = precision) for i in 1:size(primal_solution)[1]])
    # double_dual_value, double_dual_solution = SDPModels.double_dual_model(ρ)
    # println("Double dual POVM:\n", [round.(double_dual_solution[i], digits = precision) for i in 1:size(double_dual_solution)[1]])
    println(Distribs.posterior_distr(primal_solution, ρ))
end

function simulator_non_design_tmd()
    # it was still a design in the end
    tetr = [[0.7886751345948129 + 0.0im 0.28867513459481287 - 0.28867513459481287im; 0.28867513459481287 + 0.28867513459481287im 0.21132486540518705 + 0.0im], [0.21132486540518716 + 0.0im 0.28867513459481287 + 0.28867513459481287im; 0.28867513459481287 - 0.28867513459481287im 0.7886751345948125 + 0.0im], [0.21132486540518716 + 0.0im -0.28867513459481287 - 0.28867513459481287im; -0.28867513459481287 + 0.28867513459481287im 0.7886751345948125 + 0.0im], [0.7886751345948129 + 0.0im -0.28867513459481287 + 0.28867513459481287im; -0.28867513459481287 - 0.28867513459481287im 0.21132486540518705 + 0.0im]]
    almost_tetr = [[1.0+0.0im  0.0; 0.0     0.0], [0.25+0.0im  0.433-0.0im; 0.433+0.0im   0.75+0.0im], [0.25+0.0im    -0.2165-0.375im; -0.2165+0.375im     0.75+0.0im], [0.25+0.0im    -0.2165+0.375im; -0.2165-0.375im     0.75+0.0im]]

    ρ = almost_tetr
    precision = 4

    dual_value, dual_solution, dual_prior = SDPModels.dual_model(ρ)
    println(round(dual_value, digits = precision), ' ', round.(dual_prior, digits = precision))
    primal_value, primal_solution = SDPModels.primal_model(ρ, dual_prior)
    # println("Primal POVM:\n", [round.(primal_solution[i], digits = precision) for i in 1:size(primal_solution)[1]])
    # double_dual_value, double_dual_solution = SDPModels.double_dual_model(ρ)
    # println("Double dual POVM:\n", [round.(double_dual_solution[i], digits = precision) for i in 1:size(double_dual_solution)[1]])
    println(Distribs.posterior_distr(primal_solution, ρ))
end

function simulator_max_min_max()
    # we checked for [2/5, 1/5, 1/5, 1/5] - TMD gives a thetraedr 
    state_iter1 = [[1/2 0; 0 1/2], [0.5 + 0.0im 0.5 + 0.0im; 0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im -0.25 - 0.433im; -0.25 + 0.433im 0.5 + 0.0im], [0.5 + 0.0im -0.25 + 0.433im; -0.25 - 0.433im 0.5 + 0.0im]] # TMD for uniform - fully mixed plus triangle
    p_iter1 = [2/5, 1/5, 1/5, 1/5] # hardest prior for state_iter1

    state_iter2 = [[1.0+0.0im  0.0; 0.0     0.0], [0.25+0.0im  0.433-0.0im; 0.433+0.0im   0.75+0.0im], [0.25+0.0im    -0.2165-0.375im; -0.2165+0.375im     0.75+0.0im], [0.25+0.0im    -0.2165+0.375im; -0.2165-0.375im     0.75+0.0im]] # TMD for p_iter1
    p_iter2 = [0.222872, 0.259045, 0.259041, 0.259041] # hardest prior for state_iter2

    state_iter3 = [[0.5+0.0im     0.0; 0.0     0.5+0.0im], [1.0+0.0im  0.0; 0.0     0.0], [0.25+0.0im  0.433-0.0im; 0.433+0.0im   0.75+0.0im], [0.25+0.0im  -0.433-0.0im; -0.433+0.0im    0.75+0.0im]]

    ρ = state_iter3
    p = p_iter2
    precision = 6

    dual_value, dual_solution, dual_prior = SDPModels.dual_model(ρ)
    println(round(dual_value, digits = precision), ' ', round.(dual_prior, digits = precision))
    # primal_value, primal_solution = SDPModels.primal_model(ρ, p)
    # println("Primal objective value: ", primal_value)
    # println("Primal POVM:\n", [round.(primal_solution[i], digits = precision) for i in 1:size(primal_solution)[1]])
    # double_dual_value, double_dual_solution = SDPModels.double_dual_model(ρ)
    # println("Double dual POVM:\n", [round.(double_dual_solution[i], digits = precision) for i in 1:size(double_dual_solution)[1]])
    # println(Distribs.posterior_distr(primal_solution, ρ))
end

function simulator_max_min_max_k()
    # we checked for [2/5, 1/5, 1/5, 1/5] - TMD gives a thetraedr 
    state_iter1 = [[1/2 0; 0 1/2], [0.5 + 0.0im 0.5 + 0.0im; 0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im -0.25 - 0.433im; -0.25 + 0.433im 0.5 + 0.0im], [0.5 + 0.0im -0.25 + 0.433im; -0.25 - 0.433im 0.5 + 0.0im]] # TMD for uniform - fully mixed plus triangle
    p_iter1 = [2/5, 1/5, 1/5, 1/5] # hardest prior for state_iter1

    state_iter2 = [[1.0+0.0im  0.0; 0.0     0.0], [0.25+0.0im  0.433-0.0im; 0.433+0.0im   0.75+0.0im], [0.25+0.0im    -0.2165-0.375im; -0.2165+0.375im     0.75+0.0im], [0.25+0.0im    -0.2165+0.375im; -0.2165-0.375im     0.75+0.0im]] # TMD for p_iter1
    p_iter2 = [0.222872, 0.259045, 0.259041, 0.259041] # hardest prior for state_iter2

    state_iter3 = [[0.5+0.0im     0.0; 0.0     0.5+0.0im], [1.0+0.0im  0.0; 0.0     0.0], [0.25+0.0im  0.433-0.0im; 0.433+0.0im   0.75+0.0im], [0.25+0.0im  -0.433-0.0im; -0.433+0.0im    0.75+0.0im]]

    ρ = kron_k_times(state_iter2, 2)
    # println(ρ)

    p = p_iter2
    precision = 6

    dual_value, dual_solution, dual_prior = SDPModels.dual_model(ρ)
    println(round(dual_value, digits = precision), ' ', round.(dual_prior, digits = precision))
    # primal_value, primal_solution = SDPModels.primal_model(ρ, p)
    # println("Primal objective value: ", primal_value)
    # println("Primal POVM:\n", [round.(primal_solution[i], digits = precision) for i in 1:size(primal_solution)[1]])
    # double_dual_value, double_dual_solution = SDPModels.double_dual_model(ρ)
    # println("Double dual POVM:\n", [round.(double_dual_solution[i], digits = precision) for i in 1:size(double_dual_solution)[1]])
    # println(Distribs.posterior_distr(primal_solution, ρ))
end

function simulator_march_09()
    # masha asked to check
    state1 = [[0.7886751345948129 + 0.0im 0.2886751345948129 - 0.28867513459481287im; 0.2886751345948129 + 0.28867513459481287im 0.21132486540518713 + 0.0im],
    [0.21132486540518713 + 0.0im 0.2886751345948129 + 0.28867513459481287im; 0.2886751345948129 - 0.28867513459481287im 0.7886751345948129 + 0.0im],
    [0.21132486540518713 + 0.0im -0.28867513459481287 - 0.2886751345948129im; -0.28867513459481287 + 0.2886751345948129im 0.7886751345948129 + 0.0im],
    [0.7886751345948129 + 0.0im -0.28867513459481287 + 0.2886751345948129im; -0.28867513459481287 - 0.2886751345948129im 0.21132486540518713 + 0.0im]]

    state2 = [[1.0 + 0.0im 0.0 - 0.0im; 0.0 + 0.0im 0.0 + 0.0im],
    [0.1961873907446175 + 0.0im 0.3971119470091982 - 0.0im; 0.3971119470091982 + 0.0im 0.8038126092553826 + 0.0im],
    [0.1961873907446175 + 0.0im -0.1985559735045992 - 0.34390903425626546im; -0.1985559735045992 + 0.34390903425626546im 0.8038126092553826 + 0.0im],
    [0.1961873907446175 + 0.0im -0.1985559735045992 + 0.34390903425626546im; -0.1985559735045992 - 0.34390903425626546im 0.8038126092553826 + 0.0im],
    [0.6371459425887158 + 0.0im 0.4808232423993797 - 0.0im; 0.4808232423993797 + 0.0im 0.3628540574112841 + 0.0im],
    [0.6371459425887158 + 0.0im -0.24041162119968995 - 0.41640514264786577im; -0.24041162119968995 + 0.41640514264786577im 0.3628540574112841 + 0.0im],
    [0.6371459425887158 + 0.0im -0.24041162119968995 + 0.41640514264786577im; -0.24041162119968995 - 0.41640514264786577im 0.3628540574112841 + 0.0im]] 


    ρ = kron_k_times(state2, 3)
    # println(ρ)

    p = p_iter2
    precision = 6

    dual_value, dual_solution, dual_prior = SDPModels.dual_model(ρ)
    println(round(dual_value, digits = precision), ' ', round.(dual_prior, digits = precision))
    primal_value, primal_solution = SDPModels.primal_model(ρ, p)
    # println("Primal objective value: ", primal_value)
    println("Primal POVM:\n", [round.(primal_solution[i], digits = precision) for i in 1:size(primal_solution)[1]])
    # double_dual_value, double_dual_solution = SDPModels.double_dual_model(ρ)
    # println("Double dual POVM:\n", [round.(double_dual_solution[i], digits = precision) for i in 1:size(double_dual_solution)[1]])
    # println(Distribs.posterior_distr(primal_solution, ρ))
end

function simulator_k_copy_poster(d = 2, k = 1)
    # state = [[1, 0; 0, 0], [1/2, 1/2; 1/2, 1/2]]

    zero = zeros(d, d)
    zero[1,1]=1
    state = [LinearAlgebra.I(d)/d, zero]

    ρ = kron_k_times(state, k)
    precision = 4

    priors = Distribs.discrete_prob_sets(length(ρ), 0.001)
    println(priors)
    res = []

    for p in priors
        primal_value, primal_solution = SDPModels.primal_model(ρ, p)
        append!(res, primal_value)
    end
    println(res)

    # dual_value, dual_solution, dual_prior = SDPModels.dual_model(ρ)
    # println(round(dual_value, digits = precision), ' ', round.(dual_prior, digits = precision))
    # primal_value, primal_solution = SDPModels.primal_model(ρ, dual_prior)
    # println("Primal POVM:\n", [round.(primal_solution[i], digits = precision) for i in 1:size(primal_solution)[1]])
    # double_dual_value, double_dual_solution = SDPModels.double_dual_model(ρ)
    # println("Double dual POVM:\n", [round.(double_dual_solution[i], digits = precision) for i in 1:size(double_dual_solution)[1]])
    # println(Distribs.posterior_distr(primal_solution, ρ))
end

function swap(vec, i, j)
    spare = vec[i]
    vec[i] = vec[j]
    vec[j] = spare
    vec
end

function simulator_pbt_states(d, N)
    #studying states that arise in port based teleportation - they have group covariance and permutation invariance, also mixed

    # psi_plus = [1/2 0 0 1/2; 0 0 0 0; 0 0 0 0; 1/2 0 0 1/2]
    # sigma1 = LinearAlgebra.kron(id, psi_plus)
    # sigma2 = Ket.permute_systems(sigma1, [2,1,3], [2,2,2])

    pre_state = zeros(d^2)
    for i in 1:d+1:d^2
        pre_state[i] = 1
    end
    psi_plus = Ket.ketbra(pre_state) / d
    id = LinearAlgebra.I(d)/d

    sigma = psi_plus
    for i in 1:(N-1)
        sigma = LinearAlgebra.kron(id, sigma)
    end

    ρ = [sigma]
    perms = [i for i in 1:(N+1)]
    dims = [d for i in 1:(N+1)]
    for i in (N-1):-1:1
        perms = swap(perms, i, N)
        sigma1 = Ket.permute_systems(sigma, perms, dims)
        perms = swap(perms, i, N)
        append!(ρ, [sigma1])
    end
    ρ = reverse(ρ)
    println("sum o states:\n", sum(ρ)/N)

    # println("check partial trace: ", Ket.partial_trace(ρ[3], [1,2], dims) == psi_plus)

    precision = 4

    dual_value, dual_solution, dual_prior = SDPModels.dual_model(ρ)
    println(round(dual_value, digits = precision), ' ', round.(dual_prior, digits = precision))
    primal_value, primal_solution = SDPModels.primal_model(ρ, [1/N for i in 1:N])
    println("Primal objective value: ", round(primal_value, digits = precision))
    # println("Primal POVM:\n", [round.(primal_solution[i], digits = precision) for i in 1:size(primal_solution)[1]])
    double_dual_value, double_dual_povm, double_dual_q = SDPModels.double_dual_model_with_q(ρ)
    for i in 1:N
        for j in 1:N
            print(round(real(LinearAlgebra.tr(ρ[i] * double_dual_povm[j])), digits = precision), " ")
        end
        print("\n")
    end
    pgm_povm = Measures.PGM_observables(ρ)
    println("PGM POVM:\n", [round.(pgm_povm[i], digits = precision) for i in 1:size(pgm_povm)[1]])
    # println("Double dual POVM:\n", [round.(double_dual_solution[i], digits = precision) for i in 1:size(double_dual_solution)[1]])
    # println(Distribs.posterior_distr(primal_solution, ρ))
end

function simulator_classical_states(d, N=2, k=1)

    zero = zeros(d, d)
    zero[1,1]=1
    one = zeros(d, d)
    one[d,d]=1
    i = LinearAlgebra.I(d)/d
    
    ρ = [zero, one, i]
    precision = 4

    dual_value, dual_solution, dual_prior = SDPModels.dual_model(ρ)
    println(round(dual_value, digits = precision), ' ', round.(dual_prior, digits = precision))


    primal_value, primal_solution = SDPModels.primal_model(ρ, [1/N for i in 1:N])
    println("Primal objective value: ", round(primal_value, digits = precision))

end

function simulator_signed_prior(d, N = 2, k = 1, rank = 0)

    if rank == 0
        rho = [Ket.random_state(d) for i in 1:N]
    else 
        rho = [Ket.random_state(d, rank) for i in 1:N]
    end

    ρ = kron_k_times(rho, k) # sum(rho) = identity * N/d, tr(rho[i]) = 1

    dual_value, dual_solution, dual_prior = SDPModels.dual_model(ρ)
    dual_value_sp, dual_solution_sp, dual_prior_sp = SDPModels.dual_model_signed_prior(ρ)

    probs = round.(dual_prior, digits = 5)
    probs_sp = round.(dual_prior_sp, digits = 5)

    println("With non-neg constraint:    ", probs)
    println("Without non-neg constraint: ", probs_sp)
end

function simulator_zero_is_worse()
    # here i want to verify that if i have a zero in prior probabilities, it necessarily makes it easier to discriminate
    # ρ = [[1 + 0im 0 ; 0 0], [0 0 ; 0 1], [0.5 0.5 ; 0.5 0.5], [0.5 -0.5 ; -0.5 0.5]]
    # ρ = [[1.0 + 0im 0 0; 0 0 0; 0 0 0], [1.0 + 0im 0 0; 0 0 0; 0 0 0], [1.0 + 0im 0 0; 0 0 0; 0 0 0], [0 0 0; 0 0 0; 0 0 1.0 + 0im], [0 0 0; 0 0 0; 0 0 1.0 + 0im], [0 0 0; 0 0 0; 0 0 1.0 + 0im], [0 0 0; 0 1 0; 0 0 0]]
    ρ = [[1.0 + 0im 0 0; 0 0 0; 0 0 0], [1.0 + 0im - 0.1 0 0; 0 0 0; 0 0 0.1]]
    sol_N_minus_1 = SDPModels.dual_model(ρ)
    probs = round.(sol_N_minus_1[3], digits = 4)

    # rho = Ket.random_state(3)
    # rho = [1/2 + 0im 0 0; 0 1/2 + 0im 0; 0 0 0]
    rho = [0 0 0; 0 1. + 0im 0; 0 0 0]
    append!(ρ, [rho])
    append!(probs, [0])
    sol_primal = SDPModels.primal_model(ρ, probs)
    println(sol_primal)

    sol_dual = SDPModels.dual_model(ρ)
    println(sol_dual)
    
end

function purity_different_source(k = 1)
    #we have two states with the same purity but different source of mixedness
    q = 2/7
    # ρ = [[1-q 0 0; 0 q 0; 0 0 0], [(1-q)/2 (1-q)/2 0; (1-q)/2 (1-q)/2 0; 0 0 q]]
    # ρ = [[1-q 0 0; 0 q 0; 0 0 0], [1-q 0 0; 0 q 0; 0 0 0], [(1-q)/2 (1-q)/2 0; (1-q)/2 (1-q)/2 0; 0 0 q]]
    # ρ = [[1-q 0 0; 0 q 0; 0 0 0], [1-q 0 0; 0 q 0; 0 0 0], [(1-q)/2 (1-q)/2 0; (1-q)/2 (1-q)/2 0; 0 0 q], [(1-q)/2 (1-q)/2 0; (1-q)/2 (1-q)/2 0; 0 0 q]]

    #additional pure state (very close to one)
    # ρ = [[1 0 0; 0 0 0; 0 0 0], [1-q 0 0; 0 q 0; 0 0 0], [(1-q)/2 (1-q)/2 0; (1-q)/2 (1-q)/2 0; 0 0 q]]
    # ρ = [[1 0 0; 0 0 0; 0 0 0], [1-q 0 0; 0 q 0; 0 0 0], [1-q 0 0; 0 q 0; 0 0 0], [(1-q)/2 (1-q)/2 0; (1-q)/2 (1-q)/2 0; 0 0 q]]
    # ρ = [[1 0 0; 0 0 0; 0 0 0], [1-q 0 0; 0 q 0; 0 0 0], [1-q 0 0; 0 q 0; 0 0 0], [(1-q)/2 (1-q)/2 0; (1-q)/2 (1-q)/2 0; 0 0 q], [(1-q)/2 (1-q)/2 0; (1-q)/2 (1-q)/2 0; 0 0 q]]

    #additional id/d
    # ρ = [[1-q 0 0; 0 q 0; 0 0 0], [(1-q)/2 (1-q)/2 0; (1-q)/2 (1-q)/2 0; 0 0 q], [1/3 0 0; 0 1/3 0; 0 0 1/3]]
    # ρ = [[1-q 0 0; 0 q 0; 0 0 0], [1-q 0 0; 0 q 0; 0 0 0], [(1-q)/2 (1-q)/2 0; (1-q)/2 (1-q)/2 0; 0 0 q], [1/3 0 0; 0 1/3 0; 0 0 1/3]]
    # ρ = [[1-q 0 0; 0 q 0; 0 0 0], [1-q 0 0; 0 q 0; 0 0 0], [(1-q)/2 (1-q)/2 0; (1-q)/2 (1-q)/2 0; 0 0 q], [(1-q)/2 (1-q)/2 0; (1-q)/2 (1-q)/2 0; 0 0 q], [1/3 0 0; 0 1/3 0; 0 0 1/3]]

    #additional pure and id/d
    # ρ = [[1 0 0; 0 0 0; 0 0 0], [1-q 0 0; 0 q 0; 0 0 0], [(1-q)/2 (1-q)/2 0; (1-q)/2 (1-q)/2 0; 0 0 q], [1/3 0 0; 0 1/3 0; 0 0 1/3]]
    # ρ = [[1 0 0; 0 0 0; 0 0 0], [1-q 0 0; 0 q 0; 0 0 0], [1-q 0 0; 0 q 0; 0 0 0], [(1-q)/2 (1-q)/2 0; (1-q)/2 (1-q)/2 0; 0 0 q], [1/3 0 0; 0 1/3 0; 0 0 1/3]]
    # ρ = [[1 0 0; 0 0 0; 0 0 0], [1-q 0 0; 0 q 0; 0 0 0], [1-q 0 0; 0 q 0; 0 0 0], [(1-q)/2 (1-q)/2 0; (1-q)/2 (1-q)/2 0; 0 0 q], [(1-q)/2 (1-q)/2 0; (1-q)/2 (1-q)/2 0; 0 0 q], [1/3 0 0; 0 1/3 0; 0 0 1/3]]

    #additional pure state (hopefully further)
    # ρ = [[0 0 0; 0 0 0; 0 0 1], [1-q 0 0; 0 q 0; 0 0 0], [(1-q)/2 (1-q)/2 0; (1-q)/2 (1-q)/2 0; 0 0 q]]
    # ρ = [[0 0 0; 0 0 0; 0 0 1], [1-q 0 0; 0 q 0; 0 0 0], [1-q 0 0; 0 q 0; 0 0 0], [(1-q)/2 (1-q)/2 0; (1-q)/2 (1-q)/2 0; 0 0 q]]
    ρ = [[0 0 0; 0 0 0; 0 0 1], [1-q 0 0; 0 q 0; 0 0 0], [1-q 0 0; 0 q 0; 0 0 0], [(1-q)/2 (1-q)/2 0; (1-q)/2 (1-q)/2 0; 0 0 q], [(1-q)/2 (1-q)/2 0; (1-q)/2 (1-q)/2 0; 0 0 q]]

    ρ = kron_k_times(ρ, k) # sum(rho) = identity * N/d, tr(rho[i]) = 1

    precision = 4
    dual_value, dual_solution, dual_prior = SDPModels.dual_model(ρ)
    println("Dual HP for k=", k, ": ", round(dual_value, digits = precision), " & ", round.(dual_prior, digits = precision))
end

function purity_same_source(k = 1)
    # q = 1/2
    # ρ = 
    # Ket.random_state(d)
end

function main()
    d = 2
    N = 4

    #=
        before running check to not overwrite data!
    =#

    for i in 1:100
        # rho, objective_value, objective_solution, objective_prior = simulator_mixed_covariant(d, N)

        # Fio.write_txt("data/special_cases/mixed_covariant", rho, objective_value, objective_solution, objective_prior)
    end
    
    # sum = [0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im]
    # povm = [[0.5 + 0.0im 0.5 + 0.0im; 0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im -0.5 - 0.0im; -0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im 0.5 + 0.0im; 0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im -0.25 - 0.4330127018922193im; -0.25 + 0.4330127018922193im 0.5 + 0.0im], [0.5 + 0.0im -0.25 + 0.4330127018922193im; -0.25 - 0.4330127018922193im 0.5 + 0.0im]]
    # for p in povm
    #     sum += p
    # end
    # println(sum)

    # d = 2
    # N = 4

    # pre_state = zeros(d^2)
    # for i in 1:d+1:d^2
    #     pre_state[i] = 1
    # end
    # psi_plus = Ket.ketbra(pre_state / d)
    # # println("PSI PLUS:\n", psi_plus)
    # id = LinearAlgebra.I(d)/d
    # sigma = psi_plus
    # for i in 1:(N-1)
    #     sigma = LinearAlgebra.kron(id, sigma)
    # end
    # sigma1 = Ket.permute_systems(sigma, perms, dims)

    # println(Ket.partial_trace(sigma1, [1,2], [2,2,2,2]))
end

# simulator_signed_prior(3, 4, 1, 0)

# simulator_zero_is_worse()

# simulator_spherical_design(6)
# simulator_sphere_id()

# for i in range(1, 4)    
#     purity_different_source(i)
# end