#=
max_min_max:
- Julia version: 1.10.5
- Author: ChernyshovaP
- Date: 2026-09-02
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



function simulator_maxminmax_pure(N = 2, k = 1)
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
    precision = 5
    # for i in 1:N
    #     println("purity?: ", round(real(LinearAlgebra.tr(ρ[i]^2)), digits = precision))
    # end

    primal_value, primal_solution = SDPModels.primal_model(ρ, [1/N for i in 1:N])
    println("Primal objective value: ", round(primal_value, digits = precision))
    # println("Primal POVM:\n", [round.(primal_solution[i], digits = precision) for i in 1:size(primal_solution)[1]])

    double_dual_value, double_dual_povm = SDPModels.double_dual_model_with_s_eq(ρ)
    println("Double dual objective value: ", round(double_dual_value, digits = precision))
    # println("Double dual POVM:\n", [round.(double_dual_povm[i], digits = precision) for i in 1:size(double_dual_povm)[1]])

    # pgm_povm = Measures.PGM_observables(ρ)
    # println("PGM POVM:\n", [round.(pgm_povm[i], digits = precision) for i in 1:size(pgm_povm)[1]])
    # println("states:\n", [round.(ρ[i], digits = precision) for i in 1:size(ρ)[1]])
    # println("sum of states: ", sum(ρ))
    # println(Distribs.posterior_distr(double_dual_povm, ρ))
end

function simulator_maxminmax_mixed(N = 2, k = 1)
    d = 2
    sets = [[[1/2 0; 0 1/2]],
            [[1/2 0; 0 1/2], [0.5 + 0.0im 0.5 + 0.0im; 0.5 + 0.0im 0.5 + 0.0im]],
            [[1/2 0; 0 1/2], [0.5 + 0.0im 0.5 + 0.0im; 0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im -0.5 - 0.0im; -0.5 + 0.0im 0.5 + 0.0im]],
            [[1/2 0; 0 1/2], [0.5 + 0.0im 0.5 + 0.0im; 0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im -0.25 - 0.4330127018922193im; -0.25 + 0.4330127018922193im 0.5 + 0.0im], [0.5 + 0.0im -0.25 + 0.4330127018922193im; -0.25 - 0.4330127018922193im 0.5 + 0.0im]],
            [[1/2 0; 0 1/2], [0.7886751345948129 + 0.0im 0.28867513459481287 - 0.28867513459481287im; 0.28867513459481287 + 0.28867513459481287im 0.21132486540518705 + 0.0im], [0.21132486540518716 + 0.0im 0.28867513459481287 + 0.28867513459481287im; 0.28867513459481287 - 0.28867513459481287im 0.7886751345948125 + 0.0im], [0.21132486540518716 + 0.0im -0.28867513459481287 - 0.28867513459481287im; -0.28867513459481287 + 0.28867513459481287im 0.7886751345948125 + 0.0im], [0.7886751345948129 + 0.0im -0.28867513459481287 + 0.28867513459481287im; -0.28867513459481287 - 0.28867513459481287im 0.21132486540518705 + 0.0im]],
            [[1/2 0; 0 1/2], [0.5 + 0.0im 0.5 + 0.0im; 0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im -0.5 - 0.0im; -0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im 0.5 + 0.0im; 0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im -0.25 - 0.4330127018922193im; -0.25 + 0.4330127018922193im 0.5 + 0.0im], [0.5 + 0.0im -0.25 + 0.4330127018922193im; -0.25 - 0.4330127018922193im 0.5 + 0.0im]],
            [[1/2 0; 0 1/2], [0.5 + 0.0im 0.5 + 0.0im; 0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im -0.5 - 0.0im; -0.5 + 0.0im 0.5 + 0.0im], [0.5 + 0.0im 0.0 - 0.5im; 0.0 + 0.5im 0.5 + 0.0im], [0.5 + 0.0im 0.0 + 0.5im; 0.0 - 0.5im 0.5 + 0.0im], [1.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im], [0 0; 0 1]],
            [[1/2 0; 0 1/2], [1.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im], [0.19618739074461747 + 0.0im 0.3971119470091982 + 0.0im; 0.3971119470091982 + 0.0im 0.8038126092553826 + 0.0im], [0.19618739074461747 + 0.0im -0.1985559735045991 - 0.34390903425626546im; -0.1985559735045991 + 0.34390903425626546im 0.8038126092553826 + 0.0im], [0.19618739074461747 + 0.0im -0.1985559735045991 + 0.34390903425626546im; -0.1985559735045991 - 0.34390903425626546im 0.8038126092553826 + 0.0im], [0.6371459425887157 + 0.0im 0.4808232423993796 + 0.0im; 0.4808232423993796 + 0.0im 0.3628540574112841 + 0.0im], [0.6371459425887157 + 0.0im -0.2404116211996898 - 0.4164051426478659im; -0.2404116211996898 + 0.4164051426478659im 0.36285405741128424 + 0.0im], [0.6371459425887157 + 0.0im -0.2404116211996898 + 0.4164051426478659im; -0.2404116211996898 - 0.4164051426478659im 0.36285405741128424 + 0.0im]],
            [[1/2 0; 0 1/2], [0.7886751345948129 + 0.0im 0.28867513459481287 - 0.28867513459481287im; 0.28867513459481287 + 0.28867513459481287im 0.21132486540518705 + 0.0im], [0.21132486540518716 + 0.0im 0.28867513459481287 - 0.28867513459481287im; 0.28867513459481287 + 0.28867513459481287im 0.7886751345948125 + 0.0im], [0.7886751345948129 + 0.0im 0.28867513459481287 + 0.28867513459481287im; 0.28867513459481287 - 0.28867513459481287im 0.21132486540518705 + 0.0im], [0.7886751345948129 + 0.0im -0.28867513459481287 - 0.28867513459481287im; -0.28867513459481287 + 0.28867513459481287im 0.21132486540518705 + 0.0im], [0.21132486540518716 + 0.0im 0.28867513459481287 + 0.28867513459481287im; 0.28867513459481287 - 0.28867513459481287im 0.7886751345948125 + 0.0im], [0.21132486540518716 + 0.0im -0.28867513459481287 - 0.28867513459481287im; -0.28867513459481287 + 0.28867513459481287im 0.7886751345948125 + 0.0im], [0.7886751345948129 + 0.0im -0.28867513459481287 + 0.28867513459481287im; -0.28867513459481287 - 0.28867513459481287im 0.21132486540518705 + 0.0im], [0.21132486540518716 + 0.0im -0.28867513459481287 + 0.28867513459481287im; -0.28867513459481287 - 0.28867513459481287im 0.7886751345948125 + 0.0im]]]

    # ρ = kron_k_times(sets[N] * d / N, k) - sum(rho)=identity
    ρ = kron_k_times(sets[N], k) # sum(rho) = identity * N/d, tr(rho[i]) = 1
    precision = 5

    primal_value, primal_solution = SDPModels.primal_model(ρ, [1/N for i in 1:N])
    println("Primal objective value for uniform: ", round(primal_value, digits = precision))
    # println("Primal POVM:\n", [round.(primal_solution[i], digits = precision) for i in 1:size(primal_solution)[1]])

    coef = 1 / (2^k + N - 1)
    fracs = [coef for i in 1:N]
    fracs[1] = 2^k * coef
    println("Guess HP: ", round.(fracs, digits = precision))
    primal_value, primal_solution = SDPModels.primal_model(ρ, [fracs[i] for i in 1:N])
    println("Guess primal objective value: ", round(primal_value, digits = precision))
    # println("Primal POVM:\n", [round.(primal_solution[i], digits = precision) for i in 1:size(primal_solution)[1]])

    dual_value, dual_solution, dual_prior = SDPModels.dual_model(ρ)
    println("Dual HP: ", round.(dual_prior, digits = precision))

    double_dual_value, double_dual_povm = SDPModels.double_dual_model_with_s_eq(ρ)
    println("Double dual objective value: ", round(double_dual_value, digits = precision))
    # println("Double dual POVM:\n", [round.(double_dual_povm[i], digits = precision) for i in 1:size(double_dual_povm)[1]])

    # pgm_povm = Measures.PGM_observables(ρ)
    # println("PGM POVM:\n", [round.(pgm_povm[i], digits = precision) for i in 1:size(pgm_povm)[1]])
    # println("states:\n", [round.(ρ[i], digits = precision) for i in 1:size(ρ)[1]])
    # println("sum of states: ", sum(ρ))
    # println(Distribs.posterior_distr(double_dual_povm, ρ))
end

function gradient_descent(alpha, beta1, beta2)
    alpha = 0.001       # alpha - stepsize
    beta1 = 0.9         # beta1, beta2 in [0,1) - exponential decay rates for the moment estimates
    beta2 = 0.999
    # f(theta) - stochastic objective function with parameters theta
    # theta0 - initial parameters

    t = 0               # time
    m = 0
    v = 0

    while theta0 > 0.5
        t = t + 1
        m = m
    end
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


# simulator_maxminmax_pure(4, 2)
gradient_descent()
