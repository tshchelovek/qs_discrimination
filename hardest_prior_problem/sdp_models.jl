#=
models:
- Julia version: 1.10.5
- Author: ChernyshovaP
- Date: 2025-11-12
=#

using JuMP, MosekTools, IsApprox, Ket
import LinearAlgebra
import Dualization
import SCS
import Mosek

module Distribs
    include("distribs.jl")
end

function kron_k_times(density_matrix, k)
    tensor_product = density_matrix
    for i in 1:k-1
        tensor_product = [LinearAlgebra.kron(tensor_product[i], density_matrix[i]) for i in eachindex(density_matrix)]
    end
    return tensor_product
end

optimizer = SCS.Optimizer

function crooked_model(ρ, E, opt = optimizer)
    d = size(ρ[1])[1]
    N = length(ρ)
    # input states and (optimal) POVM

    model = Model(opt)
    set_silent(model)

    p = [@variable(model) for i in 1:N]
    cs = [@constraint(model, p[i] >= 0) for i in 2:N]
    @constraint(model, sum(p) == 1)

    @objective(
        model,
        Max,
        sum(p[i] * real(LinearAlgebra.tr(ρ[i] * E[i])) for i in 1:N),
    )

    optimize!(model)
    @assert is_solved_and_feasible(model)

    solution = [value.(pp) for pp in p]
    return objective_value(model), solution
end

function primal_model(ρ, p = [1/N for i in 1:N], opt = optimizer)
    d = size(ρ[1])[1]
    N = length(ρ)

    model = Model(opt)
    set_silent(model)

    E = [@variable(model, [1:d, 1:d] in HermitianPSDCone()) for i in 1:N]

    @constraint(model, sum(E) == LinearAlgebra.I)

    @objective(
        model,
        Max,
        sum(p[i] * real(LinearAlgebra.tr(ρ[i] * E[i])) for i in 1:N),
    )

    optimize!(model)
    @assert is_solved_and_feasible(model)

    solution = [value.(e) for e in E]
    return objective_value(model), solution
end

function primal_model_PGM(ρ, opt = optimizer)
    d = size(ρ[1])[1]
    N = length(ρ)

    model = Model(opt)
    set_silent(model)

    E = Ket.pretty_good_measurement(ρ)
    # println(E)

    p = [@variable(model) for i in 1:N]
    cs2 = [@constraint(model, p[i] >= 0) for i in 1:N]
    @constraint(model, sum(p) == 1)

    @objective(
        model,
        Min,
        sum(p[i] * real(LinearAlgebra.tr(ρ[i] * E[i])) for i in 1:N),
    )

    optimize!(model)
    @assert is_solved_and_feasible(model)

    solution = [value.(e) for e in p]
    return objective_value(model), solution
end

function dual_model_with_p(ρ, p, opt = optimizer)
    d = size(ρ[1])[1]
    N = length(ρ)

    model_dual = Model(opt)
    set_silent(model_dual)

    y = @variable(model_dual, [1:d, 1:d] in HermitianPSDCone())

    cs = [@constraint(model_dual, LinearAlgebra.Hermitian(p[i] * ρ[i]) <= y, HermitianPSDCone()) for i in 1:N]

    @objective(
        model_dual, 
        Min, 
        LinearAlgebra.tr(y))

    optimize!(model_dual)
    @assert is_solved_and_feasible(model_dual; dual = true)

    return objective_value(model_dual), value.(y)
end

function dual_model(ρ, opt = optimizer)
    d = size(ρ[1])[1]
    N = length(ρ)

    model_dual = Model(opt)
    set_silent(model_dual)

    y = @variable(model_dual, [1:d, 1:d] in HermitianPSDCone())
    p = [@variable(model_dual) for i in 1:N]

    cs1 = [@constraint(model_dual, LinearAlgebra.Hermitian(p[i] * ρ[i]) <= y, HermitianPSDCone()) for i in 1:N]
    cs2 = [@constraint(model_dual, p[i] >= 0) for i in 1:N]
    @constraint(model_dual, sum(p) == 1)

    @objective(
        model_dual, 
        Min, 
        LinearAlgebra.tr(y))

    optimize!(model_dual)
    @assert is_solved_and_feasible(model_dual; dual = true)

    return objective_value(model_dual), value.(y), value.(p)
end

function dual_model_signed_prior(ρ, opt = optimizer)
    d = size(ρ[1])[1]
    N = length(ρ)

    model_dual = Model(opt)
    set_silent(model_dual)

    y = @variable(model_dual, [1:d, 1:d] in HermitianPSDCone())
    p = [@variable(model_dual) for i in 1:N]

    cs1 = [@constraint(model_dual, LinearAlgebra.Hermitian(p[i] * ρ[i]) <= y, HermitianPSDCone()) for i in 1:N]
    # cs2 = [@constraint(model_dual, p[i] >= 0) for i in 2:N]
    @constraint(model_dual, sum(p) == 1)

    @objective(
        model_dual, 
        Min, 
        LinearAlgebra.tr(y))

    optimize!(model_dual)
    @assert is_solved_and_feasible(model_dual; dual = true)

    return objective_value(model_dual), value.(y), value.(p)
end

function double_dual_model_with_s(ρ, opt = optimizer)
    d = size(ρ[1])[1]
    N = length(ρ)

    model = Model(opt)
    set_silent(model)

    s = @variable(model)
    M = [@variable(model, [1:d, 1:d] in HermitianPSDCone()) for i in 1:N]
    # q = [@variable(model) for i in 1:N]

    @constraint(model, sum(M) == LinearAlgebra.I)
    # cs1 = [@constraint(model, q[i] >= 0) for i in 1:N]
    cs2 = [@constraint(model, real(LinearAlgebra.tr(ρ[i] * M[i])) >= s) for i in 1:N]

    @objective(
        model,
        Max,
        s,
    )

    optimize!(model)
    @assert is_solved_and_feasible(model)

    solution = [value.(m) for m in M]
    # solution_q = [value.(qq) for qq in q]
    return objective_value(model), solution
end

function double_dual_model_with_s_eq(ρ, opt = optimizer)
    d = size(ρ[1])[1]
    N = length(ρ)

    model = Model(opt)
    set_silent(model)

    s = @variable(model)
    M = [@variable(model, [1:d, 1:d] in HermitianPSDCone()) for i in 1:N]
    # q = [@variable(model) for i in 1:N]

    @constraint(model, sum(M) == LinearAlgebra.I)
    # cs1 = [@constraint(model, q[i] >= 0) for i in 1:N]
    cs2 = [@constraint(model, real(LinearAlgebra.tr(ρ[i] * M[i])) == s) for i in 1:N]

    @objective(
        model,
        Max,
        s,
    )

    optimize!(model)
    @assert is_solved_and_feasible(model)

    solution = [value.(m) for m in M]
    # solution_q = [value.(qq) for qq in q]
    return objective_value(model), solution
end

function double_dual_model_with_q(ρ, opt = optimizer)
    d = size(ρ[1])[1]
    N = length(ρ)

    model = Model(opt)
    set_silent(model)

    M = [@variable(model, [1:d, 1:d] in HermitianPSDCone()) for i in 1:N]
    q = [@variable(model) for i in 1:N]

    @constraint(model, sum(M) == LinearAlgebra.I)
    cs1 = [@constraint(model, q[i] >= 0) for i in 1:N]
    cs2 = [@constraint(model, real(LinearAlgebra.tr(ρ[i] * M[i])) - q[i] == real(LinearAlgebra.tr(ρ[1] * M[1])) - q[1]) for i in 2:N]

    @objective(
        model,
        Max,
        real(LinearAlgebra.tr(ρ[1] * M[1])) - q[1],
    )

    optimize!(model)
    @assert is_solved_and_feasible(model)

    solution = [value.(m) for m in M]
    solution_q = [value.(qq) for qq in q]
    return objective_value(model), solution, solution_q
end

function double_dual_model_zero_q(ρ, opt = optimizer)
    d = size(ρ[1])[1]
    N = length(ρ)

    model = Model(opt)
    set_silent(model)

    M = [@variable(model, [1:d, 1:d] in HermitianPSDCone()) for i in 1:N]

    @constraint(model, sum(M) == LinearAlgebra.I)
    cs2 = [@constraint(model, real(LinearAlgebra.tr(ρ[i] * M[i])) == real(LinearAlgebra.tr(ρ[1] * M[1]))) for i in 2:N]

    @objective(
        model,
        Max,
        real(LinearAlgebra.tr(ρ[1] * M[1])),
    )

    optimize!(model)
    @assert is_solved_and_feasible(model)

    solution = [value.(m) for m in M]
    return objective_value(model), solution
end

function double_dual_model(ρ, opt = optimizer)
    d = size(ρ[1])[1]
    N = length(ρ)

    model = Model(opt)
    set_silent(model)

    M = [@variable(model, [1:d, 1:d] in HermitianPSDCone()) for i in 1:N]

    @constraint(model, sum(M) == LinearAlgebra.I)
    cs = [@constraint(model, real(LinearAlgebra.tr(ρ[i] * M[i])) >= real(LinearAlgebra.tr(ρ[1] * M[1]))) for i in 2:N]

    @objective(
        model,
        Max,
        real(LinearAlgebra.tr(ρ[1] * M[1])),
    )

    # print(model)

    optimize!(model)
    @assert is_solved_and_feasible(model)

    solution = [value.(m) for m in M]
    return objective_value(model), solution
end

function sdp_solver(d = 2, N = 2, k = 1)

    # ρ = kron_k_times([Ket.random_state(d) for i in 1:N], k)
    # println([round.(ρ[i], digits = 3) for i in 1:N])
    # println([LinearAlgebra.rank(rho) for rho in ρ])
    # ρ = [[1 0 ; 0 0], [0.5 0.5 ; 0.5 0.5]] # |0> and |+>
    # ρ = [[1 0 ; 0 0], [0 0 ; 0 1]]
    # ρ = [[1 0 ; 0 0], [1/2 0 ; 0 1/2]]
    # ρ = [[1 0 ; 0 0], [0 0 ; 0 1], [0.5 0.5 ; 0.5 0.5]]
    # ρ = [[1 0 ; 0 0], [0.81 0.39 ; 0.39 0.19], [0.64 0.48 ; 0.48 0.36]]
    # ρ = [[0.65831 + 0.0im -0.0 - 0.0im; 0.0 + 0.0im 0.34169 - 0.0im], [0.37922 - 0.0im -0.17691 + 0.42249im; -0.17691 - 0.42249im 0.62078 + 0.0im]]
    # ρ = [[1 0 ; 0 0], [0 0 ; 0 1], [0.5 0.5 ; 0.5 0.5], [0.5 -0.5 ; -0.5 0.5]]
    # ρ = [[1 0 ; 0 0], [0.1464466094067263 0.35355339059327384 ; 0.35355339059327384 0.8535533905932737], [0.1464466094067263 -0.35355339059327384 ; -0.35355339059327384 0.8535533905932737]]

    #??very antisymmetric states??
    # d = 4
    # zero = zeros(d, d)
    # zero[1,1]=1
    # one = zeros(d, d)
    # one[2,2]=1
    # two = zeros(d, d)
    # two[d,d]=1
    # q = 0.8
    # presque_zero = zeros(d, d)
    # presque_zero[1,1] = q
    # presque_zero[3,3] = 1 - q
    # plus = zeros(d, d)
    # plus[1,1] = 1/2
    # plus[1,2] = 1/2
    # plus[2,1] = 1/2
    # plus[2,2] = 1/2
    # middle = (zero + LinearAlgebra.I(d)/d) / 2
    # ρ = [LinearAlgebra.I(d)/d, middle, zero]
    # ρ = [zero, zero, one]
    # ρ = [LinearAlgebra.I(d)/d, zero]
    # N = length(ρ)
    # ket 0 and fully mixed give as close as you want to deterministic
    # ρ = [LinearAlgebra.I(d)/d, zero]
    # ρ = [zero, presque_zero, one, two]
    # N = length(ρ)

    # SAME PURITY LEVEL
    # d = 2
    # N = 2
    # eta = 0.3
    # ρ = [eta * Ket.random_state(d, 1) + (1 - eta) * LinearAlgebra.I(d)/d for i in 1:N]
    # println([LinearAlgebra.tr(rho^2) for rho in ρ])

    #SAME PURITY Jason
    # q = 2/3
    # ρ = [[1-q 0 0; 0 q 0; 0 0 0], [(1-q)/2 (1-q)/2 0; (1-q)/2 (1-q)/2 0; 0 0 q]]

    # p = [0, 1]
    # p = [0.5, 0.5]
    # p = [0.001, 0.999]
    # p = [1 / N for i in 1:N]
    # p = [d/(d+1), 1/(d+1)]

    # println(ρ, '\n', p)
    # sol_primal = primal_model(ρ, p)
    # sol_primal = primal_model_PGM(ρ)
    sol_dual = dual_model(ρ)
    # sol_double_dual = double_dual_model(ρ)
    # println("normal")
    # println("objective value: ", round(sol_double_dual[1], digits = 4))
    # println("tr(rho_i * M_i): ", round.([real(LinearAlgebra.tr(ρ[i] * sol_double_dual[2][i])) for i in 1:N], digits = 4))
    # println("q: ", round.(sol_double_dual[3], digits = 4))
    # probs = round.(sol_double_dual[3], digits = 4)
    # sol_double_dual_no_pi = dual_model_signed_prior(ρ)
    # probs_no_pi = round.(sol_double_dual_no_pi[3], digits = 4)
    # println("With non-neg constraint:    ", probs)
    # println("Without non-neg constraint: ", probs_no_pi)

    # println([real(round.(LinearAlgebra.tr(ρ[i] * sol_double_dual[2][i]), digits = 4)) for i in 1:3])
    

    # croocked model has only one variable - the prior
    # ρ = Matrix{ComplexF64}[[0.087104 + 0.0im -0.241407 + 0.145737im; -0.241407 - 0.145737im 0.912896 + 0.0im], [0.381261 + 0.0im 0.463218 - 0.146049im; 0.463218 + 0.146049im 0.618739 + 0.0im], [0.419018 + 0.0im -0.145266 + 0.471529im; -0.145266 - 0.471529im 0.580982 + 0.0im]]
    # M = Matrix{ComplexF64}[[0.0724 + 0.0im -0.1871 - 0.0842im; -0.1871 + 0.0842im 0.5815 + 0.0im], [0.436 + 0.0im 0.2468 - 0.2074im; 0.2468 + 0.2074im 0.2384 + 0.0im], [0.4917 + 0.0im -0.0597 + 0.2915im; -0.0597 - 0.2915im 0.1801 + 0.0im]]
    # d = 2
    # N = 3
    # sol_crooked = crooked_model(d, N, ρ, M)

    #classical states evenly distributed over the axis
    # d = 2
    # N = 11
    # ρ = [[1.0 0 ; 0 0]]
    # step = 1/(N-1)
    # cur = ρ
    # for i in 1:N-1
    #     cur = cur + [[-step 0 ; 0 step]]
    #     append!(ρ, cur)
    # end
    # sol_dual = dual_model(d, N, ρ)
    # println(maximum(sol_dual[3]) / sol_dual[3][1])
    # println(round.(sol_dual[3], digits = 3))
    # print(sol_dual[3])

    # println("prob_primal = ", sol_primal[1], "\n  prob_dual = ", sol_dual[1])
end


# sdp_solver(2, 3, 1)
