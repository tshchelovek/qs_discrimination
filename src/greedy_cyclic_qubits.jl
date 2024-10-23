using JuMP
import LinearAlgebra
import SCS

function qubit_bloch_sphere(theta, phi)
    qubit = [cos(theta / 2), sin(theta / 2) * exp(im * phi)]
    return [qubit * qubit', theta, phi]
end

function iterate_over_disc(start = 0, stop = 1, step = 0.1, phi = 0)
    start += step
    result = []
    for idx in start:step:stop
        result = [result; [qubit_bloch_sphere(idx * pi, phi)]]
        result = [result; [qubit_bloch_sphere(idx * pi, phi + pi)]]
    end
    return result
end

function solve_sdp(set_of_states)
    N, d = length(set_of_states), size(set_of_states[1], 1)
    model = Model(SCS.Optimizer)
    set_silent(model)

    E = [@variable(model, [1:d, 1:d] in HermitianPSDCone()) for i in 1:N]

    @constraint(model, sum(E) == LinearAlgebra.I)

    @objective(
        model,
        Max,
        sum(real(LinearAlgebra.tr(set_of_states[i] * E[i])) for i in 1:N) / N,
    )

    optimize!(model)
    @assert is_solved_and_feasible(model)
    solution_summary(model)

    prob_guess = objective_value(model)
    solution = [value.(e) for e in E]

    return [prob_guess, solution]
end

function sqrt_measurement(set_of_states)
    average = 1 / length(set_of_states) * sum(set_of_states)
    println(average)
    eigen_result = LinearAlgebra.eigen(average)
    diag = LinearAlgebra.Diagonal(eigen_result.values)
    println(diag)
    println(sqrt(diag))
    println(LinearAlgebra.pinv(sqrt(diag)))
    println(LinearAlgebra.pinv(sqrt(average)))

    POVM = []
    for state in set_of_states
        push!(POVM, LinearAlgebra.pinv(sqrt(diag)) * state * LinearAlgebra.pinv(sqrt(diag)))
    end
    return POVM
end

function greedy_cyclic_qubits(start = 0, stop = 1, step = 0.1)
    best_prob = 0
    best_triplet = []
    best_povm = []
    states = iterate_over_disc(start, stop, step)
    state1 = [1 0 ; 0 0]
    for i in 1 : 1
        state2 = states[i]
        for j in i + 2 : 3
            state3 = states[j]

            state2 = [0.5 0.5 ; 0.5 0.5]
            state3 = [0 0 ; 0 1]

            println("1: ", state1)
            println("2: ", state2)
            println("3: ", state3)

            sqrt_POVM = sqrt_measurement([state1, state2, state3])
            println("SQRT: ")
            for M in sqrt_POVM
                println(round.(M, digits=5))
            end

            result = solve_sdp([state1, state2, state3])
            println("SDP:")
            for M in result[2]
                println(round.(M, digits=5))
            end

            # println("Result for ", values[i], " and ", values[j], ":")
            # println("Probability of success: ", result[1])
            # println("Solution:\n", result[2], "\n")
#             if result[1] > best_prob
#                 best_prob = result[1]
#                 best_triplet = [state0, state1, state2]
#                 best_povm = result[2]
#             end
        end
    end
#     println("Best probability: ", best_prob)
#     println("Best triplet:\n", best_triplet, "\n")
#     println("Best POVM:\n", best_povm, "\n")
    return [best_prob, best_povm]
end

# result = solve_sdp([[1 0 ; 0 0], [0.5 0.5 ; 0.5 0.5]])
result = greedy_cyclic_qubits()
# println(result)
