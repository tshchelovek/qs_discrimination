#=
fio:
- Julia version: 1.10.5
- Author: ChernyshovaP
- Date: 2025-12-03
=#

using Parsers
using JSON3

function parsing_json()
    # json_string = """{"a": 1, "b": "hello, world"}"""
    # open("double_dual_3_2.json", "w") do f
    #     JSON3.write(f, json_string)
    #     # JSON3.pretty(f, JSON3.write(x))
    #     println(f)
    # end
end

function writing_txt(name, rho, objective_value, objective_solution = [], prior = [])
    N = size(rho)[1]
    d = size(rho[1])[1]
    file_name = string(name, "_", N, "_", d, ".txt")
    # file_name = string("data/", name, "_", N, "_", d, ".txt")

    try
        open(file_name, "a") do file
            write(file, string(rho), ":")
            write(file, string(objective_value), ":")
            write(file, string(objective_solution), ":")
            write(file, string(prior), "\n")
        end
    catch e
        open(file_name, "w") do file
            write(file, string(rho), ":")
            write(file, string(objective_value), ":")
            write(file, string(objective_solution), ":")
            write(file, string(prior), "\n")
        end
    end
end

function reading_txt(file_name, N = 0, d = 0)
    if N > 0 && d > 0
        file_name = string(file_name, "_", N, "_", d, ".txt")
    end
    try
        open(file_name, "r") do file
            result = []
            for line in eachline(file)
                rho_str, objective_value_str, objective_solution_str, prior_str = rsplit(line, ":")
                tpl = tuple(eval(Meta.parse(rho_str)), 
                            eval(Meta.parse(objective_value_str)), 
                            eval(Meta.parse(objective_solution_str)), 
                            eval(Meta.parse(prior_str)))
                push!(result, tpl)
            end
            return result
        end
    catch e
        println("File ", file_name, " doesn't exist!")
    end
end

function main()
    M = [[0.640000272025707 + 0.0im -0.1600006833360107 + 0.0im; -0.1600006833360107 + 0.0im 0.03999974346694657 + 0.0im], 
         [0.31999975198746244 + 0.0im 0.3200011567343735 + 0.0im; 0.3200011567343735 - 0.0im 0.3200008315665967 + 0.0im], 
         [0.03999997737555185 + 0.0im -0.16000047478529925 + 0.0im; -0.16000047478529925 + 0.0im 0.6399994273708338 + 0.0im]]
    rho1 = [[1 0 ; 0 0], [0.5 0.5 ; 0.5 0.5], [0 0 ; 0 1]]
    rho2 = [[1 0 ; 0 0], [0.5 0.5 ; 0.5 0.5], [1 1 ; 0 1]]
    obj_val = 1
    prior = [1/3, 1/3, 1/3]

    # writing_txt("data/test", rho1, obj_val, M, prior)
    # writing_txt("data/test", rho2, obj_val, M, prior)
    # cases = reading_txt("data/test", 3, 2)
    # rho, objective_value, objective_solution, prior = cases[1]
    # rho

end

main()