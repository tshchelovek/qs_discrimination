#=
fixed_prior:
- Julia version: 1.10.5
- Author: ChernyshovaP
- Date: 2025-12-08
=#

module SDPModels
    include("sdp_models.jl")
end
module Measures
    include("measures.jl")
end
module Fio
    include("fio.jl")
end


function simulator(filename, d, N, p = [1/N for i in 1:N])
    filename_read = string("data/double_dual/", filename)
    filename_write = string("data/primal/", filename)

    cases = Fio.read_txt(filename_read, d, N)

    for i in 1:size(cases)[1]
        rho, objective_value, objective_solution, prior = cases[i]
        obj_value, obj_solution = SDPModels.primal_model(d, N, rho, p)
        Fio.write_txt(filename_write, rho, obj_value, obj_solution, p)
    end

    return
end

function main()
    d = 2
    N = 3
    
    # simulator("mixed", d, N)
    simulator("pure", d, N)

    # Fio.analyze_value_txt("data/double_dual/pure", N, d) / Fio.analyze_value_txt("data/primal/pure", N, d)
end

main()