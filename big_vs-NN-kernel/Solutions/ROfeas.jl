using LinearAlgebra
using JuMP
using MosekTools

function evaluateSolution(x, scenarios)
    maximumValue = -10000000
    averageValue = 0
    vals = []

    for i in 1:size(scenarios, 2)
        val = dot(x, scenarios[:, i])
        push!(vals, val)
        averageValue += val
        if val > maximumValue
            maximumValue = val
        end
    end

    averageValue /= size(scenarios, 2)

    return averageValue, maximumValue, vals
end

function ROFeas(Hs, x_hats, rhs_feas)

    K = length(Hs)
    d = size(Hs[1], 1)

    model = Model(MosekTools.Optimizer)

    @variable(model, x[1:d])

    for k = 1:K
        @constraint(model, [rhs_feas / sqrt(d) - 1/sqrt(d) * x_hats[k]' * x; inv(cholesky(Hs[k]).L) * x] in SecondOrderCone())
    end

    # @constraint(model, sum(x) == rhs_feas)

    for i = 1:d
        @constraint(model, -1 <= x[i] <= 1)
    end

    @objective(model, Max, sum(x))

    optimize!(model)

    return objective_value(model), value.(x)
end

function ROFeasQuad(Hs, x_hats, rhs_Q, c_bar)

    K = length(Hs)
    d = size(Hs[1], 1)

    Ls = [cholesky(Hs[k]).L for k = 1:K]

    model = Model(MosekTools.Optimizer)

    @variable(model, x[1:d])

    Vs = [ @variable(model, [1:d]) for _ in 1:K ] 
    Ss = [ [ @variable(model, [1:d]) for _ in 1:d ] for _ in 1:K ]
    for k = 1:K

        @constraint(model, [rhs_feas - x_hats[k]' * Vs[k] + sum(Ss[k]); sqrt(d) * inv(Ls[k]) * Vs[k]] in SecondOrderCone())
        @constraint(model, sum(Ss[k]) == Vs[k])
    end

    for i = 1:d
        @constraint(model, 0 <= x[i] <= 1)
    end

    @objective(model, Max, sum(x))

    optimize!(model)

    return objective_value(model), value.(x)
end
