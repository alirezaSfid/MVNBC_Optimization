using CSV
using DataFrames
using Random
using DelimitedFiles
using LinearAlgebra
using Statistics
using StatsBase
using Plots

include("ROfeas.jl")

datadir = "Exp_2/Data/"
Usetdir = "Exp_2/Uncertainty_set/mvnbc/"

mvnbc1valsdir = "Exp_2/Solutions/t-x/mvnbc1/"
mvnbc2valsdir = "Exp_2/Solutions/t-x/mvnbc2/"
mvnbc3valsdir = "Exp_2/Solutions/t-x/mvnbc3/"

num_clusers_range = [1, 2, 3]
e_range = 0.6:0.05:0.9

for A in [1, 2, 3]
    for C in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for num_clusters in num_clusers_range
            for e in e_range
                println("A C $A $C")

                fileName = datadir * "train-" * string(A) * "-" * string(C) * ".txt"
                X_train = copy(readdlm(fileName, ',')')

                d, n = size(X_train)

                # solve mvnbc
                # clusters
                Hs = []
                Hs_raw = readdlm(Usetdir * "Hs-" * string(A) * "-" * string(C) * "-" * string(num_clusters) * "-" * string(e) * ".txt", ',')
                K = size(Hs_raw, 1)
                for i in 1:K
                    push!(Hs, reshape(Hs_raw[i, :], (d, d)))
                end

                x_hats = []
                x_hats_raw = readdlm(Usetdir * "x_hats-" * string(A) * "-" * string(C) * "-" * string(num_clusters) * "-" * string(e) * ".txt", ',')
                for i in 1:K
                    push!(x_hats, x_hats_raw[i, :])
                end

                start = time()
                objMvnbc, x_mvnbc = ROFeas(Hs, x_hats, 50*d)
                finish = time()
                t_k = finish - start

                varname = "mvnbc" * string(num_clusters) * "valsdir"
                dir = eval(Meta.parse(varname))
                filename = dir * "vals-" * string(A) * "-" * string(C) * "-" * string(e) * ".txt"

                open(filename, "w") do io
                    write(io, "t, x\n")  # write the header
                    write(io, "$t_k, $x_mvnbc\n")  # write the data
                end
            end
        end
    end
end
