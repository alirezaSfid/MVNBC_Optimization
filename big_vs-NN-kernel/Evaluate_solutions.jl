using DelimitedFiles
using LinearAlgebra
using Statistics
using Plots
using DataFrames
using XLSX


function evalSolution(x, x_test, RHS_feas)
    maximumValue = -10000000
    averageValue = 0
    vals = []

    for i in eachindex(x_test)
        val = dot(x, x_test[i])
        push!(vals, val)
        averageValue += val
        if val > maximumValue
            maximumValue = val
        end
    end
    averageValue /= size(x_test, 1)
    percentiles = [quantile(vals, p) for p in [0.5, 0.9]]

    feass = [1 - sum(vals .> p * RHS_feas) / size(x_test, 1) for p in [0.9, 0.95, 1]]

    return percentiles, averageValue, maximumValue, vals, sum(x), feass
end


mvnbc1valsdir = "Exp_2/Solutions/t-x/mvnbc1/"
mvnbc2valsdir = "Exp_2/Solutions/t-x/mvnbc2/"
mvnbc3valsdir = "Exp_2/Solutions/t-x/mvnbc3/"
kernelvalsdir = "Exp_2/Solutions/t-x/kernel/"
nnvalsdir = "Exp_2/Solutions/t-x/nn/"

methods = [
    "mvnbc1",
    "mvnbc2",
    "mvnbc3",
    "kernel",
    "nn"
]


# xf = "/Users/alireza_1/Library/Mobile Documents/com~apple~CloudDocs/PhD/Coding/Julia-Mac/RO-MV-master/20d/results.xlsx"

# Create a new Excel workbook
xf = XLSX.openxlsx("output.xlsx", mode="w")


for A in [1, 2, 3]
    for C in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        sheet = XLSX.addsheet!(xf, string(A) * "-" * string(C))

        # Initialize the row number
        row_number = 1

        # Loading test data
        filename = "/Users/alireza_1/surfdrive/coding/Julia-Mac/RO-MV-master/20d/data/test-" * string(A) * "-" * string(C) * ".txt"
        # Read the file line by line
        lines = readlines(filename)
        x_test = Vector{Float64}[]
        # For each line in the file
        for line in lines
            # Split the line into a vector of strings
            vec_str = split(line, ',')
            # Convert the vector of strings to a vector of floats
            vec_float = parse.(Float64, vec_str)
            # Append the vector of floats to the vector of vectors
            push!(x_test, vec_float)
        end


        for e in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.90]

            row_number += 1

            # Write the e value to the row above the table
            XLSX.setdata!(sheet, XLSX.CellRef("A" * string(row_number)), "e")
            XLSX.setdata!(sheet, XLSX.CellRef("B" * string(row_number)), e)
            # Increment the row number
            row_number += 1


            dict = Dict()
            df = DataFrame()

            #Read values for mvnbcs
            for num_clusters in [1, 2, 3]
                varname = "mvnbc" * string(num_clusters) * "valsdir"
                dir = eval(Meta.parse(varname))
                filename = dir * "vals-" * string(A) * "-" * string(C) * "-" * string(e) * ".txt"
                data = readdlm(filename, ',', skipstart=1)
                t = data[1, 1]
                x = data[1, 2:end]
                x[1] = parse(Float64, strip(x[1], [' ', '[']))
                x[end] = parse(Float64, strip(x[end], [' ', ']']))
                x = Float64.(x)
                percentiles, averageValue, maximumValue, vals, sumx, feass = evalSolution(x, x_test, 1000)
                row = DataFrame(
                    method = "mvnbc" * string(num_clusters),
                    t = t,
                    percentile_50 = percentiles[1],
                    percentile_90 = percentiles[2],
                    averageValue = averageValue,
                    maximumValue = maximumValue,
                    sumx = sumx,
                    feas_90 = feass[1],
                    feas_95 = feass[2],
                    feas_100 = feass[3]
                )
                append!(df, row)
                # dict["mvnbc" * string(num_clusters)] = (x = x, t = t, percentiles = percentiles, averageValue = averageValue, maximumValue = maximumValue, sumx = sumx, feass = feass)
            end

            #Read values for kernels
            filename = kernelvalsdir * "vals-" * string(A) * "-" * string(C) * "-" * string(e) * ".txt"
            data = readdlm(filename, ',', skipstart=1)
            t = data[1, 1]
            x = data[1, 2:end]
            x = split(strip(x[1], ['[', ']']))
            x_kernel = parse.(Float64, x)
            percentiles, averageValue, maximumValue, vals, sumx, feass = evalSolution(x_kernel, x_test, 1000)
            row = DataFrame(
                method = "kernel",
                t = t,
                percentile_50 = percentiles[1],
                percentile_90 = percentiles[2],
                averageValue = averageValue,
                maximumValue = maximumValue,
                sumx = sumx,
                feas_90 = feass[1],
                feas_95 = feass[2],
                feas_100 = feass[3]
            )
            append!(df, row)
            # dict["kernel"] = (x = x_kernel, t = t, percentiles = percentiles, averageValue = averageValue, maximumValue = maximumValue, sumx = sumx, feass = feass)

            #Read values for nn
            filename = nnvalsdir * "vals-" * string(A) * "-" * string(C) * "-" * string(e) * ".txt"
            data = readdlm(filename, ',', skipstart=1)
            t = data[1, 1]
            x = data[1, 2:end]
            x = split(strip(x[1], ['[', ']']))
            x_nn = parse.(Float64, x)
            percentiles, averageValue, maximumValue, vals, sumx, feass = evalSolution(x_nn, x_test, 1000)
            row = DataFrame(
                method = "nn",
                t = t,
                percentile_50 = percentiles[1],
                percentile_90 = percentiles[2],
                averageValue = averageValue,
                maximumValue = maximumValue,
                sumx = sumx,
                feas_90 = feass[1],
                feas_95 = feass[2],
                feas_100 = feass[3]
            )
            append!(df, row)

            # Write the DataFrame to the Excel sheet
            XLSX.writetable!(sheet, collect(DataFrames.eachcol(df)), DataFrames.names(df), anchor_cell=XLSX.CellRef("A" * string(row_number)))


            # Increment the row number by the number of rows in the table plus 2
            row_number += nrow(df) + 2
        end
    end
end

XLSX.writexlsx("output.xlsx", xf)

XLSX.closexlsx(xf)



e = 0.95
XLSX.openxlsx("output.xlsx", mode="rw") do xf

    for A in [1, 2, 3]
        for C in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            df = DataFrame()
            # Loading test data
            filename = "/Users/alireza_1/surfdrive/coding/Julia-Mac/RO-MV-master/20d/data/test-" * string(A) * "-" * string(C) * ".txt"
            # Read the file line by line
            lines = readlines(filename)
            x_test = Vector{Float64}[]
            # For each line in the file
            for line in lines
                # Split the line into a vector of strings
                vec_str = split(line, ',')
                # Convert the vector of strings to a vector of floats
                vec_float = parse.(Float64, vec_str)
                # Append the vector of floats to the vector of vectors
                push!(x_test, vec_float)
            end
            
            #Read values for mvnbcs
            for num_clusters in [1, 2, 3]
                varname = "mvnbc" * string(num_clusters) * "valsdir"
                dir = eval(Meta.parse(varname))
                filename = dir * "vals-" * string(A) * "-" * string(C) * "-" * string(e) * ".txt"
                data = readdlm(filename, ',', skipstart=1)
                t = data[1, 1]
                x = data[1, 2:end]
                x[1] = parse(Float64, strip(x[1], [' ', '[']))
                x[end] = parse(Float64, strip(x[end], [' ', ']']))
                x = Float64.(x)
                percentiles, averageValue, maximumValue, vals, sumx, feass = evalSolution(x, x_test, 1000)
                row = DataFrame(
                    method = "mvnbc" * string(num_clusters),
                    t = t,
                    percentile_50 = percentiles[1],
                    percentile_90 = percentiles[2],
                    averageValue = averageValue,
                    maximumValue = maximumValue,
                    sumx = sumx,
                    feas_90 = feass[1],
                    feas_95 = feass[2],
                    feas_100 = feass[3]
                )
                append!(df, row)
                # dict["mvnbc" * string(num_clusters)] = (x = x, t = t, percentiles = percentiles, averageValue = averageValue, maximumValue = maximumValue, sumx = sumx, feass = feass)
            end
            # Write the DataFrame to an Excel file
            XLSX.writetable("output.xlsx", collect(DataFrames.eachcol(df)), DataFrames.names(df), overwrite=true, sheetname=string(A) * "-" * string(C))

        end
    end
end




# Write the train times in the Excel file
mvnbcdir = "/Users/alireza_1/surfdrive/coding/Julia-Mac/RO-MV-master/20d/mvnbc/"
XLSX.openxlsx("output.xlsx", mode="rw") do xf
    for A in [1, 2, 3]
        for C in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            sheet = xf[string(A) * "-" * string(C)]
            for (eidx, e) in enumerate([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.90])
                t_trains = []
                rownumber = eidx * 9 - 6
                for num_cluster in [1, 2, 3]
                    filename = mvnbcdir * "time_elapsed-" * string(A) * "-" * string(C) * "-" * string(num_cluster) * "-" * string(e) * ".txt"
                    push!(t_trains, readdlm(filename)[1])
                end
                sheet["K" * string(rownumber)] = "t_train"
                sheet["K" * string(rownumber+1), dim=1] = t_trains
            end
        end
    end
end


