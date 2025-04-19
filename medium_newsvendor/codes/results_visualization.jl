using JSON
using DataFrames
using Statistics
using Glob
using Plots
using StatsPlots
using LaTeXStrings
using CSV

theme(:bright)

# Define the directory containing the JSON files
results_dir = "../results/results200points-AA/eval_results/"

# Define the directory for saving plots
saving_plots_dir = "../results/results200points-AA/eval_plots/"


# Create the directory if it does not exist
if !isdir(saving_plots_dir)
    mkdir(saving_plots_dir)
end

# Create a DataFrame with columns for means and standard deviations
results_df = DataFrame(
    r = Float64[],
    P = Vector{Float64}[],
    R = Int64[],
    TotalCost = Float64[],
    TotalCost_std = Float64[],
    Revenue = Float64[],
    Revenue_std = Float64[],
    NetProfit = Float64[],
    NetProfit_std = Float64[],
    LostSalesCost = Float64[],
    LostSalesCost_std = Float64[],
    OverstockCost = Float64[],
    OverstockCost_std = Float64[],
    InfeasibilityRate = Float64[],
    InfeasibilityRate_std = Float64[],
    UnexpectedLoss = Float64[],
    UnexpectedLoss_std = Float64[],
    elapsed_time = Float64[],
    elapsed_time_std = Float64[],
    # total_volume = Float64[],
    # total_volume_std = Float64[]
)

# Loop over parameter combinations and read CSV files
for P in [[1.0], [2.0], [Inf], [1.0, 2.0], [1.0, Inf], [2.0, Inf]]
    for r in 0.0:0.05:0.5
        # Construct file path based on P
        if length(P) == 1
            file_path = joinpath(results_dir, "results_r=$(r)_P=$(P[1]).csv")
        else
            file_path = joinpath(results_dir, "results_r=$(r)_P=$(P[1])_$(P[2]).csv")
        end

        # Read the CSV file (assuming it has one row with the aggregated stats)
        df = CSV.read(file_path, DataFrame)
        R_val = Int(df.R[1])
        # Retrieve means and stds; adjust column names as needed
        total_cost      = df[!, "Total Cost Mean"][1]
        total_cost_std  = df[!, "Total Cost Std"][1]
        revenue         = df[!, "Avg Revenue Mean"][1]
        revenue_std     = df[!, "Avg Revenue Std"][1]
        net_profit      = df[!, "Avg Net Profit Mean"][1]
        net_profit_std  = df[!, "Avg Net Profit Std"][1]
        lost_sales_cost = df[!, "Avg Lost Sales Cost Mean"][1]
        lost_sales_cost_std = df[!, "Avg Lost Sales Cost Std"][1]
        overstock_cost  = df[!, "Avg Overstock Cost Mean"][1]
        overstock_cost_std  = df[!, "Avg Overstock Cost Std"][1]
        infeasibility_rate = df[!, "Infeasibility Rate Mean"][1]
        infeasibility_rate_std = df[!, "Infeasibility Rate Std"][1]
        unexpected_loss = df[!, "Avg Unexpected Loss Mean"][1]
        unexpected_loss_std = df[!, "Avg Unexpected Loss Std"][1]
        elapsed_time    = df[!, "Avg Elapsed Time Mean"][1]
        elapsed_time_std = df[!, "Avg Elapsed Time Std"][1]
        # total_volume    = df[!, "Total Volume Mean"][1]
        # total_volume_std = df[!, "Total Volume Std"][1]

        push!(results_df, (r, P, R_val,
            total_cost, total_cost_std,
            revenue, revenue_std,
            net_profit, net_profit_std,
            lost_sales_cost, lost_sales_cost_std,
            overstock_cost, overstock_cost_std,
            infeasibility_rate, infeasibility_rate_std,
            unexpected_loss, unexpected_loss_std,
            elapsed_time, elapsed_time_std,
            # total_volume, total_volume_std
            ))
    end
end

# Save the combined results for reference
CSV.write(joinpath(saving_plots_dir, "results_200.csv"), results_df)

# Define marker shapes and default colors
marker_shapes = [:rect, :diamond, :circle, :rect, :star6, :hexagon, :cross, :heptagon, :octagon]
default_colors = [:white, :blue, :red, :gray, :purple, :green, :orange]
labels_P = [L"\mathcal{P} = \{1\}", L"\mathcal{P} = \{2\}", L"\mathcal{P} = \{\infty\}", L"\mathcal{P} = \{1, 2\}", L"\mathcal{P} = \{1, \infty\}", L"\mathcal{P} = \{2, \infty\}"]
# Sort unique P values (first by length then by content)
sorted_P = sort(unique(results_df.P), by = p -> (length(p), p))

plot()
for (i, P) in enumerate(sorted_P)
    df_p = filter(row -> row.P == P, results_df)
    sort!(df_p, :r)
    plot!(df_p.r, df_p.NetProfit, label=labels_P[i], lw=1.5, marker=marker_shapes[i % length(marker_shapes) + 1], markersize=5, color=default_colors[i % length(default_colors) + 1],
    ribbon = df_p.NetProfit_std/sqrt(30),
    fillalpha=0.2)
end
xlabel!(L"r", guidefont=font("Times New Roman", 12))
ylabel!("Avg Net Profit", guidefont=font("Times New Roman", 12))
display(plot!(legend=:bottomright))
savefig(saving_plots_dir * "net_profit_vs_r.pdf")

plot()
for (i, P) in enumerate(sorted_P)
    df_p = filter(row -> row.P == P, results_df)
    sort!(df_p, :r)
    plot!(df_p.r, df_p.elapsed_time/1000,
    label = labels_P[i],  # Correctly format the label
    lw=1.5, marker=marker_shapes[i % length(marker_shapes) + 1], markersize=5, color=default_colors[i % length(default_colors) + 1],
    ribbon=df_p.elapsed_time_std/sqrt(30)/1000, fillalpha=0.2)
end
xlabel!(L"r", guidefont=font("Times New Roman", 12))
ylabel!("Avg Elapsed Time (s)", guidefont=font("Times New Roman", 12))
display(plot!(legend=:topright))
savefig(saving_plots_dir * "elapsed_time_vs_r.pdf")


plot()
for (i, P) in enumerate(sorted_P)
    df_p = filter(row -> row.P == P, results_df)
    sort!(df_p, :r)
    plot!(df_p.r, df_p.OverstockCost, label=labels_P[i], lw=1.5, marker=marker_shapes[i % length(marker_shapes) + 1], markersize=5, color=default_colors[i % length(default_colors) + 1],
    ribbon = df_p.OverstockCost_std/sqrt(30),
    fillalpha=0.2)
end
xlabel!(L"r", guidefont=font("Times New Roman", 12))
ylabel!("Avg Total Overage Cost", guidefont=font("Times New Roman", 12))
display(plot!(legend=:topleft))
savefig(saving_plots_dir * "overstock_cost_vs_r.pdf")

plot()
for (i, P) in enumerate(sorted_P)
    df_p = filter(row -> row.P == P, results_df)
    sort!(df_p, :r)
    plot!(df_p.NetProfit, df_p.OverstockCost, label=labels_P[i], lw=1.5, marker=marker_shapes[i % length(marker_shapes) + 1], markersize=5, color=default_colors[i % length(default_colors) + 1]
    )
end
xlabel!("Avg Net Profit", guidefont=font("Times New Roman", 12))
ylabel!("Total Overage Cost", guidefont=font("Times New Roman", 12))
display(plot!(legend=:topleft))
savefig(saving_plots_dir * "net_profit_vs_overstock_cost.pdf")


plot()
NetProfit = []
TotalCost = []
Revenue = []
for (i, P) in enumerate(sorted_P)
    df_p = filter(row -> row.P == P, results_df)
    sort!(df_p, :P)
    push!(NetProfit, mean(df_p.NetProfit))
    push!(TotalCost, mean(df_p.TotalCost))
    push!(Revenue, mean(df_p.Revenue))
end
groupedbar(hcat(NetProfit, TotalCost, Revenue), bar_position = :dodge, bar_width=0.8, bar_edges=false, bar_spacing=0.5, xticks=(1:6, labels_P), label=["Net Profit" "Total Cost" "Revenue"], guidefont=font("Times New Roman", 12))
# xlabel!(L"\mathcal{P}")
ylabel!("Average Value", guidefont=font("Times New Roman", 12))
display(plot!(legend=:topleft))
savefig(saving_plots_dir * "avg_values_vs_P.pdf")

# plot()
# P_strings = [[1.0], [2.0], [Inf], [1.0, 2.0], [1.0, Inf], [2.0, Inf]]
# sorted_r = sort(unique(results_df.r))
# vol_P = Dict()
# for (i, P) in enumerate(sorted_P)
#     df_p = filter(row -> row.P == P, results_df)
#     sort!(df_p, :r)
#     vol_P[P] = df_p.total_volume
# end
# groupedbar(hcat([vol_P[P] for P in P_strings]...), bar_position = :dodge, bar_width=0.8, bar_edges=false, bar_spacing=0.5, xticks=(1:11, [r for r in sorted_r]), label=[L"\mathcal{P} =\{1\}" L"\mathcal{P} = \{2\}" L"\mathcal{P} = \{\infty\}" L"\mathcal{P} = \{1, 2\}" L"\mathcal{P} = \{1, \infty\}" L"\mathcal{P} = \{2, \infty\}"])
# xlabel!(L"r")
# ylabel!("Avg Total Volume", guidefont=font("Times New Roman", 12))
# display(plot!(legend=:topright))
# savefig(saving_plots_dir * "total_volume_vs_r.pdf")

plot()
elapsed_time = []
for (i, P) in enumerate(sorted_P)
    df_p = filter(row -> row.P == P, results_df)
    sort!(df_p, :P)
    push!(elapsed_time, mean(df_p.elapsed_time)/1000)
end
bar(labels_P, elapsed_time, bar_width=0.8, bar_edges=false, bar_spacing=0.5, label="Elapsed Time", guidefont=font("Times New Roman", 12))
# xlabel!("P")
ylabel!("Average Elapsed Time (s)", guidefont=font("Times New Roman", 12))
display(plot!(legend=:topleft))
savefig(saving_plots_dir * "elapsed_time_vs_P.pdf")


# # Arrays to store means
# NetProfit_means = Float64[]
# TotalCost_means = Float64[]
# Revenue_means = Float64[]
# # Arrays to store SEM
# NetProfit_sem = Float64[]
# TotalCost_sem = Float64[]
# Revenue_sem = Float64[]
# for P_val in sorted_P
#     df_p = filter(row -> row.P == P_val, results_df)
#     # Calculate means
#     netprofit_mean = mean(df_p.NetProfit)
#     totalcost_mean = mean(df_p.TotalCost)
#     revenue_mean   = mean(df_p.Revenue)
#     # Calculate standard deviations
#     netprofit_std = std(df_p.NetProfit)
#     totalcost_std = std(df_p.TotalCost)
#     revenue_std   = std(df_p.Revenue)
#     # Number of samples for this P
#     n = nrow(df_p)
#     # Compute SEM: std / sqrt(n)
#     netprofit_sem_val = netprofit_std / sqrt(n)
#     totalcost_sem_val = totalcost_std / sqrt(n)
#     revenue_sem_val   = revenue_std / sqrt(n)
#     # Push results into arrays
#     push!(NetProfit_means, netprofit_mean)
#     push!(TotalCost_means, totalcost_mean)
#     push!(Revenue_means, revenue_mean)
#     push!(NetProfit_sem, netprofit_sem_val)
#     push!(TotalCost_sem, totalcost_sem_val)
#     push!(Revenue_sem, revenue_sem_val)
# end
# # Build the matrices for groupedbar
# # - The first argument is a matrix with your means.
# # - yerr should be a matrix with the same dimensions as the means matrix.
# data_means = hcat(NetProfit_means, TotalCost_means, Revenue_means)
# data_sem   = hcat(NetProfit_sem, TotalCost_sem, Revenue_sem)
# labels = ["Net Profit" "Total Cost" "Revenue"]
# x_labels = ["P = $p" for p in sorted_P]
# # Create the grouped bar plot, specifying error bars
# p = groupedbar(
#     data_means,
#     bar_position = :dodge,
#     bar_width = 0.8,
#     bar_edges = false,
#     bar_spacing = 0.5,
#     yerr = data_sem,               # <--- error bars
#     xticks = (1:length(sorted_P), x_labels),
#     label = labels
# )
# xlabel!(p, "P")
# ylabel!(p, "Average Value")
# plot!(p, legend=:topleft)
