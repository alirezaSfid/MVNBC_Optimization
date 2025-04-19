
using JSON
using NPZ
using Statistics
using CSV
using DataFrames

function evaluate_solution(X_test, x, obj, c, h)
    num_products, num_samples = size(X_test)
    
    # Initialize accumulators for total measures
    total_cost = 0
    total_revenue = 0
    total_lost_sales_cost = 0
    total_overstock_cost = 0
    total_net_profit = 0
    infeasible_points = 0
    unexpected_loss = 0

    # Store results per product
    per_product_metrics = Dict(
        "Total Cost" => zeros(num_products),
        "Revenue" => zeros(num_products),
        "Net Profit" => zeros(num_products),
        "Lost Sales Cost" => zeros(num_products),
        "Overstock Cost" => zeros(num_products)
    )

    for i in 1:num_samples
        # Get demand for this sample
        demand = X_test[:, i]

        # Compute per-product metrics
        ordering_cost = h .* x  # Total ordering cost
        satisfied_demand = min.(demand, x)  # Sold quantity per product
        revenue = c .* satisfied_demand
        lost_sales_cost = c .* max.(0, demand .- x)  # Missed sales cost
        overstock_cost = h .* max.(0, x .- demand)  # Cost of excess inventory
        net_profit = revenue - ordering_cost  # Profit after costs

        # Aggregate totals
        total_cost += sum(ordering_cost)
        total_revenue += sum(revenue)
        total_lost_sales_cost += sum(lost_sales_cost)
        total_overstock_cost += sum(overstock_cost)
        total_net_profit += sum(net_profit)

        if total_net_profit < - obj
            infeasible_points += 1
            unexpected_loss += -obj -total_net_profit
        end

        # Store per-product values
        per_product_metrics["Total Cost"] += ordering_cost
        per_product_metrics["Revenue"] += revenue
        per_product_metrics["Net Profit"] += net_profit
        per_product_metrics["Lost Sales Cost"] += lost_sales_cost
        per_product_metrics["Overstock Cost"] += overstock_cost
    end

    # Compute averages per sample
    avg_total_cost = total_cost / num_samples
    avg_revenue = total_revenue / num_samples
    avg_net_profit = total_net_profit / num_samples
    avg_lost_sales_cost = total_lost_sales_cost / num_samples
    avg_overstock_cost = total_overstock_cost / num_samples
    infeasibility_rate = infeasible_points / num_samples
    avg_unexpected_loss = unexpected_loss / num_samples

    return Dict(
        "Total Cost" => avg_total_cost,
        "Avg Revenue" => avg_revenue,
        "Avg Net Profit" => avg_net_profit,
        "Avg Lost Sales Cost" => avg_lost_sales_cost,
        "Avg Overstock Cost" => avg_overstock_cost,
        "Infeasibility Rate" => infeasibility_rate,
        "Per Product Metrics" => per_product_metrics,
        "Avg Unexpected Loss" => avg_unexpected_loss
    )
end

# Define the file path
demand_data_eval_path = "../data/demand_data_eval_clustered_R=30.npy"

dir_path = "../results/results200points-AA/json/"

eval_results_path = "../results/results200points-AA/eval_results/"


if !isdir(eval_results_path)
    mkdir(eval_results_path)
end

Data_eval_full = npzread(demand_data_eval_path)
h = [4.0, 5.0]    # Purchasing cost of the product
c = [5.0, 6.5]    # Selling price of the product

for P in [[1.0], [2.0], [Inf], [1.0, 2.0], [1.0, Inf], [2.0, Inf]]
    for r in 0.0:0.05:0.5
        # Initialize arrays to store per-dataset metrics
        total_cost_array = Float64[]
        revenue_array = Float64[]
        net_profit_array = Float64[]
        lost_sales_cost_array = Float64[]
        overstock_cost_array = Float64[]
        infeasibility_rate_array = Float64[]
        unexpected_loss_array = Float64[]
        elapsed_time_array = Float64[]
        tot_vol_array = Float64[]
        
        R_actual = 0  # Count of valid evaluations

        for R in 1:30
            # Construct file path based on P parameters
            if length(P) == 1
                file_path = joinpath(dir_path, "results_R=$(R)_r=$(r)_P=$(P[1]).json")
            else
                file_path = joinpath(dir_path, "results_R=$(R)_r=$(r)_P=$(P[1])_$(P[2]).json")
            end
            
            x = []
            objective_value = 0
            elapsed_time = 0
            tot_vol = 0
            
            # Read the JSON file
            try
                json_data = JSON.parsefile(file_path)
                R_actual += 1
                objective_value = json_data["objective_value"]
                x = json_data["optimal_solutions"]
                elapsed_time = json_data["elapsed_time"]["value"]
                tot_vol = json_data["total_vol"]
            catch
                println("Error reading file for r=$(r), P=$(P) and R=$(R)")
                continue
            end
            
            # Extract the test data for this dataset
            X_test = Data_eval_full[1:100, R, :]'  # Note the transpose
            
            eval_results = evaluate_solution(X_test, x, objective_value, c, h)
            
            # Save the metrics from this evaluation
            push!(total_cost_array, eval_results["Total Cost"])
            push!(revenue_array, eval_results["Avg Revenue"])
            push!(net_profit_array, eval_results["Avg Net Profit"])
            push!(lost_sales_cost_array, eval_results["Avg Lost Sales Cost"])
            push!(overstock_cost_array, eval_results["Avg Overstock Cost"])
            push!(infeasibility_rate_array, eval_results["Infeasibility Rate"])
            push!(unexpected_loss_array, eval_results["Avg Unexpected Loss"])
            push!(elapsed_time_array, elapsed_time)
            push!(tot_vol_array, tot_vol)
        end

        if R_actual > 0
            # Compute means and standard deviations for each metric
            results = Dict(
                "Total Cost Mean" => mean(total_cost_array),
                "Total Cost Std" => std(total_cost_array),
                "Avg Revenue Mean" => mean(revenue_array),
                "Avg Revenue Std" => std(revenue_array),
                "Avg Net Profit Mean" => mean(net_profit_array),
                "Avg Net Profit Std" => std(net_profit_array),
                "Avg Lost Sales Cost Mean" => mean(lost_sales_cost_array),
                "Avg Lost Sales Cost Std" => std(lost_sales_cost_array),
                "Avg Overstock Cost Mean" => mean(overstock_cost_array),
                "Avg Overstock Cost Std" => std(overstock_cost_array),
                "Infeasibility Rate Mean" => mean(infeasibility_rate_array),
                "Infeasibility Rate Std" => std(infeasibility_rate_array),
                "Avg Unexpected Loss Mean" => mean(unexpected_loss_array),
                "Avg Unexpected Loss Std" => std(unexpected_loss_array),
                "R" => R_actual,
                "Avg Elapsed Time Mean" => mean(elapsed_time_array),
                "Avg Elapsed Time Std" => std(elapsed_time_array),
                "Total Volume Mean" => mean(tot_vol_array),
                "Total Volume Std" => std(tot_vol_array)
            )
            df = DataFrame(results)

            if length(P) == 1
                out_file = joinpath(eval_results_path, "results_r=$(r)_P=$(P[1]).csv")
            else
                out_file = joinpath(eval_results_path, "results_r=$(r)_P=$(P[1])_$(P[2]).csv")
            end

            CSV.write(out_file, df)
        else
            println("No valid evaluations for r=$(r) and P=$(P)")
        end
    end
end