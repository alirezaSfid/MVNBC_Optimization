import numpy as np
import os

def gen_demand(N, R, seed):
    """
    Generates demand data based on the given parameters.
    """
    np.random.seed(seed)
    sig = np.array([[0.3, -0.1], [-0.1, 0.2]])
    mu = np.array([3, 2.8])
    norms = np.random.multivariate_normal(mu, sig, (N, R))
    d_train = np.exp(norms)
    d_train = np.minimum(d_train, 40)
    return d_train

def gen_demand_clustered(N, R, seed):
    """
    Generates demand data based on the given parameters.
    """
    np.random.seed(seed)
    sig1 = np.array([[0.03, -0.01], [0.01, 0.02]])
    mu1 = np.array([2.6, 3.4])
    norms1 = np.random.multivariate_normal(mu1, sig1, (int(N/2), R))
    d_train1 = np.exp(norms1)
    
    sig2 = np.array([[0.01, 0.05], [0.05, 0.01]])
    mu2 = np.array([3.5, 2.8])
    norms2 = np.random.multivariate_normal(mu2, sig2, (int(N/2), R))
    d_train2 = np.exp(norms2)
    
    d_train = np.concatenate((d_train1, d_train2), axis=0)
    d_train = np.minimum(d_train, 40)
    return d_train

# # Parameters
# N = 100  # Number of data points
# R = 30  # R value
# seed_demand = 2  # Seed for demand_data
# seed_eval = 3  # Seed for demand_data_eval

# # Get the directory where the script is located
# script_dir = os.path.dirname(os.path.abspath(__file__))

# # Generate demand_data
# demand_data = gen_demand(N, R, seed_demand)
# demand_data_path = os.path.join(script_dir, f"demand_data_R={R}.npy")
# np.save(demand_data_path, demand_data)

# # Generate demand_data_eval
# demand_data_eval = gen_demand(N, R, seed_eval)
# demand_data_eval_path = os.path.join(script_dir, f"demand_data_eval_R={R}.npy")
# np.save(demand_data_eval_path, demand_data_eval)

# print(f"Generated datasets for R={R}")

# print("Dataset generation complete. Datasets saved in the script directory.")

N = 100
R = 30
seed_demand = 2
seed_eval = 3

script_dir = os.path.dirname(os.path.abspath(__file__))

demand_data = gen_demand_clustered(N, R, seed_demand)

demand_data_path = os.path.join(script_dir, f"demand_data_clustered_R={R}.npy")
np.save(demand_data_path, demand_data)

demand_data_eval = gen_demand_clustered(N, R, seed_eval)
demand_data_eval_path = os.path.join(script_dir, f"demand_data_eval_clustered_R={R}.npy")
np.save(demand_data_eval_path, demand_data_eval)

print(f"Generated datasets for R={R}")

