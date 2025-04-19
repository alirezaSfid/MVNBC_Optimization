# Minimum-Volume Norm-Based Clustering (MVNBC) Optimization

This repository contains the implementation and experimental codes associated with the paper *"A Clustering-Based Uncertainty Set for Robust Optimization"*.

## Repository Structure

- **`src/`**: Main source code to construct uncertainty sets for small and medium size datsets:
  - **`BendersDecomposition.jl`**: Code to solve MVNBC exactly using GBD
  - **`HeuristicMethod.jl`**: Code to solve MVNBC approximately using heuristic method proposed in the paper
  - **`multi_dimensional_data_generation.jl/`**: Code to generate synthetic data.
  - **`LocalSearchFunctions.jl/`** and **`LocalSearchFunctions.jl/`**: Codes which are used in `BendersDecomposition.jl` and `HeuristicMethod.jl`.

- **`clustering_exp/`**: Codes, scripts, and results for clustering experiment performed in the paper.

- **`small_newsvendor/`**: Data, codes, scripts, and results for 2-product newsvendor problem with 50 data points performed in the paper. This experiments are done for exact solution method (GBD), and approximation algorithm (AA).

- **`medium_newsvendor/`**: Data, codes, scripts, and results for 2-product newsvendor problem with 200 data points performed in the paper. This experiments are done approximation algorithm (AA).

- **`Uset_construction/`**: Main source code to construct uncertainty sets for big size datsets.

- **`big_vs-NN-kernel/`**: Data, uncertainty sets parameters, and solutions for the big-size experiment in the paper.

## Cite this Work

Please cite our paper if you use this code or results in your research:

```bibtex
@article{YourCitation,
    author = {Alireza Yazdani, Ahmadreza Marandi, Rob Basten, Lijia Tan},
    title = {A Clustering-Based Uncertainty Set for Robust Optimization},
    journal = {Journal Name},
    year = {2025},
    volume = {X},
    number = {Y},
    pages = {Z-ZZ},
}