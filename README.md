# Simulation-Supported Bayesian Network for Bridge Networks

This repository contains Python and R code implementing a simulation-supported Bayesian network approach tailored for analyzing selected bridge networks. The methodology integrates simulation techniques with Bayesian networks to assess and predict the performance and reliability of bridge infrastructures.

## Overview

Bridge networks are critical components of transportation infrastructure, necessitating robust methods for their analysis and maintenance. This project employs a Bayesian network framework enhanced by simulation to model the complex interdependencies and failure mechanisms within bridge systems. By leveraging both Python and R, the approach facilitates comprehensive probabilistic analysis and decision support for bridge management.

## Features

- **Simulation Integration**: Combines simulation data with Bayesian network modeling to enhance predictive accuracy.
- **Probabilistic Analysis**: Evaluates the likelihood of various failure modes and their potential impacts on the bridge network.
- **Cross-Platform Implementation**: Utilizes both Python and R to provide flexibility and leverage the strengths of each programming environment.

## Repository Structure

- `Python/`: Contains Python scripts and modules for simulation and Bayesian network modeling.
- `R/`: Includes R scripts for data analysis, visualization, and additional statistical modeling.
- `Data/`: Sample datasets used for simulations and model validation.
- `Docs/`: Documentation and resources detailing the methodology and usage instructions.

## Getting Started

### Prerequisites

Ensure that you have the following software installed:

- Python 3.x
- R (version 4.0 or higher)

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/MJ2417/SimulationSupportedBayesianNetwork.git
   cd SimulationSupportedBayesianNetwork
   ```

2. **Set up Python environment**:

   - Create a virtual environment:

     ```bash
     python -m venv env
     source env/bin/activate  # On Windows: env\Scripts\activate
     ```

   - Install required Python packages:

     ```bash
     pip install -r requirements.txt
     ```

3. **Set up R environment**:

   - Install required R packages by running the following in your R console:

     ```R
     source('R/install_packages.R')
     ```

## Usage

1. **Data Preparation**: Place your bridge network data in the `Data/` directory. Ensure the data is formatted as specified in the documentation.

2. **Running Simulations**:

   - **Python**: Execute the main simulation script:

     ```bash
     python Python/main_simulation.py
     ```

   - **R**: Run the R analysis script:

     ```R
     source('R/main_analysis.R')
     ```

3. **Results**: Output files and visualizations will be saved in the `Results/` directory.

## Documentation

Comprehensive documentation is available in the `Docs/` directory, including:

- Methodology overview
- Data formatting guidelines
- Detailed usage instructions
- Interpretation of results


## Acknowledgments

We acknowledge the use of open-source libraries and tools that have facilitated this project. Special thanks to Joost Berkhout (https://research.vu.nl/en/persons/joost-berkhout) for providing a couple of MC related functions and also KDA algorithm.

---

For any questions or support, please open an issue in this repository. 
