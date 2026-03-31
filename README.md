# PMSCP_B-C_Quality

This repository provides the implementation of the research paper:  
**"A Decomposition-Based Optimization Method for Parallel Machine Scheduling under Quality Uncertainty in Wafer Manufacturing"** Published in *Journal of Manufacturing Systems* (2026).  
[Read the paper here (DOI: 10.1016/j.jmsy.2026.02.002)](https://doi.org/10.1016/j.jmsy.2026.02.002)

## Overview

This project addresses a complex scheduling problem originating from a real-world **wafer manufacturing** process. It focuses on the **Parallel Machine Scheduling Problem (PMSP)** with the following characteristics:
* **Sequence-dependent setup times**
* **Quality constraints**
* **Strict deadlines**

The primary objective is to **minimize the total number of setups** required during the production process.

## Methodology

The solution is built on a **Logic-Based Benders Decomposition (LBBD)** approach, specifically utilizing a **Branch-and-Check** framework to efficiently solve the integrated assignment and sequencing problem.

### Key Components
The LBBD logic is modularized into the following files:
* `Assignment.py`: Handles the master problem for job-to-machine assignment.
* `Sequence.py`: Solves the sub-problems to validate and optimize the sequence of jobs on individual machines.
* `framework.py`: Orchestrates the overall Branch-and-Check logic and communication between the master and sub-problems.

## Getting Started

### Prerequisites
Ensure you have the necessary dependencies installed. You can install them via the `requirements.txt` file:
```bash
pip install -r requirements.txt
