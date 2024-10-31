# Genetic Algorithm for Solving TSP

This project contains a Python-based implementation of a Genetic Algorithm to solve the **Traveling Salesman Problem (TSP)**. The entire implementation, including various methods (selection, crossover, and replacement), is in a single file: `GA_TSV_With_All_Methods_Types.py`.

## Features
- **Dynamic Number of Cities**: The algorithm allows for a dynamic input of cities, with no fixed limit on the number of cities you can use.
- **User-Defined Methods**: Users can dynamically select the methods for selection, crossover, and replacement according to their preferences.
- **Visualization**: Uses the `matplotlib` library in Python to display a graphical representation of the shortest path, showing cities, distances, and the optimal route.
- **Selection Techniques**: Includes multiple selection methods (e.g., tournament, roulette wheel).
- **Crossover Options**: Supports several crossover types (e.g., ordered, partially matched).
- **Replacement Mechanisms**: Implements customizable replacement strategies to control evolution effectively.

## Getting Started
1. **Clone the repository**:
    ```bash
    git clone https://github.com/FaresAlhomazah/Genetic-Algorithm-TSV-.git
    cd Genetic-Algorithm-TSV-
    ```
2. **Install dependencies**:
    Install the required libraries from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the algorithm**:
    ```bash
    python GA_TSV_With_All_Methods_Types.py
    ```

## Usage
- **Input Parameters**: The script accepts a dynamic number of cities. You can specify cities and distances, and the algorithm will automatically adjust.
- **Dynamic Method Selection**: Users can choose the selection method, crossover technique, and replacement strategy at runtime, allowing for flexible experimentation.
- **Visualization**: After finding the optimal route, the algorithm will display a plot with city locations and the shortest path.

## Requirements
The `requirements.txt` file includes:
```plaintext
matplotlib
numpy
