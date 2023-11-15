# Modeling-2B: Fake News Spread in Graphs

## Overview

This project models the spread of fake news in a graph using Python. The simulation is designed to analyze how fake news propagates through a network, offering insights into the dynamics of misinformation dissemination.

## Installation

To run the project, follow these steps:

### 1. Create a Virtual Environment

Use the following command to create a virtual environment named "venv" within your project folder:

```bash
python -m venv venv
```

### 2. Activate the Virtual Environment

Activate the virtual environment. Use one of the following commands based on your operating system:

- On Windows:

  ```bash
  venv\Scripts\activate
  ```

- On Unix or MacOS:

  ```bash
  source venv/bin/activate
  ```

Your terminal prompt should now display `(venv)`.

### 3. Install Dependencies

Install the required packages specified in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

**note:** Remember to update your virtual environment whenever you add or remove dependencies. You can do this by running `pip freeze > requirements.txt` to update the `requirements.txt` file

### 4. Explore the Graph Code

Refer to the `simple_graph.py` file to understand how to create a graph. The underlying package (`networkx`) is an interface for Python for C/C++ code, providing increased performance for graph operations.
