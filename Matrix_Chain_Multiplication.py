import streamlit as st
import numpy as np
import pandas as pd

def matrix_chain_order(dims):
    """Computes the optimal order of matrix multiplication and the minimum cost."""
    n = len(dims) - 1
    m = [[0] * n for _ in range(n)]
    s = [[0] * n for _ in range(n)]

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            m[i][j] = float('inf')
            for k in range(i, j):
                cost = m[i][k] + m[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                if cost < m[i][j]:
                    m[i][j] = cost
                    s[i][j] = k

    return m, s

def print_optimal_parens(s, i, j):
    """Recursively prints the optimal parenthesization."""
    if i == j:
        return f"A{i+1}"
    else:
        k = s[i][j]
        return f"({print_optimal_parens(s, i, k)} x {print_optimal_parens(s, k + 1, j)})"

def multiply_matrices(matrices, order):
    """Multiplies the matrices according to the given parenthesization order."""
    def get_matrix(label):
        index = int(label[1:]) - 1
        return matrices[index]

    def perform_multiplication(op_str):
        if 'x' not in op_str:
            return get_matrix(op_str)

        # Find the outermost multiplication
        balance = 0
        split_index = -1
        for i, char in enumerate(op_str):
            if char == '(':
                balance += 1
            elif char == ')':
                balance -= 1
            elif char == 'x' and balance == 1:
                split_index = i
                break

        left_op = op_str[1:split_index]
        right_op = op_str[split_index + 1:-1]

        left_matrix = perform_multiplication(left_op)
        right_matrix = perform_multiplication(right_op)
        return np.dot(left_matrix, right_matrix)

    return perform_multiplication(order)

def main():
    st.title("Matrix Chain Multiplication Simulation")
    st.subheader("Visualize the optimal order and cost of multiplying matrices.")

    num_matrices = st.number_input("Number of matrices:", min_value=2, max_value=10, value=3, step=1)
    dimensions = []
    matrices = []

    st.subheader("Enter Matrix Dimensions:")
    for i in range(num_matrices):
        cols = 0
        if dimensions:
            cols = dimensions[-1][1]
        rows = st.number_input(f"Matrix A{i+1} - Rows:", min_value=1, value=np.random.randint(2, 6), key=f"rows_{i}")
        if i < num_matrices - 1:
            cols = st.number_input(f"Matrix A{i+1} - Columns (must match next matrix's rows):", min_value=1, value=np.random.randint(2, 6), key=f"cols_{i}")
        elif dimensions:
            cols = dimensions[-1][1]
        else:
            cols = st.number_input(f"Matrix A{i+1} - Columns:", min_value=1, value=np.random.randint(2, 6), key=f"cols_{i}")

        dimensions.append((rows, cols))

        if st.checkbox(f"Show/Edit Matrix A{i+1}", key=f"show_matrix_{i}"):
            matrix_data = np.zeros((rows, cols))
            for r in range(rows):
                cols_input = st.text_input(f"Row {r+1} (comma-separated values):", key=f"matrix_{i}_row_{r}", value=", ".join(map(str, np.random.randint(1, 10, cols))))
                try:
                    matrix_data[r] = np.array([int(x.strip()) for x in cols_input.split(',')])
                except ValueError:
                    st.error("Please enter valid integer values separated by commas.")
                    return
            matrices.append(matrix_data)
        else:
            matrices.append(np.random.randint(1, 10, size=(rows, cols)))

    if len(dimensions) > 1:
        dims_for_algo = [dimensions[0][0]] + [d[1] for d in dimensions]

        if st.button("Calculate Optimal Order"):
            m, s = matrix_chain_order(dims_for_algo)
            optimal_order = print_optimal_parens(s, 0, len(dimensions) - 1)
            min_cost = m[0][len(dimensions) - 1]

            st.subheader("Optimal Multiplication Order:")
            st.markdown(f"**{optimal_order}**")

            st.subheader("Minimum Cost (Number of scalar multiplications):")
            st.markdown(f"**{min_cost}**")

            if st.checkbox("Simulate Multiplication with Optimal Order"):
                try:
                    result_matrix = multiply_matrices(matrices, optimal_order)
                    st.subheader("Resultant Matrix:")
                    st.dataframe(pd.DataFrame(result_matrix))
                except ValueError as e:
                    st.error(f"Error during matrix multiplication: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred during multiplication: {e}")

    elif num_matrices == 1:
        st.info("Enter at least two matrices to find the optimal multiplication order.")

if __name__ == "__main__":
    main()
