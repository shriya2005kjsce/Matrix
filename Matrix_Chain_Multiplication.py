import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Matrix Chain Multiplication", layout="wide")
st.title("🧮 Matrix Chain Multiplication - Step by Step")

# Sidebar input
st.sidebar.header("Setup")
dims_input = st.sidebar.text_input(
    "Matrix Dimensions",
    value="5,4,6,2,7",
    help="Enter dimensions as comma-separated values (e.g., 10,20,30,40)"
)
reset = st.sidebar.button("Reset Algorithm")

if reset:
    st.experimental_rerun()

# Helper function to parse and validate input
def parse_dimensions(input_str):
    try:
        dims = list(map(int, input_str.strip().split(',')))
        if len(dims) < 2:
            st.error("Please enter at least two dimensions (like 5,4).")
            return None
        return dims
    except ValueError:
        st.error("Please enter only comma-separated integers.")
        return None

# Cached function to compute cost and split tables
@st.cache_data
def matrix_chain_order(p):
    n = len(p) - 1
    m = [[0 for _ in range(n)] for _ in range(n)]
    s = [[0 for _ in range(n)] for _ in range(n)]

    for chain_len in range(2, n + 1):
        for i in range(n - chain_len + 1):
            j = i + chain_len - 1
            m[i][j] = float('inf')
            for k in range(i, j):
                cost = m[i][k] + m[k + 1][j] + p[i] * p[k + 1] * p[j + 1]
                if cost < m[i][j]:
                    m[i][j] = cost
                    s[i][j] = k
    return m, s

# Format matrix for display
def format_matrix(matrix):
    df = pd.DataFrame(matrix)
    df.replace(float('inf'), '∞', inplace=True)
    df.index = [f"A{i+1}" for i in range(len(matrix))]
    df.columns = [f"A{j+1}" for j in range(len(matrix[0]))]
    return df

# Parse input
dims = parse_dimensions(dims_input)

# If valid input, display matrices and results
if dims:
    if len(dims) < 3:
        st.warning("At least two matrices (3 dimensions) are required for multiplication.")
        st.stop()

    st.subheader("📥 Input Matrices")
    for i in range(len(dims) - 1):
        st.markdown(f"*A{i+1}: {dims[i]}×{dims[i+1]}*")

    # Compute MCM
    m, s = matrix_chain_order(dims)

    # Show dynamic programming tables
    st.subheader("📊 Dynamic Programming Tables")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("*Cost Table (m)*")
        st.dataframe(format_matrix(m), use_container_width=True)
    with col2:
        st.markdown("*Split Table (s)*")
        st.dataframe(format_matrix(s), use_container_width=True)

    # Recursive function to build optimal parenthesization
    def print_optimal_parens(s, i, j):
        if i == j:
            return f"A{i+1}"
        else:
            return f"({print_optimal_parens(s, i, s[i][j])} × {print_optimal_parens(s, s[i][j] + 1, j)})"

    # Display final result
    st.subheader("✅ Optimal Parenthesization")
    optimal_order = print_optimal_parens(s, 0, len(dims) - 2)
    st.markdown(f"**{optimal_order}**")

    st.success(f"Minimum number of scalar multiplications: {m[0][len(dims) - 2]}")

