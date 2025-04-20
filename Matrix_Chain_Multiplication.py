import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Matrix Chain Multiplication", layout="wide")
st.title("🧮 Matrix Chain Multiplication - Step by Step")

# Sidebar
st.sidebar.header("Setup")
dims_input = st.sidebar.text_input(
    "Matrix Dimensions",
    value="5,4,6,2,7",
    help="Enter dimensions as comma-separated values (e.g., 10,20,30,40)"
)
reset = st.sidebar.button("Reset")

# Helper to parse dimensions
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

dims = parse_dimensions(dims_input)

# Reset session state
if reset or 'step' not in st.session_state:
    st.session_state.step = 0
    st.session_state.steps_list = []
    st.session_state.m = []
    st.session_state.s = []

# Compute steps and fill m/s tables
def generate_steps(dims):
    n = len(dims) - 1
    m = [[0 for _ in range(n)] for _ in range(n)]
    s = [[-1 for _ in range(n)] for _ in range(n)]
    steps = []

    for chain_len in range(2, n + 1):
        for i in range(n - chain_len + 1):
            j = i + chain_len - 1
            m[i][j] = float('inf')
            for k in range(i, j):
                cost = m[i][k] + m[k+1][j] + dims[i] * dims[k+1] * dims[j+1]
                steps.append({
                    "i": i, "j": j, "k": k,
                    "cost": cost,
                    "prev_cost": m[i][j],
                    "dims": (dims[i], dims[k+1], dims[j+1]),
                    "update": cost < m[i][j]
                })
                if cost < m[i][j]:
                    m[i][j] = cost
                    s[i][j] = k

    return m, s, steps

# Initialize tables and steps once
if dims and not st.session_state.steps_list:
    if len(dims) < 3:
        st.warning("At least two matrices (3 dimensions) are required.")
        st.stop()
    m, s, steps_list = generate_steps(dims)
    st.session_state.m = m
    st.session_state.s = s
    st.session_state.steps_list = steps_list

# Buttons for navigation
colA, colB = st.columns([1, 1])
with colA:
    if st.button("⬅️ Previous Step") and st.session_state.step > 0:
        st.session_state.step -= 1
with colB:
    if st.button("Next Step ➡️") and st.session_state.step < len(st.session_state.steps_list) - 1:
        st.session_state.step += 1

# Show matrices involved
if dims:
    st.subheader("📥 Input Matrices")
    for i in range(len(dims) - 1):
        st.markdown(f"A{i+1}: {dims[i]}×{dims[i+1]}")

# Show current step info
if dims and st.session_state.steps_list:
    step_data = st.session_state.steps_list[st.session_state.step]
    i, j, k = step_data["i"], step_data["j"], step_data["k"]
    cost, prev, update = step_data["cost"], step_data["prev_cost"], step_data["update"]
    d1, d2, d3 = step_data["dims"]

    st.subheader("🔍 Current Step")
    st.markdown(f"**Updating m[{i+1},{j+1}] with split at k = {k+1}**")
    st.markdown(f"Cost = m[{i+1},{k+1}] + m[{k+2},{j+1}] + {d1}×{d2}×{d3} = {cost}")
    if update:
        st.success(f"✅ Updated m[{i+1},{j+1}] from {prev} to {cost}")
    else:
        st.info(f"ℹ️ No update. Current m[{i+1},{j+1}] = {prev}")

# Show simulation table for current step
def format_simulation(step, dims_len):
    n = dims_len - 1
    matrix = [["" for _ in range(n)] for _ in range(n)]
    i, j = step["i"], step["j"]
    matrix[i][j] = f"{step['cost']} (k={step['k']+1})"
    df = pd.DataFrame(matrix)
    df.index = [f"A{i+1}" for i in range(n)]
    df.columns = [f"A{j+1}" for j in range(n)]
    return df

st.subheader("📽️ Simulation (Current Step Only)")
if dims and st.session_state.steps_list:
    st.dataframe(format_simulation(st.session_state.steps_list[st.session_state.step], len(dims)), use_container_width=True)
else:
    st.info("Enter matrix dimensions to start the simulation.")

# Final tables (unchanged)
def format_matrix(matrix):
    df = pd.DataFrame(matrix)
    df.replace(float('inf'), '∞', inplace=True)
    df.index = [f"A{i+1}" for i in range(len(matrix))]
    df.columns = [f"A{j+1}" for j in range(len(matrix[0]))]
    return df

st.subheader("📊 Tables (Final after all steps)")
col1, col2 = st.columns(2)
with col1:
    st.markdown("Cost Table (m)")
    if st.session_state.m:
        st.dataframe(format_matrix(st.session_state.m), use_container_width=True)
    else:
        st.info("Cost table will be shown after the simulation runs.")
with col2:
    st.markdown("Split Table (s)")
    if st.session_state.s:
        s_formatted = [[val + 1 if val != -1 else "" for val in row] for row in st.session_state.s]
        df_s = pd.DataFrame(s_formatted)
        df_s.index = [f"A{i+1}" for i in range(len(st.session_state.s))]
        df_s.columns = [f"A{j+1}" for j in range(len(st.session_state.s[0]))]
        st.dataframe(df_s, use_container_width=True)
    else:
        st.info("Split table will be shown after the simulation runs.")

# Parenthesis construction
def print_optimal_parens(s, i, j):
    if i == j:
        return f"A{i+1}"
    else:
        return f"({print_optimal_parens(s, i, s[i][j])} × {print_optimal_parens(s, s[i][j]+1, j)})"

if dims and st.session_state.steps_list and st.session_state.step == len(st.session_state.steps_list) - 1:
    st.subheader("✅ Optimal Parenthesization")
    result = print_optimal_parens(st.session_state.s, 0, len(dims)-2)
    st.markdown(f"**{result}**")
    st.success(f"Minimum number of scalar multiplications: {st.session_state.m[0][len(dims)-2]}")
elif dims and not st.session_state.steps_list:
    st.info("Step through all operations to view final result.")
elif dims and st.session_state.steps_list and st.session_state.step < len(st.session_state.steps_list) - 1:
    st.info("Step through all operations to view the final result.")
