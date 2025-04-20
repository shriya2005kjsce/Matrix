import streamlit as st
import pandas as pd

st.set_page_config(page_title="Matrix Chain Multiplication", layout="wide")
st.title("🧮 Matrix Chain Multiplication - Step by Step")

# Sidebar input
st.sidebar.header("Setup")
dims_input = st.sidebar.text_input("Matrix Dimensions", value="5,4,6,2,7")
reset = st.sidebar.button("Reset Algorithm")

if reset:
    st.session_state.clear()
    st.experimental_rerun()

# Validate input
def parse_dimensions(input_str):
    try:
        dims = list(map(int, input_str.strip().split(',')))
        if len(dims) < 2:
            st.error("Enter at least two dimensions (e.g., 5,4).")
            return None
        return dims
    except ValueError:
        st.error("Only comma-separated integers allowed.")
        return None

# Matrix Chain Multiplication DP
@st.cache_data
def matrix_chain_order(p):
    n = len(p) - 1
    m = [[0] * n for _ in range(n)]
    s = [[0] * n for _ in range(n)]

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

# Steps for simulation
@st.cache_data
def generate_steps(p, m_table):
    n = len(p) - 1
    steps = []
    for chain_len in range(2, n + 1):
        for i in range(n - chain_len + 1):
            j = i + chain_len - 1
            min_cost = float('inf')
            best_k = -1
            for k in range(i, j):
                cost = (
                    (0 if i == k else m_table[i][k])
                    + (0 if k + 1 == j else m_table[k + 1][j])
                    + p[i] * p[k + 1] * p[j + 1]
                )
                if cost < min_cost:
                    min_cost = cost
                    best_k = k
            steps.append({
                "i": i + 1, "j": j + 1, "k": best_k + 1, "cost": min_cost
            })
    return steps

# Format matrix for display
def format_matrix(matrix):
    df = pd.DataFrame(matrix)
    df.index = [f"A{i+1}" for i in range(len(matrix))]
    df.columns = [f"A{j+1}" for j in range(len(matrix[0]))]
    return df

# Show current simulation step
def create_simulation_table(step, n):
    matrix = [["" for _ in range(n)] for _ in range(n)]
    i, j, k = step['i'], step['j'], step['k']
    matrix[i - 1][j - 1] = f"{step['cost']} (k={k})"
    return format_matrix(matrix)

# Format split matrix for display with +1
def format_split_matrix(s_matrix):
    n = len(s_matrix)
    formatted_s = [[s_matrix[i][j] + 1 if s_matrix[i][j] != 0 else "" for j in range(n)] for i in range(n)]
    df_s = pd.DataFrame(formatted_s)
    df_s.index = [f"i={i+1}" for i in range(n)]
    df_s.columns = [f"j={j+1}" for j in range(n)]
    return df_s

# Parse input
dims = parse_dimensions(dims_input)

if dims:
    st.subheader("📥 Input Matrices")
    for i in range(len(dims) - 1):
        st.markdown(f"*A{i+1}: {dims[i]}×{dims[i+1]}*")

    m, s = matrix_chain_order(dims)
    steps = generate_steps(dims, m)
    n = len(dims) - 1

    # Step control
    if "step_index" not in st.session_state:
        st.session_state.step_index = 0

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("⬅️ Previous Step"):
            if st.session_state.step_index > 0:
                st.session_state.step_index -= 1
    with col_btn2:
        if st.button("Next Step ➡️"):
            if st.session_state.step_index < len(steps) - 1:
                st.session_state.step_index += 1

    # Show simulation table
    st.subheader("🧪 Simulation - Current Step")
    step_table = create_simulation_table(steps[st.session_state.step_index], n)
    st.dataframe(step_table, use_container_width=True)

    # Show final cost and split tables
    st.subheader("📊 Dynamic Programming Tables")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("*Cost Table (m)*")
        st.dataframe(format_matrix(m), use_container_width=True)
    with col2:
        st.markdown("*Split Table (s)*")
        st.dataframe(format_split_matrix(s), use_container_width=True)

    # Optimal parenthesization
    def print_optimal_parens(s, i, j):
        if i == j:
            return f"A{i+1}"
        else:
            return f"({print_optimal_parens(s, i, s[i][j])} × {print_optimal_parens(s, s[i][j] + 1, j)})"

    st.subheader("✅ Optimal Parenthesization")
    optimal_order = print_optimal_parens(s, 0, n - 1)
    st.code(optimal_order)

    st.success(f"Minimum number of scalar multiplications: {m[0][n - 1]}")
