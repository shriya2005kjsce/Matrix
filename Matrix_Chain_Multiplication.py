import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def matrix_chain_multiplication_visual(dimensions, delay=0.5):
    n = len(dimensions) - 1
    m = [[0 if i == j else float('inf') for j in range(n)] for i in range(n)]
    s = [[0 for _ in range(n)] for _ in range(n)]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Matrix Chain Multiplication DP Table Filling", fontsize=16)

    frames = []

    for l in range(1, n):
        for i in range(n - l):
            j = i + l
            for k in range(i, j):
                cost = m[i][k] + m[k + 1][j] + dimensions[i] * dimensions[k + 1] * dimensions[j + 1]
                if cost < m[i][j]:
                    m[i][j] = cost
                    s[i][j] = k
            frames.append((np.array(m, dtype=object).copy(), np.array(s, dtype=object).copy(), i, j))

    plt.close(fig)
    return m, s, frames

def draw_tables(m, s, current_i=None, current_j=None):
    n = len(m)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].set_title("Cost Table (m)")
    axs[1].set_title("Split Table (s)")
    axs[0].imshow(m, cmap='Blues', vmin=0)
    axs[1].imshow(s, cmap='Oranges', vmin=0)

    for i in range(n):
        for j in range(n):
            if j >= i:
                val_m = m[i][j] if m[i][j] != float('inf') else '∞'
                color = 'red' if (i == current_i and j == current_j) else 'black'
                axs[0].text(j, i, f"{val_m}", ha='center', va='center', fontsize=10, color=color)
                axs[1].text(j, i, f"{s[i][j]}", ha='center', va='center', fontsize=10)

    axs[0].set_xticks(np.arange(n))
    axs[0].set_yticks(np.arange(n))
    axs[1].set_xticks(np.arange(n))
    axs[1].set_yticks(np.arange(n))

    st.pyplot(fig)

def print_optimal_parenthesization(s, i, j):
    if i == j:
        return f"A{i+1}"
    else:
        return f"({print_optimal_parenthesization(s, i, s[i][j])} × {print_optimal_parenthesization(s, s[i][j]+1, j)})"

# ------------------ Streamlit App --------------------

st.set_page_config(page_title="Matrix Chain Multiplication", layout="centered")
st.title("📐 Matrix Chain Multiplication Visualizer")
st.markdown("""
Enter the dimensions of matrices such that matrix A<sub>1</sub> has dimension `d0 x d1`, A<sub>2</sub> is `d1 x d2`, etc.
""", unsafe_allow_html=True)

dim_input = st.text_input("Enter matrix dimensions (e.g., `5 4 6 2 7`):", "5 4 6 2 7")
dimensions = list(map(int, dim_input.strip().split()))

if len(dimensions) < 2:
    st.warning("Please enter at least two dimensions.")
    st.stop()

if 'frames' not in st.session_state:
    m, s, frames = matrix_chain_multiplication_visual(dimensions)
    st.session_state.frames = frames
    st.session_state.final_m = m
    st.session_state.final_s = s
    st.session_state.step = 0

# Step controller
if st.button("🔄 Reset"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()

if st.button("▶️ Next Step"):
    if st.session_state.step < len(st.session_state.frames) - 1:
        st.session_state.step += 1

step = st.session_state.step
frames = st.session_state.frames
m_frame, s_frame, i, j = frames[step]

draw_tables(m_frame, s_frame, current_i=i, current_j=j)
st.info(f"Step {step+1} of {len(frames)} — Evaluating A{i+1} to A{j+1}")

if st.session_state.step == len(frames) - 1:
    final_m = st.session_state.final_m
    final_s = st.session_state.final_s
    st.success(f"✅ Minimum scalar multiplications: **{final_m[0][len(dimensions) - 2]}**")
    st.write("**Optimal Parenthesization:**")
    st.code(print_optimal_parenthesization(final_s, 0, len(dimensions)-2))
