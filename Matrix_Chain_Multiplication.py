import streamlit as st
import numpy as np
import pandas as pd
import time

def print_optimal_parenthesization(s, i, j):
    if i == j:
        return f"A{i+1}"
    else:
        return f"({print_optimal_parenthesization(s, i, s[i][j])} × {print_optimal_parenthesization(s, s[i][j]+1, j)})"

def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'dimensions' not in st.session_state:
        st.session_state.dimensions = [5, 4, 6, 2, 7]
    if 'm' not in st.session_state:
        st.session_state.m = None
    if 's' not in st.session_state:
        st.session_state.s = None
    if 'current_l' not in st.session_state:
        st.session_state.current_l = 1
    if 'current_i' not in st.session_state:
        st.session_state.current_i = 0
    if 'current_j' not in st.session_state:
        st.session_state.current_j = 1
    if 'current_k' not in st.session_state:
        st.session_state.current_k = 0
    if 'best_k' not in st.session_state:
        st.session_state.best_k = -1
    if 'best_cost' not in st.session_state:
        st.session_state.best_cost = float('inf')
    if 'step_phase' not in st.session_state:
        st.session_state.step_phase = 'start'  # start, evaluate_k, update_best, next_cell
    if 'k_costs' not in st.session_state:
        st.session_state.k_costs = []
    if 'algorithm_complete' not in st.session_state:
        st.session_state.algorithm_complete = False

def reset_algorithm():
    """Reset the algorithm to start over"""
    st.session_state.initialized = True
    dimensions = st.session_state.dimensions
    n = len(dimensions) - 1
    
    st.session_state.m = [[0 for _ in range(n)] for _ in range(n)]
    st.session_state.s = [[0 for _ in range(n)] for _ in range(n)]
    
    st.session_state.current_l = 1
    st.session_state.current_i = 0
    st.session_state.current_j = 1
    st.session_state.current_k = 0
    st.session_state.best_k = -1
    st.session_state.best_cost = float('inf')
    st.session_state.step_phase = 'start'
    st.session_state.k_costs = []
    st.session_state.algorithm_complete = False

def handle_next_step():
    """Process the next step in the algorithm"""
    if st.session_state.algorithm_complete:
        return
    
    n = len(st.session_state.dimensions) - 1
    
    if st.session_state.step_phase == 'start':
        # Initialize a new cell calculation
        i = st.session_state.current_i
        j = st.session_state.current_j
        st.session_state.m[i][j] = float('inf')
        st.session_state.current_k = i
        st.session_state.best_k = -1
        st.session_state.best_cost = float('inf')
        st.session_state.k_costs = []
        st.session_state.step_phase = 'evaluate_k'
    
    elif st.session_state.step_phase == 'evaluate_k':
        # Calculate the cost for current k
        i = st.session_state.current_i
        j = st.session_state.current_j
        k = st.session_state.current_k
        dimensions = st.session_state.dimensions
        
        cost = (st.session_state.m[i][k] + 
                st.session_state.m[k+1][j] + 
                dimensions[i] * dimensions[k+1] * dimensions[j+1])
        
        st.session_state.k_costs.append((k, cost))
        
        if cost < st.session_state.best_cost:
            st.session_state.best_cost = cost
            st.session_state.best_k = k
        
        # Move to next k or update best
        if k + 1 < j:
            st.session_state.current_k += 1
        else:
            st.session_state.step_phase = 'update_best'
    
    elif st.session_state.step_phase == 'update_best':
        # Update m and s tables with best values
        i = st.session_state.current_i
        j = st.session_state.current_j
        
        st.session_state.m[i][j] = st.session_state.best_cost
        st.session_state.s[i][j] = st.session_state.best_k
        
        st.session_state.step_phase = 'next_cell'
    
    elif st.session_state.step_phase == 'next_cell':
        # Move to the next cell or next chain length
        i = st.session_state.current_i
        j = st.session_state.current_j
        l = st.session_state.current_l
        
        # Find next i, j position
        if i + 1 < n - l:
            # Move to next position in the same diagonal
            st.session_state.current_i += 1
            st.session_state.current_j += 1
            st.session_state.step_phase = 'start'
        else:
            # Move to next diagonal (chain length)
            st.session_state.current_l += 1
            if st.session_state.current_l < n:
                st.session_state.current_i = 0
                st.session_state.current_j = st.session_state.current_l
                st.session_state.step_phase = 'start'
            else:
                # Algorithm complete
                st.session_state.algorithm_complete = True
                st.session_state.step_phase = 'complete'

def display_matrix_tables(m, s, highlight_cell=None):
    """Display the m and s tables as styled matrix boxes"""
    n = len(m)
    
    # Function to style a cell
    def get_cell_style(is_highlight=False, is_current=False):
        if is_current:
            return "background-color: #ffcccc; padding: 10px; text-align: center; font-weight: bold; border: 1px solid #ddd;"
        elif is_highlight:
            return "background-color: #e6f7ff; padding: 10px; text-align: center; font-weight: bold; border: 1px solid #ddd;"
        else:
            return "background-color: #f0f8ff; padding: 10px; text-align: center; border: 1px solid #ddd;"
    
    # Display m table
    st.write("### Cost Table (m)")
    
    # Create a table with HTML for better styling
    html_table = '<div style="overflow-x: auto;"><table style="border-collapse: collapse; width: 100%; border: 2px solid #ddd;">'
    
    # Headers
    html_table += '<tr><th style="padding: 8px; border: 1px solid #ddd; background-color: #f2f2f2;"></th>'
    for j in range(n):
        html_table += f'<th style="padding: 8px; border: 1px solid #ddd; background-color: #f2f2f2;">j={j}</th>'
    html_table += '</tr>'
    
    # Data rows
    for i in range(n):
        html_table += f'<tr><th style="padding: 8px; border: 1px solid #ddd; background-color: #f2f2f2;">i={i}</th>'
        for j in range(n):
            is_current = highlight_cell and highlight_cell == (i, j)
            is_highlight = j >= i and not is_current
            
            value = m[i][j] if j >= i else ""
            if value == float('inf'):
                value_str = "∞"
            else:
                value_str = str(value)
                
            html_table += f'<td style="{get_cell_style(is_highlight, is_current)}">{value_str}</td>'
        html_table += '</tr>'
    
    html_table += '</table></div>'
    st.markdown(html_table, unsafe_allow_html=True)
    
    # Display s table
    st.write("### Split Table (s)")
    
    # Create a table with HTML for better styling
    html_table = '<div style="overflow-x: auto;"><table style="border-collapse: collapse; width: 100%; border: 2px solid #ddd;">'
    
    # Headers
    html_table += '<tr><th style="padding: 8px; border: 1px solid #ddd; background-color: #f2f2f2;"></th>'
    for j in range(n):
        html_table += f'<th style="padding: 8px; border: 1px solid #ddd; background-color: #f2f2f2;">j={j}</th>'
    html_table += '</tr>'
    
    # Data rows
    for i in range(n):
        html_table += f'<tr><th style="padding: 8px; border: 1px solid #ddd; background-color: #f2f2f2;">i={i}</th>'
        for j in range(n):
            is_current = highlight_cell and highlight_cell == (i, j)
            is_highlight = j >= i and not is_current
            
            value = s[i][j] if j >= i else ""
                
            html_table += f'<td style="{get_cell_style(is_highlight, is_current)}">{value}</td>'
        html_table += '</tr>'
    
    html_table += '</table></div>'
    st.markdown(html_table, unsafe_allow_html=True)

def main():
    st.title("Matrix Chain Multiplication - Step by Step")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for setup
    st.sidebar.header("Setup")
    
    # Input for matrix dimensions
    default_dims = "5,2,4,7"
    dims_input = st.sidebar.text_input(
        "Matrix Dimensions",
        value=default_dims,
        help="Enter dimensions separated by commas (e.g., '5,2,4,7' for matrices A1(5×2), A2(2×4), A3(4×7))"
    )
    
    try:
        dimensions = [int(x.strip()) for x in dims_input.split(",")]
        if len(dimensions) < 2:
            st.error("Please enter at least 2 dimensions (for 1 matrix).")
            return
        
        st.session_state.dimensions = dimensions
    except ValueError:
        st.error("Please enter valid integer dimensions.")
        return
    
    # Initialize or reset button
    if not st.session_state.initialized or st.sidebar.button("Reset Algorithm"):
        reset_algorithm()
    
    # Display matrix information
    st.subheader("Input Matrices")
    matrix_info = []
    for i in range(len(dimensions) - 1):
        matrix_info.append(f"A{i+1}: {dimensions[i]}×{dimensions[i+1]}")
    
    # Format as columns for better display
    cols = st.columns(min(5, len(matrix_info)))
    for i, matrix in enumerate(matrix_info):
        cols[i % len(cols)].write(matrix)
    
    # Main visualization area
    st.subheader("Dynamic Programming Tables")
    
    # Add step explanation
    step_container = st.container()
    
    # Show step details
    with step_container:
        if st.session_state.algorithm_complete:
            st.success("Algorithm complete! Final tables shown below.")
        else:
            phase = st.session_state.step_phase
            i = st.session_state.current_i
            j = st.session_state.current_j
            k = st.session_state.current_k
            l = st.session_state.current_l
            
            if phase == 'start':
                st.info(f"Starting calculation for cell m[{i}][{j}] (chain length {l+1})")
            elif phase == 'evaluate_k':
                st.info(f"Testing split at k={k} for m[{i}][{j}]")
            elif phase == 'update_best':
                st.info(f"Updating m[{i}][{j}] with best cost {st.session_state.best_cost} at split k={st.session_state.best_k}")
            elif phase == 'next_cell':
                st.info("Moving to next cell")
    
    # Draw current state using styled tables
    highlight_cell = None
    if not st.session_state.algorithm_complete:
        highlight_cell = (st.session_state.current_i, st.session_state.current_j)
    
    display_matrix_tables(st.session_state.m, st.session_state.s, highlight_cell)
    
    # Show calculation details
    detail_container = st.container()
    with detail_container:
        if st.session_state.step_phase == 'evaluate_k':
            i, j, k = st.session_state.current_i, st.session_state.current_j, st.session_state.current_k
            dims = st.session_state.dimensions
            st.write(f"**Current calculation:**")
            st.write(f"m[{i}][{j}] = min_{{k}} (m[{i}][{k}] + m[{k+1}][{j}] + d{i}·d{k+1}·d{j+1})")
            
            if k > i:
                st.write(f"Testing k={k}:")
                st.write(f"m[{i}][{k}] = {st.session_state.m[i][k]}")
                st.write(f"m[{k+1}][{j}] = {st.session_state.m[k+1][j]}")
                st.write(f"d{i}·d{k+1}·d{j+1} = {dims[i]}·{dims[k+1]}·{dims[j+1]} = {dims[i] * dims[k+1] * dims[j+1]}")
            
        elif st.session_state.step_phase == 'update_best' or st.session_state.step_phase == 'next_cell':
            if st.session_state.k_costs:
                st.write("**Cost for each split point:**")
                for k, cost in st.session_state.k_costs:
                    if k == st.session_state.best_k:
                        st.write(f"k={k}: {cost} ← **BEST**")
                    else:
                        st.write(f"k={k}: {cost}")
        
        elif st.session_state.algorithm_complete:
            i, j = 0, len(st.session_state.dimensions) - 2
            st.write(f"**Final Result:**")
            st.write(f"Minimum multiplications: {st.session_state.m[i][j]}")
            parenthesization = print_optimal_parenthesization(st.session_state.s, i, j)
            st.write(f"Optimal parenthesization: {parenthesization}")
    
    # Next step button
    next_step_container = st.container()
    with next_step_container:
        if st.session_state.algorithm_complete:
            if st.button("Restart Algorithm"):
                reset_algorithm()
        else:
            if st.button("Next Step"):
                handle_next_step()
    
    # Show explanation
    st.subheader("How It Works")
    with st.expander("Algorithm Explanation"):
        st.write("""
        The Matrix Chain Multiplication algorithm uses dynamic programming to find the optimal way to multiply a sequence of matrices.
        
        1. **Problem**: Given a chain of matrices A₁, A₂, ..., Aₙ with dimensions d₀×d₁, d₁×d₂, ..., dₙ₋₁×dₙ, find the most efficient way to multiply them together.
        
        2. **DP Tables**:
           - m[i][j] stores the minimum number of scalar multiplications needed to compute A₁ · A₁₊₁ · ... · Aⱼ
           - s[i][j] stores the optimal split point k where the chain is divided into (A₁...Aₖ)(Aₖ₊₁...Aⱼ)
        
        3. **Filling the Tables**:
           - We fill the tables diagonally, starting with chains of length 1, then length 2, and so on.
           - For each m[i][j], we try all possible split points k between i and j, and choose the one with minimum cost.
        
        4. **Cost Formula**:
           - m[i][j] = min{ m[i][k] + m[k+1][j] + d₁ × dₖ₊₁ × dⱼ₊₁ } for all k where i ≤ k < j
        """)

if __name__ == "__main__":
    main()
