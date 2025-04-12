import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import time

def print_optimal_parenthesization(s, i, j):
    if i == j:
        return f"A{i+1}"
    else:
        return f"({print_optimal_parenthesization(s, i, s[i][j])} × {print_optimal_parenthesization(s, s[i][j]+1, j)})"

def create_heatmap_data(matrix, highlight_cell=None, highlight_k=None):
    """Create data for Altair heatmap."""
    n = len(matrix)
    data = []
    
    for i in range(n):
        for j in range(n):
            if j >= i:  # Only use upper triangular part
                value = matrix[i][j]
                if value == float('inf'):
                    value_str = "∞"
                else:
                    value_str = str(value)
                
                # Determine highlight type
                highlight = "none"
                if highlight_cell and highlight_cell == (i, j):
                    highlight = "current"
                elif highlight_k and ((i, highlight_k) == (highlight_cell[0], j) or (highlight_k+1, j) == (i, highlight_cell[1])):
                    highlight = "k_related"
                
                data.append({
                    'row': i,
                    'col': j,
                    'value': value if value != float('inf') else 0,
                    'label': value_str,
                    'highlight': highlight
                })
    
    return pd.DataFrame(data)

def create_heatmap_chart(df, title, color_scheme='blues'):
    """Create an Altair heatmap from dataframe."""
    # Base heatmap
    base = alt.Chart(df).encode(
        x=alt.X('col:O', axis=alt.Axis(title='j')),
        y=alt.Y('row:O', axis=alt.Axis(title='i', orient='left')),
        tooltip=['row', 'col', 'label']
    )
    
    # Color cells based on value
    heatmap = base.mark_rect().encode(
        color=alt.Color('value:Q', 
                       scale=alt.Scale(scheme=color_scheme),
                       legend=alt.Legend(title='Value'))
    )
    
    # Add text labels
    text = base.mark_text().encode(
        text='label:N',
        color=alt.condition(
            alt.datum.value > 100, 
            alt.value('white'),
            alt.value('black')
        )
    )
    
    # Add borders for highlighted cells
    highlight_current = base.mark_rect(
        stroke='red',
        strokeWidth=2,
        fill=None
    ).transform_filter(
        alt.datum.highlight == 'current'
    )
    
    highlight_k = base.mark_rect(
        stroke='orange',
        strokeWidth=1,
        fill=None
    ).transform_filter(
        alt.datum.highlight == 'k_related'
    )
    
    # Combine layers
    chart = (heatmap + text + highlight_current + highlight_k).properties(
        title=title,
        width=300,
        height=300
    )
    
    return chart

def draw_tables_altair(m, s, highlight_cell=None, highlight_k=None):
    """Create Altair charts for m and s tables."""
    m_data = create_heatmap_data(m, highlight_cell, highlight_k)
    s_data = create_heatmap_data(s, highlight_cell)
    
    m_chart = create_heatmap_chart(m_data, "Cost Table (m)", "blues")
    s_chart = create_heatmap_chart(s_data, "Split Table (s)", "oranges")
    
    combined_chart = alt.hconcat(m_chart, s_chart).properties(
        title="Matrix Chain Multiplication DP Table Filling"
    )
    
    return combined_chart

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

def main():
    st.title("Matrix Chain Multiplication - Step by Step")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for setup
    st.sidebar.header("Setup")
    
    # Input for matrix dimensions
    default_dims = "5,4,6,2,7"
    dims_input = st.sidebar.text_input(
        "Matrix Dimensions",
        value=default_dims,
        help="Enter dimensions separated by commas (e.g., '5,4,6,2,7' for matrices A1(5×4), A2(4×6), A3(6×2), A4(2×7))"
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
    
    # Display matrix information
    st.subheader("Input Matrices")
    matrix_info = []
    for i in range(len(dimensions) - 1):
        matrix_info.append(f"A{i+1}: {dimensions[i]}×{dimensions[i+1]}")
    
    # Format as columns for better display
    cols = st.columns(min(5, len(matrix_info)))
    for i, matrix in enumerate(matrix_info):
        cols[i % len(cols)].write(matrix)
    
    # Initialize or reset button
    if not st.session_state.initialized or st.sidebar.button("Reset Algorithm"):
        reset_algorithm()
    
    # Main visualization area
    st.subheader("Dynamic Programming Tables")
    
    # Add step explanation
    step_container = st.container()
    vis_container = st.container()
    detail_container = st.container()
    next_step_container = st.container()
    
    # Draw current state
    highlight_cell = None
    highlight_k = None
    
    if not st.session_state.algorithm_complete:
        highlight_cell = (st.session_state.current_i, st.session_state.current_j)
        if st.session_state.step_phase == 'evaluate_k':
            highlight_k = st.session_state.current_k
    
    with vis_container:
        chart = draw_tables_altair(
            st.session_state.m, 
            st.session_state.s, 
            highlight_cell, 
            highlight_k
        )
        st.altair_chart(chart, use_container_width=True)
    
    # Show step details
    with step_container:
        if st.session_state.algorithm_complete:
            st.success("Algorithm complete! Final tables shown above.")
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
    
    # Show calculation details
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
