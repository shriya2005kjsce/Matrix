import streamlit as st
import numpy as np
import time

# Add requirements file for deployment
# Save this in a file named "requirements.txt" in your repository
# requirements.txt:
# streamlit
# numpy
# matplotlib
# altair

# Add import for altair as a backup visualization option
import altair as alt
import pandas as pd

# Import matplotlib safely with fallback option
try:
    import matplotlib.pyplot as plt
    matplotlib_available = True
except ImportError:
    matplotlib_available = False
    st.warning("matplotlib is not installed. Using Altair for visualizations instead.")

def matrix_chain_multiplication(dimensions):
    n = len(dimensions) - 1
    m = [[0 for _ in range(n)] for _ in range(n)]
    s = [[0 for _ in range(n)] for _ in range(n)]
    
    # Fill DP tables
    for l in range(1, n):
        for i in range(n - l):
            j = i + l
            m[i][j] = float('inf')
            for k in range(i, j):
                cost = m[i][k] + m[k+1][j] + dimensions[i] * dimensions[k+1] * dimensions[j+1]
                if cost < m[i][j]:
                    m[i][j] = cost
                    s[i][j] = k
    
    return m, s

def print_optimal_parenthesization(s, i, j):
    if i == j:
        return f"A{i+1}"
    else:
        return f"({print_optimal_parenthesization(s, i, s[i][j])} × {print_optimal_parenthesization(s, s[i][j]+1, j)})"

def create_heatmap_data(matrix, highlight_cell=None):
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
                
                highlighted = (highlight_cell and highlight_cell == (i, j))
                
                data.append({
                    'row': i,
                    'col': j,
                    'value': value if value != float('inf') else 0,
                    'label': value_str,
                    'highlighted': highlighted
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
    
    # Add red border for highlighted cell
    highlight = base.mark_rect(
        stroke='red',
        strokeWidth=2,
        fill=None
    ).transform_filter(
        alt.datum.highlighted == True
    )
    
    # Combine layers
    chart = (heatmap + text + highlight).properties(
        title=title,
        width=300,
        height=300
    )
    
    return chart

def draw_tables_altair(m, s, highlight_cell=None):
    """Create Altair charts for m and s tables."""
    m_data = create_heatmap_data(m, highlight_cell)
    s_data = create_heatmap_data(s, highlight_cell)
    
    m_chart = create_heatmap_chart(m_data, "Cost Table (m)", "blues")
    s_chart = create_heatmap_chart(s_data, "Split Table (s)", "oranges")
    
    combined_chart = alt.hconcat(m_chart, s_chart).properties(
        title="Matrix Chain Multiplication DP Table Filling"
    )
    
    return combined_chart

def draw_tables_matplotlib(m, s, highlight_cell=None):
    """Create matplotlib visualization for m and s tables."""
    if not matplotlib_available:
        return None
        
    n = len(m)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Matrix Chain Multiplication DP Table Filling", fontsize=16)
    
    # Draw m table
    axs[0].set_title("Cost Table (m)")
    axs[0].imshow(m, cmap='Blues', vmin=0)
    
    # Draw s table
    axs[1].set_title("Split Table (s)")
    axs[1].imshow(s, cmap='Oranges', vmin=0)
    
    # Add text and coordinates
    for i in range(n):
        for j in range(n):
            if j >= i:
                # Highlight the currently processed cell
                if highlight_cell and highlight_cell == (i, j):
                    axs[0].add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='red', lw=2))
                    axs[1].add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='red', lw=2))
                
                # Add text values
                axs[0].text(j, i, f"{m[i][j] if m[i][j] != float('inf') else '∞'}", 
                            ha='center', va='center', fontsize=10,
                            color='black' if m[i][j] < 1000 else 'white')
                axs[1].text(j, i, f"{s[i][j]}", ha='center', va='center', fontsize=10)
    
    axs[0].set_xticks(np.arange(n))
    axs[0].set_yticks(np.arange(n))
    axs[1].set_xticks(np.arange(n))
    axs[1].set_yticks(np.arange(n))
    
    return fig

def main():
    st.title("Matrix Chain Multiplication Visualization")
    
    st.write("This app visualizes the dynamic programming algorithm for Matrix Chain Multiplication.")
    
    # Sidebar for controls
    st.sidebar.header("Controls")
    
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
    except ValueError:
        st.error("Please enter valid integer dimensions.")
        return
    
    # Animation speed
    animation_speed = st.sidebar.slider("Animation Speed", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
    delay = 1 / animation_speed
    
    # Visualization method
    viz_method = st.sidebar.radio(
        "Visualization Method",
        ["Auto (Recommended)", "Altair", "Matplotlib"],
        index=0,
        help="Choose the visualization library. Auto will use Matplotlib if available, otherwise Altair."
    )
    
    if viz_method == "Matplotlib" and not matplotlib_available:
        st.warning("Matplotlib is not available. Using Altair instead.")
        viz_method = "Altair"
    
    if viz_method == "Auto (Recommended)":
        viz_method = "Matplotlib" if matplotlib_available else "Altair"
    
    # Display matrix information
    st.markdown("### Input Matrices")
    matrix_info = []
    for i in range(len(dimensions) - 1):
        matrix_info.append(f"A{i+1}: {dimensions[i]}×{dimensions[i+1]}")
    
    # Format as columns for better display
    cols = st.columns(min(5, len(matrix_info)))
    for i, matrix in enumerate(matrix_info):
        cols[i % len(cols)].write(matrix)
    
    # Run button
    if st.sidebar.button("Run Algorithm"):
        n = len(dimensions) - 1
        
        # Initialize tables
        m = [[0 for _ in range(n)] for _ in range(n)]
        s = [[0 for _ in range(n)] for _ in range(n)]
        
        # Create containers for different parts of the visualization
        vis_container = st.container()
        with vis_container:
            st.markdown("### Visualization")
            vis_placeholder = st.empty()
            progress_bar = st.progress(0)
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                current_step = st.empty()
            with info_col2:
                calculation_details = st.empty()
        
        # Show initial state
        with vis_placeholder.container():
            st.write("Initial state: diagonal elements are 0")
            if viz_method == "Matplotlib":
                fig = draw_tables_matplotlib(m, s)
                st.pyplot(fig)
                plt.close(fig)
            else:
                chart = draw_tables_altair(m, s)
                st.altair_chart(chart, use_container_width=True)
        
        # Fill DP tables with visualization
        total_iterations = sum(range(1, n))
        current_iteration = 0
        
        for l in range(1, n):
            for i in range(n - l):
                j = i + l
                m[i][j] = float('inf')
                
                current_step.write(f"Computing m[{i}][{j}] for chain length {l+1}")
                
                with vis_placeholder.container():
                    if viz_method == "Matplotlib":
                        fig = draw_tables_matplotlib(m, s, highlight_cell=(i, j))
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        chart = draw_tables_altair(m, s, highlight_cell=(i, j))
                        st.altair_chart(chart, use_container_width=True)
                
                time.sleep(delay)
                
                best_k = -1
                best_cost = float('inf')
                
                k_details = []
                for k in range(i, j):
                    cost = m[i][k] + m[k+1][j] + dimensions[i] * dimensions[k+1] * dimensions[j+1]
                    
                    k_info = f"Split at k={k}: {m[i][k]} + {m[k+1][j]} + {dimensions[i]}×{dimensions[k+1]}×{dimensions[j+1]} = {cost}"
                    k_details.append(k_info)
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_k = k
                
                calculation_details.write("\n".join(k_details))
                time.sleep(delay)
                
                m[i][j] = best_cost
                s[i][j] = best_k
                
                current_step.write(f"Best split at k={best_k} with cost {best_cost}")
                
                with vis_placeholder.container():
                    if viz_method == "Matplotlib":
                        fig = draw_tables_matplotlib(m, s, highlight_cell=(i, j))
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        chart = draw_tables_altair(m, s, highlight_cell=(i, j))
                        st.altair_chart(chart, use_container_width=True)
                
                time.sleep(delay)
                
                current_iteration += 1
                progress_bar.progress(current_iteration / total_iterations)
        
        # Show final results
        st.markdown("### Results")
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.write(f"Minimum number of scalar multiplications: {m[0][n-1]}")
        
        with result_col2:
            optimal_parenthesization = print_optimal_parenthesization(s, 0, n-1)
            st.write(f"Optimal parenthesization: {optimal_parenthesization}")
        
        # Final visualization
        with vis_placeholder.container():
            st.write("Final DP tables")
            if viz_method == "Matplotlib":
                fig = draw_tables_matplotlib(m, s)
                st.pyplot(fig)
                plt.close(fig)
            else:
                chart = draw_tables_altair(m, s)
                st.altair_chart(chart, use_container_width=True)
        
        # Explanation
        st.markdown("### How It Works")
        st.write("""
        The algorithm uses dynamic programming to find the optimal way to multiply a sequence of matrices.
        
        - The m[i][j] table stores the minimum cost (number of scalar multiplications) to compute the product of matrices A_i through A_j.
        - The s[i][j] table stores the optimal split point k where the chain is divided into two sub-chains: A_i...A_k and A_(k+1)...A_j.
        
        The algorithm fills these tables diagonally, starting with the main diagonal (chain length 1) and working outward.
        For each cell m[i][j], it tries all possible split points k between i and j, and picks the one that minimizes the cost.
        """)
    
    else:
        st.info("Click 'Run Algorithm' in the sidebar to start the visualization.")

if __name__ == "__main__":
    main()
