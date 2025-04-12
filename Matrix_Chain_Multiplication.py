import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time

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

def draw_tables(m, s, step_info=None, highlight_cell=None):
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
    
    # Add step information if provided
    if step_info:
        plt.figtext(0.5, 0.01, step_info, ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    return fig

def main():
    st.title("Matrix Chain Multiplication Visualization")
    
    st.write("""
    This app visualizes the dynamic programming algorithm for Matrix Chain Multiplication.
    Enter the dimensions of the matrices and adjust the animation speed to see how the algorithm works.
    """)
    
    # Input for matrix dimensions
    st.subheader("Enter Matrix Dimensions")
    
    default_dims = "5,4,6,2,7"
    dims_input = st.text_input(
        "Enter dimensions separated by commas (e.g., '5,4,6,2,7' for matrices A1(5×4), A2(4×6), A3(6×2), A4(2×7))",
        value=default_dims
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
    animation_speed = st.slider("Animation Speed", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
    delay = 1 / animation_speed
    
    # Display matrix information
    st.subheader("Matrices")
    matrix_info = ""
    for i in range(len(dimensions) - 1):
        matrix_info += f"A{i+1}: {dimensions[i]}×{dimensions[i+1]}  "
    st.write(matrix_info)
    
    # Run button
    if st.button("Run Algorithm"):
        n = len(dimensions) - 1
        
        # Initialize tables
        m = [[0 for _ in range(n)] for _ in range(n)]
        s = [[0 for _ in range(n)] for _ in range(n)]
        
        # Create a placeholder for the visualization
        vis_placeholder = st.empty()
        info_placeholder = st.empty()
        
        # Show initial state
        fig = draw_tables(m, s, "Initial state: diagonal elements are 0")
        vis_placeholder.pyplot(fig)
        plt.close(fig)
        
        # Fill DP tables with visualization
        total_iterations = sum(range(1, n))
        current_iteration = 0
        
        progress_bar = st.progress(0)
        
        for l in range(1, n):
            for i in range(n - l):
                j = i + l
                m[i][j] = float('inf')
                
                step_info = f"Computing m[{i}][{j}] for chain length {l+1}"
                fig = draw_tables(m, s, step_info, highlight_cell=(i, j))
                vis_placeholder.pyplot(fig)
                plt.close(fig)
                time.sleep(delay)
                
                best_k = -1
                best_cost = float('inf')
                
                for k in range(i, j):
                    cost = m[i][k] + m[k+1][j] + dimensions[i] * dimensions[k+1] * dimensions[j+1]
                    
                    k_info = f"Testing split at k={k}: m[{i}][{k}] + m[{k+1}][{j}] + {dimensions[i]}×{dimensions[k+1]}×{dimensions[j+1]} = {cost}"
                    info_placeholder.info(k_info)
                    time.sleep(delay/2)
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_k = k
                
                m[i][j] = best_cost
                s[i][j] = best_k
                
                result_info = f"Best split at k={best_k} with cost {best_cost}"
                info_placeholder.success(result_info)
                
                fig = draw_tables(m, s, f"Updated m[{i}][{j}] = {best_cost}, s[{i}][{j}] = {best_k}", highlight_cell=(i, j))
                vis_placeholder.pyplot(fig)
                plt.close(fig)
                time.sleep(delay)
                
                current_iteration += 1
                progress_bar.progress(current_iteration / total_iterations)
        
        # Show final results
        st.subheader("Results")
        st.write(f"Minimum number of scalar multiplications: {m[0][n-1]}")
        optimal_parenthesization = print_optimal_parenthesization(s, 0, n-1)
        st.write(f"Optimal parenthesization: {optimal_parenthesization}")
        
        # Final visualization
        fig = draw_tables(m, s, "Final DP tables")
        st.pyplot(fig)
        plt.close(fig)
        
        # Explanation
        st.subheader("How It Works")
        st.write("""
        The algorithm uses dynamic programming to find the optimal way to multiply a sequence of matrices.
        
        - The m[i][j] table stores the minimum cost (number of scalar multiplications) to compute the product of matrices A_i through A_j.
        - The s[i][j] table stores the optimal split point k where the chain is divided into two sub-chains: A_i...A_k and A_(k+1)...A_j.
        
        The algorithm fills these tables diagonally, starting with the main diagonal (chain length 1) and working outward.
        For each cell m[i][j], it tries all possible split points k between i and j, and picks the one that minimizes the cost.
        """)

if __name__ == "__main__":
    main()
