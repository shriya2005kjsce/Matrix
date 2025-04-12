import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

def matrix_chain_order_with_visualization(dimensions, placeholder, delay=1):
    """Computes the optimal order and visualizes the DP table filling."""
    n = len(dimensions) - 1
    m = [[0] * n for _ in range(n)]
    s = [[0] * n for _ in range(n)]

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Matrix Chain Multiplication DP Table Filling", fontsize=14)

    def draw_tables(current_i=-1, current_j=-1, current_k=-1):
        axs[0].clear()
        axs[1].clear()

        # Cost Table (m)
        axs[0].set_title("Minimum Cost (m)")
        axs[0].set_xticks(np.arange(n))
        axs[0].set_yticks(np.arange(n))
        axs[0].set_xticklabels([f"j={i+1}" for i in range(n)])
        axs[0].set_yticklabels([f"i={i+1}" for i in range(n)])
        axs[0].tick_params(axis='both', which='major', labelsize=8)
        axs[0].grid(True, linestyle='--', alpha=0.6)
        im_m = axs[0].imshow(m, cmap='viridis', vmin=0)
        cbar_m = fig.colorbar(im_m, ax=axs[0], shrink=0.7)
        cbar_m.ax.tick_params(labelsize=8)
        for i_idx in range(n):
            for j_idx in range(n):
                if j_idx >= i_idx:
                    text_color = 'white' if m[i_idx][j_idx] > np.max(m) / 2 else 'black' if np.max(m) > 0 else 'black'
                    axs[0].text(j_idx, i_idx, f"{m[i_idx][j_idx] if m[i_idx][j_idx] != float('inf') else '∞'}",
                               ha='center', va='center', color=text_color, fontsize=8)
                else:
                    axs[0].text(j_idx, i_idx, "-", ha='center', va='center', color='gray', fontsize=8)
            if current_i != -1 and current_j != -1:
                rect = patches.Rectangle((current_j - 0.5, current_i - 0.5), 1, 1, linewidth=2, edgecolor='red', facecolor='none')
                axs[0].add_patch(rect)
            if current_i != -1 and current_k != -1 and current_k >= current_i:
                rect_k_left = patches.Rectangle((current_k - 0.5, current_i - 0.5), 1, 1, linewidth=1, edgecolor='blue', facecolor='none', linestyle='--')
                axs[0].add_patch(rect_k_left)
            if current_k + 1 != -1 and current_j != -1 and current_j >= current_k + 1:
                rect_k_right = patches.Rectangle((current_j - 0.5, current_k + 1 - 0.5), 1, 1, linewidth=1, edgecolor='green', facecolor='none', linestyle='--')
                axs[0].add_patch(rect_k_right)

        # Split Table (s)
        axs[1].set_title("Optimal Split (k)")
        axs[1].set_xticks(np.arange(n))
        axs[1].set_yticks(np.arange(n))
        axs[1].set_xticklabels([f"j={i+1}" for i in range(n)])
        axs[1].set_yticklabels([f"i={i+1}" for i in range(n)])
        axs[1].tick_params(axis='both', which='major', labelsize=8)
        axs[1].grid(True, linestyle='--', alpha=0.6)
        im_s = axs[1].imshow(s, cmap='plasma', vmin=-1, vmax=n - 1)
        cbar_s = fig.colorbar(im_s, ax=axs[1], shrink=0.7, ticks=np.arange(-1, n))
        cbar_s.ax.tick_params(labelsize=8)
        for i_idx in range(n):
            for j_idx in range(n):
                if j_idx >= i_idx:
                    axs[1].text(j_idx, i_idx, f"{s[i_idx][j_idx] + 1 if s[i_idx][j_idx] > 0 else '-'}",
                               ha='center', va='center', color='white', fontsize=8)
                else:
                    axs[1].text(j_idx, i_idx, "-", ha='center', va='center', color='gray', fontsize=8)
            if current_i != -1 and current_j != -1:
                rect_s = patches.Rectangle((current_j - 0.5, current_i - 0.5), 1, 1, linewidth=2, edgecolor='red', facecolor='none')
                axs[1].add_patch(rect_s)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to fit title
        placeholder.pyplot(fig)
        time.sleep(delay)

    # Fill DP tables
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            m[i][j] = float('inf')
            for k in range(i, j):
                cost = m[i][k] + m[k + 1][j] + dimensions[i] * dimensions[k + 1] * dimensions[j + 1]
                if cost < m[i][j]:
                    m[i][j] = cost
                    s[i][j] = k
                draw_tables(i, j, k)
    draw_tables() # Final draw

    plt.close(fig)
    return m, s

def print_optimal_parenthesization(s, i, j):
    """Recursively prints the optimal parenthesization."""
    if i == j:
        return f"A{i+1}"
    else:
        k = s[i][j]
        return f"({print_optimal_parenthesization(s, i, k)} × {print_optimal_parenthesization(s, k + 1, j)})"

def main():
    st.title("Visual Matrix Chain Multiplication")
    st.subheader("Observe the dynamic programming table filling process.")

    dimensions_input = st.text_input("Enter matrix dimensions (comma-separated, e.g., 10,20,30,40):", "5,4,6,2,7")
    dimensions_str_list = dimensions_input.split(',')
    try:
        dimensions = [int(d.strip()) for d in dimensions_str_list]
        if len(dimensions) < 2:
            st.error("Please enter at least two dimensions.")
            return
    except ValueError:
        st.error("Invalid input. Please enter comma-separated integers.")
        return

    visualization_placeholder = st.empty()

    if st.button("Visualize Calculation"):
        with visualization_placeholder:
            m, s = matrix_chain_order_with_visualization(dimensions, st.empty(), delay=0.5) # Adjust delay as needed

            n_matrices = len(dimensions) - 1
            if n_matrices > 0:
                st.subheader("Optimal Results:")
                st.markdown(f"**Minimum number of scalar multiplications:** {m[0][n_matrices - 1]}")
                st.markdown(f"**Optimal parenthesization:** {print_optimal_parenthesization(s, 0, n_matrices - 1)}")
            else:
                st.info("Enter at least two dimensions to see the visualization.")

if __name__ == "__main__":
    main()
