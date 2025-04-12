import streamlit as st
import numpy as np
import pandas as pd

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
        st.session_state.dimensions = [5, 2, 4, 7]
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
        st.session_state.step_phase = 'start'
    if 'k_costs' not in st.session_state:
        st.session_state.k_costs = []
    if 'algorithm_complete' not in st.session_state:
        st.session_state.algorithm_complete = False
    if 'show_final_result' not in st.session_state:
        st.session_state.show_final_result = False
    if 'step_count' not in st.session_state:
        st.session_state.step_count = 0

def reset_algorithm():
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
    st.session_state.show_final_result = False
    st.session_state.step_count = 0

def run_full_algorithm():
    dimensions = st.session_state.dimensions
    n = len(dimensions) - 1
    m = [[0 for _ in range(n)] for _ in range(n)]
    s = [[0 for _ in range(n)] for _ in range(n)]
    
    for l in range(1, n):
        for i in range(n - l):
            j = i + l
            m[i][j] = float('inf')
            for k in range(i, j):
                cost = m[i][k] + m[k+1][j] + dimensions[i] * dimensions[k+1] * dimensions[j+1]
                if cost < m[i][j]:
                    m[i][j] = cost
                    s[i][j] = k
    
    st.session_state.m = m
    st.session_state.s = s
    st.session_state.algorithm_complete = True
    st.session_state.show_final_result = True

def handle_next_step():
    if st.session_state.algorithm_complete:
        return

    st.session_state.step_count += 1

    n = len(st.session_state.dimensions) - 1
    
    if st.session_state.step_phase == 'start':
        i = st.session_state.current_i
        j = st.session_state.current_j
        st.session_state.m[i][j] = float('inf')
        st.session_state.current_k = i
        st.session_state.best_k = -1
        st.session_state.best_cost = float('inf')
        st.session_state.k_costs = []
        st.session_state.step_phase = 'evaluate_k'
    
    elif st.session_state.step_phase == 'evaluate_k':
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
        
        if k + 1 < j:
            st.session_state.current_k += 1
        else:
            st.session_state.step_phase = 'update_best'
    
    elif st.session_state.step_phase == 'update_best':
        i = st.session_state.current_i
        j = st.session_state.current_j
        st.session_state.m[i][j] = st.session_state.best_cost
        st.session_state.s[i][j] = st.session_state.best_k
        st.session_state.step_phase = 'next_cell'
    
    elif st.session_state.step_phase == 'next_cell':
        i = st.session_state.current_i
        j = st.session_state.current_j
        l = st.session_state.current_l
        
        if i + 1 < n - l:
            st.session_state.current_i += 1
            st.session_state.current_j += 1
            st.session_state.step_phase = 'start'
        else:
            st.session_state.current_l += 1
            if st.session_state.current_l < n:
                st.session_state.current_i = 0
                st.session_state.current_j = st.session_state.current_l
                st.session_state.step_phase = 'start'
            else:
                st.session_state.algorithm_complete = True
                st.session_state.show_final_result = True
                st.session_state.step_phase = 'complete'

# Include the other functions like `display_matrix_tables`, `display_final_result`, and `main()` here as you already have them.

# Just modify the part in the `main()` function that displays the current step:
# Replace the step explanation block inside main() with the following:

# --- Inside main(), after drawing step explanation ---
    st.write(f"### 🔢 Step Number: {st.session_state.step_count}")

# --- Then continue with the rest of your rendering logic ---

# Note: The rest of your code remains as-is. Just insert the line to show the step count.
