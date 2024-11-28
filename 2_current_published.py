import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.title("Efficient Frontier Visualizer")

# Function to generate a PSD correlation matrix


def generate_random_parameters(size):
    """
    Generates random returns, standard deviations, and a PSD correlation matrix.
    """
    # Generate random returns and standard deviations
    returns = np.round(np.random.uniform(0.05, 0.2, size), 2).tolist()
    std_devs = np.round(np.random.uniform(0.1, 0.4, size), 2).tolist()

    # Generate random vectors and create a correlation matrix
    random_vectors = np.random.normal(
        size=(size, 100))  # 100 samples for stability
    corr_matrix = np.corrcoef(random_vectors)

    # Round to 2 decimals for presentation
    corr_matrix = np.round(corr_matrix, 2)
    np.fill_diagonal(corr_matrix, 1.0)  # Ensure diagonal elements are 1

    return returns, std_devs, corr_matrix.tolist()


# Default values
default_values = {
    "returns": [0.12, 0.18, 0.07],
    "std_devs": [0.2, 0.25, 0.15],
    "corr_matrix": [[1.00, 0.80, 0.50], [0.80, 1.00, 0.30], [0.50, 0.30, 1.00]],
}

# Initialize session state
if "current_values" not in st.session_state:
    # Default is a random 3x3 matrix
    returns, std_devs, corr_matrix = generate_random_parameters(3)
    st.session_state.current_values = {
        "returns": returns,
        "std_devs": std_devs,
        "corr_matrix": corr_matrix,
    }

# Sidebar: Actions
st.sidebar.header("Actions")
if st.sidebar.button("Reset to Default Values"):
    st.session_state.current_values = default_values.copy()

matrix_size = st.sidebar.slider("Matrix Size", 2, 8, 3)  # Default size is 3x3
if st.sidebar.button("Generate Random"):
    returns, std_devs, corr_matrix = generate_random_parameters(matrix_size)
    st.session_state.current_values = {
        "returns": returns,
        "std_devs": std_devs,
        "corr_matrix": corr_matrix,
    }

# Retrieve current parameters
current_returns = st.session_state.current_values["returns"]
current_std_devs = st.session_state.current_values["std_devs"]
current_corr_matrix = st.session_state.current_values["corr_matrix"]

# Sidebar: Input Parameters
st.sidebar.header("Input Parameters")
returns_input = st.sidebar.text_input(
    "Expected Returns (comma-separated)", value=",".join(map(str, current_returns)))
std_devs_input = st.sidebar.text_input(
    "Standard Deviations (comma-separated)", value=",".join(map(str, current_std_devs)))
corr_matrix_input = st.sidebar.text_area(
    "Correlation Matrix (comma-separated rows)",
    value="\n".join([",".join(map(str, row)) for row in current_corr_matrix])
)
risk_free_rate = st.sidebar.slider("Risk-Free Rate", 0.0, 0.1, 0.02, 0.005)

# Process input with error handling
try:
    mu = np.array(list(map(float, returns_input.split(","))))
    sigma = np.array(list(map(float, std_devs_input.split(","))))
    corr = np.array([list(map(float, row.split(",")))
                    for row in corr_matrix_input.split("\n")])

    # Ensure dimensions match
    if len(mu) != len(sigma):
        st.error(
            "The number of expected returns must match the number of standard deviations.")
        st.stop()
    if corr.shape != (len(mu), len(mu)):
        st.error("The correlation matrix dimensions must match the number of assets.")
        st.stop()

    # Check for valid correlation matrix
    if not np.allclose(corr, corr.T) or not np.all(np.linalg.eigvals(corr) >= 0):
        st.error(
            "The correlation matrix must be symmetric and positive semi-definite.")
        st.stop()

    # Create covariance matrix
    cov_matrix = np.diag(sigma) @ corr @ np.diag(sigma)
except Exception as e:
    st.error(f"Invalid input: {e}")
    st.stop()

# Portfolio optimization
weights = []
portfolio_returns = []
portfolio_risks = []
sharpe_ratios = []

num_portfolios = 5000
asset_count = len(mu)
for _ in range(num_portfolios):
    w = np.random.random(asset_count)
    w /= np.sum(w)
    weights.append(w)
    portfolio_return = np.dot(w, mu)
    portfolio_risk = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk

    portfolio_returns.append(portfolio_return)
    portfolio_risks.append(portfolio_risk)
    sharpe_ratios.append(sharpe_ratio)

weights = np.array(weights)
portfolio_returns = np.array(portfolio_returns)
portfolio_risks = np.array(portfolio_risks)
sharpe_ratios = np.array(sharpe_ratios)

# Find optimal portfolios
max_sharpe_idx = np.argmax(sharpe_ratios)
min_variance_idx = np.argmin(portfolio_risks)

# Plotting the efficient frontier with Plotly
fig2 = go.Figure()

# Scatter plot for efficient frontier
fig2.add_trace(go.Scatter(
    x=portfolio_risks,
    y=portfolio_returns,
    mode='markers',
    marker=dict(
        size=8,
        color=sharpe_ratios,
        colorscale='Blues',
        line=dict(width=1, color='black'),
        showscale=False,  # Remove the vertical color scale bar
    ),
    name='Portfolios', showlegend=False  # Hide this trace from the legend
))

# Highlight the Maximum Sharpe ratio portfolio
fig2.add_trace(go.Scatter(
    x=[portfolio_risks[max_sharpe_idx]],
    y=[portfolio_returns[max_sharpe_idx]],
    mode='markers',
    marker=dict(
        size=16,
        color='red',
        symbol='star',
        line=dict(width=2, color='black')
    ),
    name='Max Sharpe Ratio Portfolio'
))

# Highlight the Minimum Variance portfolio
fig2.add_trace(go.Scatter(
    x=[portfolio_risks[min_variance_idx]],
    y=[portfolio_returns[min_variance_idx]],
    mode='markers',
    marker=dict(
        size=16,
        color='orange',
        symbol='circle',
        line=dict(width=2, color='black')
    ),
    name='Min Variance Portfolio'
))

# Update layout for better presentation with larger font sizes and remove vertical bar
fig2.update_layout(
    title="Efficient Frontier",
    xaxis_title="Risk (Standard Deviation)",
    yaxis_title="Return",
    template="plotly_dark",
    showlegend=True,
    height=600,
    font=dict(
        family="Arial, sans-serif",  # Set font family (optional)
        size=26,  # Increase font size for all text
        color="white"  # Set font color to white for the main chart text
    ),
    # Adjust margins to remove space for color scale bar
    margin=dict(r=0, t=40, b=40, l=40),

    # Customize x and y axis fonts and colors
    xaxis=dict(
        # Axis title font size and color (black)
        title_font=dict(size=16, color='black'),
        # Tick label font size and color (black)
        tickfont=dict(size=16, color='black')
    ),
    yaxis=dict(
        # Axis title font size and color (black)
        title_font=dict(size=16, color='black'),
        # Tick label font size and color (black)
        tickfont=dict(size=16, color='black')
    )
)

st.plotly_chart(fig2)


# Display optimal portfolio details
st.subheader("Optimal Portfolios")
st.markdown("### Maximum Sharpe Ratio Portfolio")
st.write(f"**Return**: {portfolio_returns[max_sharpe_idx]:.2%}")
st.write(f"**Risk**: {portfolio_risks[max_sharpe_idx]:.2%}")
st.write(f"**Weights**: {weights[max_sharpe_idx]}")

st.markdown("### Minimum Variance Portfolio")
st.write(f"**Return**: {portfolio_returns[min_variance_idx]:.2%}")
st.write(f"**Risk**: {portfolio_risks[min_variance_idx]:.2%}")
st.write(f"**Weights**: {weights[min_variance_idx]}")
