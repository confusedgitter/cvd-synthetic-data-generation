from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm, ks_2samp


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CVD Synthetic Data Lab",
    page_icon="🫀",
    layout="wide"
)


# =========================================================
# CONSTANTS
# =========================================================

DATA_PATH = Path(__file__).parent / "data" / "cardio_train.csv"

FEATURES = [
    "age",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "cardio"
]

CATEGORICAL_COLUMNS = [
    "gender",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "cardio"
]


# =========================================================
# LOAD REAL DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_PATH)

    return df[FEATURES].copy()


real_data = load_data()


# =========================================================
# GAUSSIAN COPULA SYNTHETIC DATA GENERATOR
# =========================================================

def generate_synthetic_data(
    real_df,
    number_of_records,
    random_seed=42
):

    rng = np.random.default_rng(random_seed)

    df = real_df.copy()

    n = len(df)

    # -----------------------------------------------------
    # STEP 1: Convert each feature into percentile ranks
    # -----------------------------------------------------

    uniform_data = pd.DataFrame(index=df.index)

    for column in FEATURES:

        ranks = df[column].rank(
            method="average"
        )

        # Convert ranks into probabilities
        uniform_values = (
            ranks - 0.5
        ) / n

        # Prevent values from reaching exactly 0 or 1
        uniform_values = np.clip(
            uniform_values,
            1e-6,
            1 - 1e-6
        )

        uniform_data[column] = uniform_values

    # -----------------------------------------------------
    # STEP 2: Convert uniform values to normal values
    # -----------------------------------------------------

    gaussian_data = pd.DataFrame(
        index=df.index
    )

    for column in FEATURES:

        gaussian_data[column] = norm.ppf(
            uniform_data[column]
        )

    # -----------------------------------------------------
    # STEP 3: Learn correlation structure
    # -----------------------------------------------------

    correlation_matrix = gaussian_data.corr().values

    # -----------------------------------------------------
    # STEP 4: Make correlation matrix numerically stable
    # -----------------------------------------------------

    eigenvalues, eigenvectors = np.linalg.eigh(
        correlation_matrix
    )

    eigenvalues = np.maximum(
        eigenvalues,
        1e-6
    )

    correlation_matrix = (
        eigenvectors
        @ np.diag(eigenvalues)
        @ eigenvectors.T
    )

    diagonal = np.sqrt(
        np.diag(correlation_matrix)
    )

    correlation_matrix = (
        correlation_matrix
        / np.outer(diagonal, diagonal)
    )

    # -----------------------------------------------------
    # STEP 5: Generate correlated Gaussian samples
    # -----------------------------------------------------

    synthetic_gaussian = rng.multivariate_normal(
        mean=np.zeros(len(FEATURES)),
        cov=correlation_matrix,
        size=number_of_records
    )

    # -----------------------------------------------------
    # STEP 6: Convert Gaussian samples to probabilities
    # -----------------------------------------------------

    synthetic_uniform = norm.cdf(
        synthetic_gaussian
    )

    # -----------------------------------------------------
    # STEP 7: Inverse empirical distribution
    # -----------------------------------------------------

    synthetic_data = pd.DataFrame()

    for index, column in enumerate(FEATURES):

        original_values = (
            df[column]
            .sort_values()
            .to_numpy()
        )

        probabilities = synthetic_uniform[:, index]

        positions = (
            probabilities * len(original_values)
        ).astype(int)

        positions = np.clip(
            positions,
            0,
            len(original_values) - 1
        )

        synthetic_data[column] = (
            original_values[positions]
        )

    # -----------------------------------------------------
    # STEP 8: Preserve integer formatting
    # -----------------------------------------------------

    integer_columns = [
        "age",
        "gender",
        "ap_hi",
        "ap_lo",
        "cholesterol",
        "gluc",
        "smoke",
        "alco",
        "active",
        "cardio"
    ]

    for column in integer_columns:

        synthetic_data[column] = (
            synthetic_data[column]
            .round()
            .astype(int)
        )

    synthetic_data["height"] = (
        synthetic_data["height"]
        .round()
        .astype(int)
    )

    synthetic_data["weight"] = (
        synthetic_data["weight"]
        .round(1)
    )

    return synthetic_data


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Synthetic Data Lab")

number_of_records = st.sidebar.slider(
    "Synthetic patient records",
    min_value=100,
    max_value=20000,
    value=2400,
    step=100
)

random_seed = st.sidebar.number_input(
    "Random seed",
    min_value=1,
    max_value=99999,
    value=42
)

generate_button = st.sidebar.button(
    "Generate Synthetic Data",
    type="primary",
    use_container_width=True
)


# =========================================================
# HEADER
# =========================================================

st.title(
    "Synthetic Cardiovascular Patient Data Generator"
)

st.write(
    "A statistical synthetic-data generation system "
    "built from the CVD risk-assessment dataset."
)


# =========================================================
# REAL DATA OVERVIEW
# =========================================================

st.subheader("Source Dataset")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Real Patients",
    f"{len(real_data):,}"
)

col2.metric(
    "Features",
    len(real_data.columns)
)

col3.metric(
    "Missing Values",
    int(real_data.isna().sum().sum())
)

col4.metric(
    "CVD Positive",
    f"{real_data['cardio'].mean() * 100:.1f}%"
)


# =========================================================
# GENERATION
# =========================================================

if (
    generate_button
    or "synthetic_data" not in st.session_state
):

    with st.spinner(
        "Learning distributions and generating synthetic patients..."
    ):

        st.session_state.synthetic_data = (
            generate_synthetic_data(
                real_data,
                number_of_records,
                random_seed
            )
        )

synthetic_data = st.session_state.synthetic_data


# =========================================================
# SYNTHETIC OVERVIEW
# =========================================================

st.subheader("Synthetic Dataset")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Synthetic Patients",
    f"{len(synthetic_data):,}"
)

col2.metric(
    "Average Age",
    f"{synthetic_data['age'].mean() / 365.25:.1f} yrs"
)

col3.metric(
    "Average BMI",
    f"{(
        synthetic_data['weight']
        / ((synthetic_data['height'] / 100) ** 2)
    ).mean():.1f}"
)

col4.metric(
    "CVD Positive",
    f"{synthetic_data['cardio'].mean() * 100:.1f}%"
)


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Synthetic Patients",
        "Distributions",
        "Correlation",
        "Real vs Synthetic"
    ]
)


# =========================================================
# TAB 1 — SYNTHETIC PATIENTS
# =========================================================

with tab1:

    st.subheader(
        "Generated Synthetic Patient Records"
    )

    display_data = synthetic_data.copy()

    display_data["age_years"] = (
        display_data["age"] / 365.25
    ).round(1)

    display_data["BMI"] = (
        display_data["weight"]
        / ((display_data["height"] / 100) ** 2)
    ).round(1)

    st.dataframe(
        display_data.head(100),
        use_container_width=True
    )

    csv = synthetic_data.to_csv(
        index=False
    )

    st.download_button(
        "Download Synthetic CVD Dataset",
        csv,
        "synthetic_cvd_patients.csv",
        "text/csv"
    )


# =========================================================
# TAB 2 — DISTRIBUTIONS
# =========================================================

with tab2:

    st.subheader(
        "Real vs Synthetic Distributions"
    )

    selected_feature = st.selectbox(
        "Select feature",
        [
            "age",
            "height",
            "weight",
            "ap_hi",
            "ap_lo",
            "cholesterol"
        ]
    )

    fig, ax = plt.subplots()

    ax.hist(
        real_data[selected_feature],
        bins=40,
        alpha=0.6,
        label="Real"
    )

    ax.hist(
        synthetic_data[selected_feature],
        bins=40,
        alpha=0.6,
        label="Synthetic"
    )

    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Frequency")
    ax.set_title(
        f"{selected_feature}: Real vs Synthetic"
    )

    ax.legend()

    st.pyplot(fig)


# =========================================================
# TAB 3 — CORRELATION
# =========================================================

with tab3:

    st.subheader(
        "Correlation Structure"
    )

    numeric_columns = [
        "age",
        "height",
        "weight",
        "ap_hi",
        "ap_lo",
        "cholesterol",
        "gluc",
        "smoke",
        "alco",
        "active",
        "cardio"
    ]

    real_corr = real_data[
        numeric_columns
    ].corr()

    synthetic_corr = synthetic_data[
        numeric_columns
    ].corr()

    col1, col2 = st.columns(2)

    with col1:

        st.write("Real Dataset")

        st.dataframe(
            real_corr.round(2),
            use_container_width=True
        )

    with col2:

        st.write("Synthetic Dataset")

        st.dataframe(
            synthetic_corr.round(2),
            use_container_width=True
        )


# =========================================================
# TAB 4 — REAL VS SYNTHETIC
# =========================================================

with tab4:

    st.subheader(
        "Synthetic Data Quality Evaluation"
    )

    numeric_features = [
        "age",
        "height",
        "weight",
        "ap_hi",
        "ap_lo",
        "cholesterol"
    ]

    results = []

    for feature in numeric_features:

        real_values = real_data[
            feature
        ]

        synthetic_values = synthetic_data[
            feature
        ]

        ks_statistic, p_value = ks_2samp(
            real_values,
            synthetic_values
        )

        results.append({
            "Feature": feature,
            "Real Mean": real_values.mean(),
            "Synthetic Mean": synthetic_values.mean(),
            "Real Std": real_values.std(),
            "Synthetic Std": synthetic_values.std(),
            "KS Statistic": ks_statistic,
            "KS p-value": p_value
        })

    comparison = pd.DataFrame(results)

    st.dataframe(
        comparison.round(3),
        use_container_width=True
    )

    # Correlation similarity

    real_corr = real_data[
        numeric_features
    ].corr()

    synthetic_corr = synthetic_data[
        numeric_features
    ].corr()

    correlation_error = np.mean(
        np.abs(
            real_corr.values
            - synthetic_corr.values
        )
    )

    st.metric(
        "Average Correlation Error",
        f"{correlation_error:.4f}"
    )

    st.info(
        "Lower correlation error indicates that the "
        "synthetic dataset better preserves the relationships "
        "between variables in the original dataset."
    )


# =========================================================
# DISCLAIMER
# =========================================================

st.divider()

st.caption(
    "Synthetic records are generated statistically from "
    "the source dataset. They are simulated records and "
    "must not be treated as real patient information or "
    "used for clinical decision-making."
)