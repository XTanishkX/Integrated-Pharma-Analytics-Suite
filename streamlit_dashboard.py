import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import statsmodels.formula.api as smf
from scipy.optimize import curve_fit
import os
import plotly.express as px

# --- 1. DATA LOADING FUNCTION ---
# We use Streamlit's caching to load data once
@st.cache_data
def load_data():
    """
    Loads the three pre-generated CSV files.
    """
    data_dir = "data"
    patient_path = os.path.join(data_dir, "patient_data.csv")
    conjoint_path = os.path.join(data_dir, "conjoint_data.csv")
    adoption_path = os.path.join(data_dir, "adoption_data.csv")
    
    if not all([os.path.exists(p) for p in [patient_path, conjoint_path, adoption_path]]):
        st.error(f"Error: One or more data files not found in '{data_dir}' directory.")
        st.info(f"Please run `data_generator.py` first to create the CSV files.")
        return None, None, None

    patient_df = pd.read_csv(patient_path)
    conjoint_df = pd.read_csv(conjoint_path)
    adoption_df = pd.read_csv(adoption_path)
    
    return patient_df, conjoint_df, adoption_df

# --- 2. STREAMLIT APPLICATION ---

st.set_page_config(layout="wide")

# Title and introduction from your one-pager
st.title("Integrated Pharma Analytics Suite")
st.subheader("Patient Journey, Conjoint & Forecasting")
st.markdown("""
**Role:** Lead data analyst (concept, modelling, insight synthesis) — personal project / portfolio  
**Duration:** 2 weeks (self-driven)  
**Tech:** Python (Pandas, NumPy, scikit-learn, statsmodels), Streamlit, matplotlib  
**Objective:** Demonstrate end-to-end commercial analytics for a hypothetical chronic-disease therapy: identify patient segments, quantify treatment preferences (conjoint), and forecast adoption for a new drug to drive launch strategy.
""")

# --- Load Data ---
patient_df, conjoint_df, adoption_df = load_data()

# Stop execution if data wasn't loaded
if patient_df is None:
    st.stop()

st.success("Analysis datasets loaded successfully.")

# Sidebar for data preview
st.sidebar.title("Data Preview")
if st.sidebar.checkbox("Show Patient Data (Head)"):
    st.sidebar.dataframe(patient_df.head())

if st.sidebar.checkbox("Show Conjoint Data (Head)"):
    st.sidebar.dataframe(conjoint_df.head())

if st.sidebar.checkbox("Show Adoption Data (Head)"):
    st.sidebar.dataframe(adoption_df.head())


# --- Main Application Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Patient Journey", 
    "2. Segmentation", 
    "3. Choice Modelling", 
    "4. Forecasting"
])

# --- TAB 1: PATIENT JOURNEY ---
with tab1:
    st.header("1. Patient Journey Analysis")
    st.markdown("Calculated time-to-treatment, adherence rates, and outcome rates.")
    
    # --- Interactive Filter ---
    st.markdown("---")
    st.subheader("Interactive Filters")
    all_treatments = patient_df['treatment_type'].unique()
    selected_treatments = st.multiselect(
        "Filter by Treatment Type:",
        options=all_treatments,
        default=all_treatments
    )
    
    if not selected_treatments:
        st.warning("Please select at least one treatment type.")
        filtered_df = patient_df.iloc[0:0] # Empty dataframe
    else:
        filtered_df = patient_df[patient_df['treatment_type'].isin(selected_treatments)]

    st.markdown("---")
    
    # Calculate metrics based on filtered data
    avg_time_to_tx = filtered_df['time_to_treatment_days'].mean()
    adherence_rate = filtered_df['adherence'].mean() * 100
    outcome_rate = filtered_df['outcome'].mean() * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg. Time to Treatment", f"{avg_time_to_tx:.1f} days")
    col2.metric("Overall Adherence Rate", f"{adherence_rate:.1f}%")
    col3.metric("Overall Outcome Rate", f"{outcome_rate:.1f}%")
    
    st.subheader("Analysis by Treatment Type")
    
    # Group by treatment type
    journey_by_tx = patient_df.groupby('treatment_type').agg({
        'time_to_treatment_days': 'mean',
        'adherence': 'mean',
        'outcome': 'mean',
        'patient_id': 'count'
    }).rename(columns={'patient_id': 'patient_count'}).reset_index()

    journey_by_tx['adherence'] *= 100
    journey_by_tx['outcome'] *= 100
    
    st.dataframe(journey_by_tx.style.format({
        'time_to_treatment_days': '{:.1f}',
        'adherence': '{:.1f}%',
        'outcome': '{:.1f}%'
    }))
    
    st.subheader("Time-to-Treatment Distribution")
    fig, ax = plt.subplots()
    # Plot data from the filtered dataframe
    if not filtered_df.empty:
        sns.histplot(filtered_df['time_to_treatment_days'], kde=True, bins=50, ax=ax, stat="density")
        ax.set_title("Distribution of Time from Diagnosis to Treatment (Filtered)")
    else:
        ax.set_title("No data to display. Please select a treatment type.")
    ax.set_xlabel("Days")
    ax.set_ylabel("Density")
    st.pyplot(fig)


# --- TAB 2: SEGMENTATION ---
with tab2:
    st.header("2. Patient Segmentation")
    st.markdown("Standardized features (age, comorbidity, time-to-treatment, adherence) → KMeans clustering (k=4) → segment profiles.")
    
    # Prepare features for clustering
    features = ['age', 'comorbidity_score', 'time_to_treatment_days', 'adherence']
    scaler = StandardScaler()
    patient_df_scaled = scaler.fit_transform(patient_df[features])
    
    # Run KMeans
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    patient_df['segment'] = kmeans.fit_predict(patient_df_scaled)
    patient_df['segment'] = 'S' + patient_df['segment'].astype(str)

    
    st.subheader("Segment Profiles (KMeans Results)")
    segment_profiles = patient_df.groupby('segment')[features + ['outcome']].mean()
    segment_profiles['segment_size'] = patient_df['segment'].value_counts() / len(patient_df) * 100
    
    st.dataframe(segment_profiles.style.format({
        'age': '{:.1f}',
        'comorbidity_score': '{:.2f}',
        'time_to_treatment_days': '{:.1f}',
        'adherence': '{:.1%}',
        'outcome': '{:.1%}',
        'segment_size': '{:.1f}%'
    }))
    
    st.markdown("""
    **Key Findings (example results):**
    * **Segment S0** (or equivalent): Tends to be older, high comorbidity, with low adherence and slow initiation. **Action:** Priority for adherence programs.
    * **Segment S2** (or equivalent): Tends to be younger, low comorbidity, with fast initiation and high outcomes. **Action:** Likely early adopters; target for launch.
    """)
    
    st.markdown("---")
    st.subheader("Segment Comparison (Radar Chart)")
    
    # Normalize data for radar chart (0-1 scale)
    radar_features = ['age', 'comorbidity_score', 'time_to_treatment_days', 'adherence', 'outcome']
    profiles_to_plot = segment_profiles[radar_features]
    scaler_radar = MinMaxScaler()
    profiles_normalized = scaler_radar.fit_transform(profiles_to_plot)
    profiles_normalized_df = pd.DataFrame(profiles_normalized, columns=radar_features, index=profiles_to_plot.index)
    
    # Melt for Plotly
    profiles_radar_df = profiles_normalized_df.reset_index().melt(id_vars='segment', var_name='Metric', value_name='Normalized Value')

    # Create Radar Chart
    fig_radar = px.line_polar(
        profiles_radar_df,
        r='Normalized Value',
        theta='Metric',
        color='segment',
        line_close=True,
        title="Comparing Segment Profiles (Normalized 0-1)"
    )
    st.plotly_chart(fig_radar, use_container_width=True)


    st.markdown("---")
    st.subheader("Segment Visualization (Scatter Plots)")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.scatterplot(data=patient_df, x='age', y='comorbidity_score', hue='segment', palette='viridis', ax=ax1, alpha=0.7)
    ax1.set_title("Age vs. Comorbidity Score")
    
    sns.scatterplot(data=patient_df, x='time_to_treatment_days', y='adherence', hue='segment', palette='viridis', ax=ax2, alpha=0.7)
    ax2.set_title("Time-to-Treatment vs. Adherence")
    st.pyplot(fig)


# --- TAB 3: CHOICE MODELLING ---
with tab3:
    st.header("3. Choice Modelling (Conjoint Analysis)")
    st.markdown("Estimated attribute utilities via Multinomial Logit to derive preference weights.")
    
    # Prepare data for model
    # Set reference levels for effect coding
    conjoint_df['efficacy'] = pd.Categorical(conjoint_df['efficacy'], categories=['50% symptom reduction', '60% symptom reduction', '70% symptom reduction'], ordered=False)
    conjoint_df['side_effects'] = pd.Categorical(conjoint_df['side_effects'], categories=['High', 'Medium', 'Low'], ordered=False)
    conjoint_df['cost'] = pd.Categorical(conjoint_df['cost'], categories=[150, 100, 50], ordered=False)
    conjoint_df['dosing'] = pd.Categorical(conjoint_df['dosing'], categories=['Daily (Pill)', 'Weekly (Injection)'], ordered=False)

    
    # Define formula with reference levels
    formula = "choice ~ C(efficacy, Treatment('50% symptom reduction')) + C(side_effects, Treatment('High')) + C(cost, Treatment(150)) + C(dosing, Treatment('Daily (Pill)'))"
    
    try:
        with st.spinner("Fitting Logit model..."):
            model = smf.logit(formula, data=conjoint_df).fit()
        
        st.subheader("Model Coefficients (Utilities)")
        
        # --- FIX: Parse the cluttered summary table ---
        st.markdown("The table below shows the 'utility' (coefficient) for each attribute level. A higher, positive number is more preferred by patients.")
        
        # Extract the coefficients table (this is the magic)
        results_summary = model.summary()
        results_df = pd.read_html(results_summary.tables[1].as_html(), header=0, index_col=0)[0]
        
        # Clean up and display
        results_df = results_df.reset_index().rename(columns={'index': 'Attribute Level'})
        st.dataframe(results_df.style.format({'coef': '{:.3f}', 'std err': '{:.3f}', 'z': '{:.2f}', 'P>|z|': '{:.3f}'}), use_container_width=True)

        # Use an expander for the full (cluttered) summary if needed
        with st.expander("Show Full Model Summary (Raw Output)"):
            st.text(model.summary())
        
        st.subheader("Key Findings")
        st.markdown("""
        * The model summary shows the log-odds (utility) for each attribute level, relative to the reference level.
        * **Efficacy:** The coefficients for '60%' and '70%' are positive and highly significant (high P>|z|), with '70%' being the largest. This confirms **efficacy is the dominant driver** of choice.
        * **Side Effects:** 'Medium' and 'Low' side effects have positive, significant coefficients, meaning patients strongly prefer to avoid 'High' side effects.
        * **Cost:** '100' and '50' have positive coefficients, meaning patients prefer lower cost (relative to $150).
        * **Implication:** The high weight on efficacy suggests pricing flexibility is possible if efficacy gains are marketed strongly.
        """)
        
        # --- IMPROVEMENT: Convert importance table to bar chart ---
        st.subheader("Attribute Importance (Based on Utility Range)")
        st.markdown("This chart shows the relative importance of each attribute. A higher bar means it's a more powerful driver of patient choice.")
        
        params = model.params
        importance = {
            'efficacy': params.filter(like='efficacy').max() - 0,
            'side_effects': params.filter(like='side_effects').max() - 0,
            'cost': params.filter(like='cost').max() - 0,
            'dosing': params.filter(like='dosing').max() - 0
        }
        importance_df = pd.DataFrame.from_dict(importance, orient='index', columns=['Utility Range'])
        importance_df['Relative Importance %'] = (importance_df['Utility Range'] / importance_df['Utility Range'].sum())
        
        # Sort for plotting
        importance_to_plot = importance_df.sort_values('Relative Importance %', ascending=False)
        
        # Display as a Bar Chart
        st.bar_chart(importance_to_plot, y='Relative Importance %')


    except Exception as e:
        st.error(f"An error occurred during model fitting: {e}")
        st.info("Note: Model fitting can sometimes fail due to data separation. Retrying may help.")


# --- TAB 4: FORECASTING ---
with tab4:
    st.header("4. Forecasting")
    st.markdown("Simulated cumulative adoption and fit logistic growth curve to estimate market potential (K), adoption rate (r), and inflection point (t0).")

    # Define the logistic function for curve fitting
    def logistic_func(t, K, r, t0):
        return K / (1 + np.exp(-r * (t - t0)))
        
    # Fit the curve
    try:
        p0 = [np.max(adoption_df['cumulative_adoption']), 0.1, np.median(adoption_df['month'])] # Initial guesses
        popt, pcov = curve_fit(logistic_func, adoption_df['month'], adoption_df['cumulative_adoption'], p0=p0, maxfev=5000)
        
        K_fit, r_fit, t0_fit = popt
        
        st.subheader("Fitted Curve Parameters")
        col1, col2, col3 = st.columns(3)
        col1.metric("Est. Market Potential (K)", f"{K_fit:,.0f} adopters")
        col2.metric("Est. Adoption Rate (r)", f"{r_fit:.3f}")
        col3.metric("Est. Inflection Point (t0)", f"{t0_fit:.1f} months")
        
        st.markdown(f"""
        **Key Findings:**
        * Forecast fit estimates a total market potential (K) of **~{K_fit:,.0f}** cumulative adopters.
        * The adoption inflection point (t0) is projected at **~{t0_fit:.1f} months** post-launch.
        * **Implication:** This informs launch timing, inventory planning, and sales team ramp-up, as adoption is expected to accelerate rapidly around month {t0_fit:.1f}.
        """)

        st.markdown("---")
        # --- INTERACTIVE IMPROVEMENT: What-if Scenarios ---
        st.subheader("Interactive Scenario Simulation")
        st.markdown("Adjust the sliders to see how changing market dynamics (e.g., due to marketing spend or competitive action) could affect the adoption curve.")
        
        # Create sliders based on fitted parameters
        K_sim = st.slider(
            "Simulated Market Potential (K)", 
            min_value=int(K_fit * 0.5), 
            max_value=int(K_fit * 1.5), 
            value=int(K_fit),
            step=1000
        )
        r_sim = st.slider(
            "Simulated Adoption Rate (r)", 
            min_value=r_fit * 0.5, 
            max_value=r_fit * 1.5, 
            value=r_fit,
            format="%.3f"
        )
        t0_sim = st.slider(
            "Simulated Inflection Point (t0)", 
            min_value=int(t0_fit * 0.5), 
            max_value=int(t0_fit * 1.5), 
            value=int(t0_fit),
            step=1
        )
        
        # Calculate new simulated curve
        adoption_df['forecast_fit'] = logistic_func(adoption_df['month'], K_fit, r_fit, t0_fit)
        adoption_df['simulated_fit'] = logistic_func(adoption_df['month'], K_sim, r_sim, t0_sim)

        # Plot the results
        st.subheader("Adoption Forecast")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.scatter(adoption_df['month'], adoption_df['cumulative_adoption'], label='Simulated Actuals', alpha=0.5, s=20, color='gray')
        ax.plot(adoption_df['month'], adoption_df['forecast_fit'], label='Original Fitted Curve', color='red', linestyle='--')
        ax.plot(adoption_df['month'], adoption_df['simulated_fit'], label='Interactive Simulation', color='blue', linestyle='-', linewidth=2)
        
        ax.axvline(t0_fit, color='red', linestyle=':', label=f'Original Inflection (t0={t0_fit:.1f})')
        ax.axvline(t0_sim, color='blue', linestyle=':', label=f'Simulated Inflection (t0={t0_sim:.1f})')
        
        ax.set_title("Cumulative Adoption Forecast (Logistic Fit)")
        ax.set_xlabel("Month")
        ax.set_ylabel("Cumulative Adopters")
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        st.pyplot(fig)
        
    except RuntimeError:
        st.error("Error: Curve fitting failed. This can sometimes happen with random data.")
