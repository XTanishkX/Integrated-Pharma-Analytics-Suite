import pandas as pd
import numpy as np
import os

print("Starting data generation...")

# --- 1. PATIENT DATA GENERATION ---
def generate_patient_data(n=5000):
    """
    Generates synthetic patient data with built-in segments.
    This is reverse-engineered to ensure KMeans finds the patterns
    from your one-pager.
    """
    segments = {}
    
    # Segment S0: older, high comorb, low adherence, slow init
    n_s0 = int(n * 0.25)
    segments[0] = pd.DataFrame({
        'patient_id': [f'P{i:05d}' for i in range(n_s0)],
        'age': np.random.normal(loc=68, scale=5, size=n_s0).astype(int).clip(60, 85),
        'comorbidity_score': np.random.normal(loc=3.5, scale=0.5, size=n_s0).clip(2.5, 4.5),
        'time_to_treatment_days': np.random.normal(loc=90, scale=20, size=n_s0).astype(int).clip(45, 150),
        'adherence': np.random.choice([0, 1], size=n_s0, p=[0.7, 0.3]), # Low adherence
        'outcome': np.random.choice([0, 1], size=n_s0, p=[0.6, 0.4]), # Low outcome
        'true_segment': 'S0'
    })
    
    # Segment S2: younger, low comorb, fast init, high outcome (Early Adopters)
    n_s2 = int(n * 0.30)
    segments[2] = pd.DataFrame({
        'patient_id': [f'P{i:05d}' for i in range(n_s0, n_s0 + n_s2)],
        'age': np.random.normal(loc=45, scale=7, size=n_s2).astype(int).clip(30, 60),
        'comorbidity_score': np.random.normal(loc=0.8, scale=0.4, size=n_s2).clip(0, 1.5),
        'time_to_treatment_days': np.random.normal(loc=15, scale=5, size=n_s2).astype(int).clip(7, 30),
        'adherence': np.random.choice([0, 1], size=n_s2, p=[0.1, 0.9]), # High adherence
        'outcome': np.random.choice([0, 1], size=n_s2, p=[0.15, 0.85]), # High outcome
        'true_segment': 'S2'
    })
    
    # Segment S1: "Average" patients, moderate everything
    n_s1 = int(n * 0.30)
    segments[1] = pd.DataFrame({
        'patient_id': [f'P{i:05d}' for i in range(n_s0 + n_s2, n_s0 + n_s2 + n_s1)],
        'age': np.random.normal(loc=55, scale=8, size=n_s1).astype(int).clip(40, 70),
        'comorbidity_score': np.random.normal(loc=2.0, scale=0.5, size=n_s1).clip(1.0, 3.0),
        'time_to_treatment_days': np.random.normal(loc=45, scale=10, size=n_s1).astype(int).clip(20, 70),
        'adherence': np.random.choice([0, 1], size=n_s1, p=[0.4, 0.6]), # Moderate adherence
        'outcome': np.random.choice([0, 1], size=n_s1, p=[0.4, 0.6]), # Moderate outcome
        'true_segment': 'S1'
    })
    
    # Segment S3: "Cautious", late start, moderate comorb
    n_s3 = n - n_s0 - n_s1 - n_s2
    segments[3] = pd.DataFrame({
        'patient_id': [f'P{i:05d}' for i in range(n_s0 + n_s2 + n_s1, n)],
        'age': np.random.normal(loc=60, scale=6, size=n_s3).astype(int).clip(50, 75),
        'comorbidity_score': np.random.normal(loc=2.2, scale=0.5, size=n_s3).clip(1.5, 3.0),
        'time_to_treatment_days': np.random.normal(loc=120, scale=20, size=n_s3).astype(int).clip(90, 180),
        'adherence': np.random.choice([0, 1], size=n_s3, p=[0.3, 0.7]), # Good adherence
        'outcome': np.random.choice([0, 1], size=n_s3, p=[0.2, 0.8]), # Good outcome
        'true_segment': 'S3'
    })

    # Combine and shuffle
    patient_df = pd.concat(segments.values()).sample(frac=1).reset_index(drop=True)
    
    # Add other fields
    patient_df['diagnosis_date'] = pd.to_datetime('2022-01-01') - pd.to_timedelta(np.random.randint(30, 365, size=n), unit='d')
    patient_df['treatment_start_date'] = patient_df['diagnosis_date'] + pd.to_timedelta(patient_df['time_to_treatment_days'], unit='d')
    patient_df['outcome_date'] = patient_df['treatment_start_date'] + pd.to_timedelta(np.random.randint(180, 365, size=n), unit='d')
    patient_df['outcome_date'] = patient_df.apply(lambda row: row['outcome_date'] if row['outcome'] == 1 else pd.NaT, axis=1)
    
    # Add treatment type
    patient_df['treatment_type'] = np.random.choice(['Drug A', 'Drug B', 'Standard of Care'], size=n, p=[0.4, 0.4, 0.2])
    
    return patient_df

# --- 2. CONJOINT DATA GENERATION ---
def generate_conjoint_data(n_respondents=800, n_tasks=8):
    """
    Generates synthetic choice-based conjoint data.
    Utilities are predefined to ensure efficacy is the most important attribute.
    """
    attributes = {
        'efficacy': ['50% symptom reduction', '60% symptom reduction', '70% symptom reduction'],
        'side_effects': ['High', 'Medium', 'Low'],
        'cost': [150, 100, 50], # Monthly cost
        'dosing': ['Daily (Pill)', 'Weekly (Injection)']
    }
    
    # Define "true" utilities (what the model will find)
    true_utilities = {
        'efficacy': {'50% symptom reduction': 0, '60% symptom reduction': 0.8, '70% symptom reduction': 1.5},
        'side_effects': {'High': 0, 'Medium': 0.4, 'Low': 0.7},
        'cost': {150: 0, 100: 0.3, 50: 0.5},
        'dosing': {'Daily (Pill)': 0, 'Weekly (Injection)': 0.2}
    }
    
    data = []
    for resp_id in range(n_respondents * n_tasks): # Each respondent-task is a choice scenario
        profile = {
            'efficacy': np.random.choice(attributes['efficacy']),
            'side_effects': np.random.choice(attributes['side_effects']),
            'cost': np.random.choice(attributes['cost']),
            'dosing': np.random.choice(attributes['dosing'])
        }
        
        utility = (true_utilities['efficacy'][profile['efficacy']] +
                   true_utilities['side_effects'][profile['side_effects']] +
                   true_utilities['cost'][profile['cost']] +
                   true_utilities['dosing'][profile['dosing']])
        
        prob = 1 / (1 + np.exp(-(utility - 1.0))) # '1.0' is an assumed intercept
        choice = np.random.binomial(1, prob)
        
        data.append({
            'respondent_id': resp_id,
            'efficacy': profile['efficacy'],
            'side_effects': profile['side_effects'],
            'cost': profile['cost'],
            'dosing': profile['dosing'],
            'choice': choice
        })

    return pd.DataFrame(data)

# --- 3. ADOPTION DATA GENERATION ---
def generate_adoption_data(n_months=48):
    """
    Generates synthetic cumulative adoption data that
    follows a logistic (S-curve) pattern.
    """
    
    def logistic_func(t, K, r, t0):
        return K / (1 + np.exp(-r * (t - t0)))

    K_true = 50000
    r_true = 0.25 
    t0_true = 18
    
    t = np.arange(1, n_months + 1)
    y_true = logistic_func(t, K_true, r_true, t0_true)
    
    noise = np.random.normal(0, K_true * 0.01, size=t.shape)
    y_simulated = y_true + np.cumsum(noise)
    y_simulated = np.maximum.accumulate(y_simulated)
    y_simulated = y_simulated.clip(0)
    
    return pd.DataFrame({'month': t, 'cumulative_adoption': y_simulated})

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Define output directory
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Generate and save patient data
    patient_df = generate_patient_data(n=5000)
    patient_path = os.path.join(output_dir, "patient_data.csv")
    patient_df.to_csv(patient_path, index=False)
    print(f"Generated and saved patient data to {patient_path}")

    # 2. Generate and save conjoint data
    conjoint_df = generate_conjoint_data(n_respondents=800, n_tasks=8)
    conjoint_path = os.path.join(output_dir, "conjoint_data.csv")
    conjoint_df.to_csv(conjoint_path, index=False)
    print(f"Generated and saved conjoint data to {conjoint_path}")

    # 3. Generate and save adoption data
    adoption_df = generate_adoption_data(n_months=48)
    adoption_path = os.path.join(output_dir, "adoption_data.csv")
    adoption_df.to_csv(adoption_path, index=False)
    print(f"Generated and saved adoption data to {adoption_path}")
    
    print("\nAll data generation complete.")
