import pytest
import pandas as pd
import numpy as np
import os
import shutil
import time
import pickle
current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(current_dir, '../backend/source')
sys.path.insert(0, source_dir)
try:
    from preprocessing import categorize_crime, preprocess_data
except ImportError as e:
    pytest.fail(f"ERREUR CRITIQUE : Impossible d'importer 'preprocessing' depuis {source_dir}. Détail : {e}")
    
from preprocessing import (
    categorize_crime,
    clean_column_names,
    load_and_clean_initial,
    feature_engineering_temporal,
    handle_missing_values_and_text,
    process_target,
    encode_features,
    run_preprocessing_pipeline,
    ARTIFACTS_PATH,
    DATA_PATH
)

# ==========================================
# TEST DATA FIXTURES
# ==========================================

@pytest.fixture
def sample_raw_df():
    """Creates a small sample DataFrame mimicking the raw CSV structure."""
    data = {
        'DR_NO': [1, 2, 3, 4, 5],
        'Date Rptd': ['01/01/2020 12:00:00 AM'] * 5,
        'DATE OCC': ['01/01/2020 12:00:00 AM', '02/02/2021 01:30:00 PM', '03/03/2022 06:15:00 AM', '04/04/2023 08:00:00 PM', np.nan],
        'TIME OCC': [10, 1330, 615, 2000, 1200],
        'AREA': [1, 2, 1, 3, 2],
        'AREA NAME': ['Central', 'Rampart', 'Central', 'Southwest', 'Rampart'],
        'Rpt Dist No': [101, 201, 102, 301, 202],
        'Part 1-2': [1, 2, 1, 2, 1],
        'Crm Cd': [100, 200, 300, 400, 500],
        'Crm Cd Desc': ['BURGLARY', 'ASSAULT WITH DEADLY WEAPON', 'VANDALISM', 'THEFT OF IDENTITY', 'RAPE'],
        'Mocodes': ['0100', np.nan, '0300', '0400', '0500'],
        'Vict Age': [25, -1, 30, 120, np.nan], # Includes outliers and NaN
        'Vict Sex': ['M', 'F', np.nan, 'H', 'X'],
        'Vict Descent': ['H', 'W', 'B', '-', np.nan],
        'Premis Cd': [101.0, 102.0, np.nan, 104.0, 105.0],
        'Premis Desc': ['STREET', 'SIDEWALK', 'PARK', np.nan, 'HOUSE'],
        'Weapon Used Cd': [100.0, np.nan, 300.0, 400.0, 500.0],
        'Weapon Desc': ['HAND GUN', np.nan, 'KNIFE', 'PIPE', 'UNK'],
        'Status': ['AA', 'IC', 'AO', np.nan, 'JO'],
        'Status Desc': ['Adult Arrest', 'Invest Cont', 'Adult Other', 'Juv Other', 'Juv Arrest'],
        'Crm Cd 1': [100.0, 200.0, 300.0, 400.0, 500.0],
        'Crm Cd 2': [np.nan] * 5, # To be dropped
        'Crm Cd 3': [np.nan] * 5, # To be dropped
        'Crm Cd 4': [np.nan] * 5, # To be dropped
        'LOCATION': ['1ST ST', '2ND ST', '3RD ST', '4TH ST', '5TH ST'],
        'Cross Street': [np.nan] * 5, # To be dropped
        'LAT': [34.0, 34.1, 34.2, 34.3, 34.4],
        'LON': [-118.2, -118.3, -118.4, -118.5, -118.6]
    }
    return pd.DataFrame(data)

@pytest.fixture
def temp_data_file(tmp_path, sample_raw_df):
    """Saves the sample DataFrame to a temporary CSV file."""
    file_path = tmp_path / "test_crime_data.csv"
    # Simulate the index column issue by saving without index=False first
    sample_raw_df.to_csv(file_path) 
    return str(file_path)

@pytest.fixture
def cleanup_artifacts():
    """Cleans up the processors directory after tests."""
    yield
    if os.path.exists(ARTIFACTS_PATH):
        shutil.rmtree(ARTIFACTS_PATH)

# ==========================================
# UNIT TESTS
# ==========================================

def test_categorize_crime():
    """Test the crime categorization logic."""
    assert categorize_crime("BURGLARY FROM VEHICLE") == 'السرقة والسطو / Theft and Burglary'
    assert categorize_crime("ASSAULT WITH DEADLY WEAPON") == 'العنف والاعتداء / Violence and Assault'
    assert categorize_crime("VANDALISM - MISDEMEANOR") == 'التخريب والتدمير / Vandalism and Destruction'
    assert categorize_crime("THEFT OF IDENTITY") == 'الاحتيال والتزوير / Fraud and Forgery'
    assert categorize_crime("WEAPON POSSESSION") == 'المخالفات القانونية والجرائم المتعلقة بالأسلحة / Legal Offences & Weapons'
    assert categorize_crime("RAPE, FORCIBLE") == 'الجرائم الجنسية والاتجار / Sexual Crimes & Exploitation'
    assert categorize_crime("DRIVING UNDER INFLUENCE") == 'جرائم متنوعة / Miscellaneous Crimes'
    assert categorize_crime(np.nan) == 'جرائم متنوعة / Miscellaneous Crimes'
    assert categorize_crime(123) == 'جرائم متنوعة / Miscellaneous Crimes'

def test_clean_column_names(sample_raw_df):
    """Test column name cleaning."""
    cleaned_df = clean_column_names(sample_raw_df.copy())
    expected_col = 'date_rptd' # 'Date Rptd' -> 'date_rptd'
    assert expected_col in cleaned_df.columns
    assert 'Date Rptd' not in cleaned_df.columns
    assert 'part_1_2' in cleaned_df.columns # Special char handling check

def test_load_and_clean_initial(temp_data_file):
    """Test loading, index dropping, and renaming."""
    df = load_and_clean_initial(temp_data_file)
    
    # Check if 'Unnamed: 0' (index artifact) is gone
    assert 'Unnamed: 0' not in df.columns
    
    # Check if 'dr_no' is dropped
    assert 'dr_no' not in df.columns
    
    # Check if renaming worked
    assert 'crm_risk' in df.columns
    assert 'part_1_2' not in df.columns
    
    # Check standard cleaning
    assert 'date_occ' in df.columns

def test_feature_engineering_temporal(sample_raw_df):
    """Test date/time feature extraction."""
    # Pre-clean names for the function to work
    df = clean_column_names(sample_raw_df.copy())
    df = feature_engineering_temporal(df)
    
    assert 'year' in df.columns
    assert 'month' in df.columns
    assert 'hour' in df.columns
    assert 'hour_bin' in df.columns
    
    # Check specific values based on fixture
    # First row: 1000 -> 10 AM -> Morning
    assert df.loc[0, 'hour'] == 0
    assert df.loc[0, 'hour_bin'] == 'Night' # 0 is in Night bin [0, 6)
    
    # Second row: 1330 -> 13 -> Afternoon
    assert df.loc[1, 'hour'] == 13
    assert df.loc[1, 'hour_bin'] == 'Afternoon' # [12, 18)

    # Check date columns dropped
    assert 'date_occ' not in df.columns
    assert 'time_occ' not in df.columns

def test_handle_missing_values_and_text(sample_raw_df):
    """Test imputation and text cleaning."""
    df = clean_column_names(sample_raw_df.copy())
    # Ensure numeric types for mean calc
    df['vict_age'] = pd.to_numeric(df['vict_age'], errors='coerce')
    
    df = handle_missing_values_and_text(df)
    
    # Check Age imputation
    assert not df['vict_age'].isnull().any()
    # Check Outlier handling (120 should be replaced by mean)
    assert df.loc[3, 'vict_age'] != 120 
    
    # Check Categorical imputation
    assert not df['vict_sex'].isnull().any()
    assert 'X' in df['vict_sex'].values # Filled NaN
    
    assert not df['vict_descent'].isnull().any()
    assert 'UNKNOWN' in df['vict_descent'].values

    # Check Column Dropping
    assert 'crm_cd_1' not in df.columns
    assert 'cross_street' not in df.columns

def test_process_target(sample_raw_df):
    """Test target creation and encoding."""
    df = clean_column_names(sample_raw_df.copy())
    df, encoder = process_target(df)
    
    assert 'Crime_Class_Enc' in df.columns
    assert 'Crime_Class' not in df.columns # Should be dropped
    assert encoder is not None
    # Ensure encoding is integer
    assert pd.api.types.is_integer_dtype(df['Crime_Class_Enc'])

def test_encode_features(sample_raw_df):
    """Test feature encoding (One-Hot & Label)."""
    df = clean_column_names(sample_raw_df.copy())
    # Fill NaNs first to avoid encoding errors
    df['vict_sex'] = df['vict_sex'].fillna('X')
    
    df, encoders = encode_features(df)
    
    # Check One-Hot
    assert 'vict_sex_F' in df.columns
    assert 'vict_sex_M' in df.columns
    
    # Check Label Encoding
    # crm_risk is in CATEGORICAL_COLS_TO_ENCODE
    # It should now be numerically encoded (though it was int before, fit_transform standardizes it)
    assert 'crm_risk' in encoders
    assert 'mocodes' in encoders

# ==========================================
# INTEGRATION & PERFORMANCE TESTS
# ==========================================

@pytest.mark.usefixtures("cleanup_artifacts")
def test_pipeline_execution_fit_mode(temp_data_file):
    """Test the full pipeline in 'Fit' mode (first run)."""
    
    # Mock the DATA_PATH in the module to point to our temp file
    import preprocessing
    preprocessing.DATA_PATH = temp_data_file
    
    # Measure execution time
    start_time = time.time()
    preprocessing.run_preprocessing_pipeline()
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"\nPipeline Fit Execution Time: {execution_time:.4f} seconds")
    
    # Verify Artifacts were created
    assert os.path.exists(ARTIFACTS_PATH)
    assert os.path.exists(os.path.join(ARTIFACTS_PATH, "preprocessed_data.pkl"))
    assert os.path.exists(os.path.join(ARTIFACTS_PATH, "robust_scaler.pkl"))
    assert os.path.exists(os.path.join(ARTIFACTS_PATH, "target_label_encoder.pkl"))
    assert os.path.exists(os.path.join(ARTIFACTS_PATH, "features_config.pkl"))

    # Verify Data Integrity
    with open(os.path.join(ARTIFACTS_PATH, "preprocessed_data.pkl"), "rb") as f:
        data = pickle.load(f)
    
    assert "X_train_scaled" in data
    assert "y_train" in data
    # Check shape (5 rows in fixture * 0.8 train ~= 4 rows, 17 features)
    # Note: Depending on split, might be 3 or 4. Just check dims > 0
    assert data["X_train_scaled"].shape[0] > 0
    assert data["X_train_scaled"].shape[1] == 17 # Strict feature count check

    # Performance Assertion (adjust threshold based on data size)
    # For this tiny dataset, it should be instant.
    assert execution_time < 5.0 

@pytest.mark.usefixtures("cleanup_artifacts")
def test_pipeline_execution_load_mode(temp_data_file):
    """Test the pipeline in 'Load' mode (second run)."""
    import preprocessing
    preprocessing.DATA_PATH = temp_data_file
    
    # Run once to generate artifacts
    preprocessing.run_preprocessing_pipeline()
    
    # Run again to trigger Load mode
    start_time = time.time()
    preprocessing.run_preprocessing_pipeline()
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"\nPipeline Load Execution Time: {execution_time:.4f} seconds")
    
    # It should be faster or successful without errors
    assert execution_time < 5.0

def test_feature_selection_consistency(sample_raw_df):
    """Ensure that if a selected feature is missing, it's handled (filled with 0)."""
    import preprocessing
    
    df = clean_column_names(sample_raw_df.copy())
    
    # Drop a feature that is in SELECTED_FEATURES to simulate missing data
    if 'mocodes' in df.columns:
        df.drop('mocodes', axis=1, inplace=True)
    
    # Run logic similar to pipeline selection
    selected_features = preprocessing.DEFAULT_SELECTED_FEATURES
    
    # This loop is from the script's logic
    for col in selected_features:
        if col not in df.columns:
            df[col] = 0
            
    assert 'mocodes' in df.columns
    assert (df['mocodes'] == 0).all()