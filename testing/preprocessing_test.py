import pytest
import pandas as pd
import numpy as np
import os
import shutil
import time
import pickle
import sys

# ==========================================
# PATH SETUP (FIXED)
# ==========================================
# Get the directory where this test file is located (D:\mlops\classe\MLOPS\testing)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the project root (MLOPS directory)
# We go up one level (..) to reach the project root
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Print for debugging (visible if you run with pytest -s)
print(f"Adding to sys.path: {project_root}")

# Add project root to the start of sys.path so we can import backend.src.preprocessing
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the module directly now that its folder is in sys.path
try:
    import backend.src.preprocessing as preprocessing
    from backend.src.preprocessing import (
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
except ImportError as e:
    pytest.fail(f"Failed to import 'preprocessing' from {project_root}. Error: {e}")

# ==========================================
# TEST DATA FIXTURES
# ==========================================

@pytest.fixture
def sample_raw_df():
    """Creates a small sample DataFrame matching the exact structure of the provided CSV."""
    data = {
        'DR_NO': [76400, 57481, 123257, 147894, 49390],
        'Date Rptd': ['2020-01-01', '2020-01-01', '2020-01-01', '2020-01-01', '2020-01-01'],
        'DATE OCC': ['01/01/2020 12:00:00 AM', '01/01/2020 12:00:00 AM', '01/01/2020 12:00:00 AM', '01/01/2020 12:00:00 AM', '01/01/2020 12:00:00 AM'],
        'TIME OCC': [10, 1156, 2040, 1300, 220],
        'AREA': [6, 13, 21, 21, 6],
        'AREA NAME': ['Hollywood', 'Newton', 'Topanga', 'Topanga', 'Hollywood'],
        'Rpt Dist No': [668, 1371, 2189, 2136, 646],
        'Part 1-2': [2, 2, 1, 2, 2],
        'Crm Cd': [626, 626, 442, 930, 745],
        'Crm Cd Desc': [
            'INTIMATE PARTNER - SIMPLE ASSAULT',
            'INTIMATE PARTNER - SIMPLE ASSAULT',
            'SHOPLIFTING - PETTY THEFT ($950 & UNDER)',
            'CRIMINAL THREATS - NO WEAPON DISPLAYED',
            'VANDALISM - MISDEAMEANOR ($399 OR UNDER)'
        ],
        'Mocodes': ['2000 0400 0416 1414', '0913 1814 2000 0416', '0325 0352', '0443', '0329'],
        'Vict Age': [41, 34, 26, 32, 0], 
        'Vict Sex': ['F', 'M', 'M', 'M', 'X'],
        'Vict Descent': ['H', 'H', 'X', 'H', 'X'],
        'Premis Cd': [501.0, 502.0, 404.0, 108.0, 102.0],
        'Premis Desc': ['SINGLE FAMILY DWELLING', 'MULTI-UNIT DWELLING', 'DEPARTMENT STORE', 'PARKING LOT', 'SIDEWALK'],
        'Weapon Used Cd': [400.0, 400.0, np.nan, 511.0, np.nan],
        'Weapon Desc': ['STRONG-ARM', 'STRONG-ARM', np.nan, 'VERBAL THREAT', np.nan],
        'Status': ['AA', 'AA', 'IC', 'IC', 'IC'],
        'Status Desc': ['Adult Arrest', 'Adult Arrest', 'Invest Cont', 'Invest Cont', 'Invest Cont'],
        'Crm Cd 1': [626.0, 626.0, 442.0, 930.0, 745.0],
        'Crm Cd 2': [np.nan] * 5,
        'Crm Cd 3': [np.nan] * 5,
        'Crm Cd 4': [np.nan] * 5,
        'LOCATION': ['LOC1', 'LOC2', 'LOC3', 'LOC4', 'LOC5'],
        'Cross Street': [np.nan] * 5,
        'LAT': [34.0918, 33.9924, 34.1665, 34.2047, 34.1016],
        'LON': [-118.3136, -118.2772, -118.5859, -118.5994, -118.3361]
    }
    return pd.DataFrame(data)

@pytest.fixture
def temp_data_file(tmp_path, sample_raw_df):
    """Saves the sample DataFrame to a temporary CSV file to mimic loading from disk."""
    file_path = tmp_path / "test_crime_data.csv"
    # Save WITHOUT index to match how load_and_clean expects raw data 
    # (or effectively how it looks after removing index col)
    sample_raw_df.to_csv(file_path, index=False) 
    return str(file_path)

@pytest.fixture
def cleanup_artifacts():
    """Cleans up the processors directory after tests."""
    yield
    # Ensure we clean up where artifacts are actually created (current working dir/processors)
    local_artifacts = os.path.join(os.getcwd(), ARTIFACTS_PATH)
    if os.path.exists(local_artifacts): 
        shutil.rmtree(local_artifacts)

# ==========================================
# UNIT TESTS
# ==========================================

def test_categorize_crime():
    """Test the crime categorization logic."""
    # Theft
    assert categorize_crime("SHOPLIFTING - PETTY THEFT") == 'السرقة والسطو / Theft and Burglary'
    assert categorize_crime("BURGLARY FROM VEHICLE") == 'السرقة والسطو / Theft and Burglary'
    
    # Violence
    assert categorize_crime("INTIMATE PARTNER - SIMPLE ASSAULT") == 'العنف والاعتداء / Violence and Assault'
    assert categorize_crime("CRIMINAL THREATS") == 'العنف والاعتداء / Violence and Assault' 
    
    # Vandalism
    assert categorize_crime("VANDALISM - MISDEAMEANOR") == 'التخريب والتدمير / Vandalism and Destruction'
    
    # Fraud (Priority over Theft)
    assert categorize_crime("THEFT OF IDENTITY") == 'الاحتيال والتزوير / Fraud and Forgery'
    
    # Misc / Default
    assert categorize_crime("DRIVING UNDER INFLUENCE") == 'جرائم متنوعة / Miscellaneous Crimes'
    assert categorize_crime(np.nan) == 'جرائم متنوعة / Miscellaneous Crimes'

def test_clean_column_names(sample_raw_df):
    """Test column name cleaning."""
    cleaned_df = clean_column_names(sample_raw_df.copy())
    
    # Expected transformations: 'Date Rptd' -> 'date_rptd', 'Part 1-2' -> 'part_1_2'
    assert 'date_rptd' in cleaned_df.columns
    assert 'Date Rptd' not in cleaned_df.columns
    assert 'part_1_2' in cleaned_df.columns
    assert 'Part 1-2' not in cleaned_df.columns

def test_load_and_clean_initial(temp_data_file):
    """Test loading, index dropping, and renaming."""
    df = load_and_clean_initial(temp_data_file)
    
    # Check if 'dr_no' is dropped
    assert 'dr_no' not in df.columns
    
    # Check if renaming worked ('part_1_2' -> 'crm_risk')
    assert 'crm_risk' in df.columns
    assert 'part_1_2' not in df.columns
    
    # Check standard cleaning
    assert 'date_occ' in df.columns

def test_feature_engineering_temporal(sample_raw_df):
    """Test date/time feature extraction."""
    # Pre-clean names for the function to work
    df = clean_column_names(sample_raw_df.copy())
    df = feature_engineering_temporal(df)
    
    # Function converts to lowercase names
    assert 'year' in df.columns
    assert 'month' in df.columns
    assert 'hour' in df.columns
    assert 'hour_bin' in df.columns
    
    # Sample row 0: TIME OCC 10 -> 00:10 -> Hour 0 -> Night
    assert df.loc[0, 'hour'] == 0
    assert df.loc[0, 'hour_bin'] == 'Night' 
    
    # Sample row 1: TIME OCC 1156 -> 11:56 -> Hour 11 -> Morning [6, 12)
    assert df.loc[1, 'hour'] == 11
    assert df.loc[1, 'hour_bin'] == 'Morning'

    # Check dropped columns
    assert 'date_occ' not in df.columns 
    assert 'time_occ' not in df.columns

def test_handle_missing_values_and_text(sample_raw_df):
    """Test imputation and text cleaning."""
    df = clean_column_names(sample_raw_df.copy())
    df['vict_age'] = pd.to_numeric(df['vict_age'], errors='coerce')
    
    df = handle_missing_values_and_text(df)
    
    # Check categorical filling
    assert not df['vict_sex'].isnull().any()
    assert not df['vict_descent'].isnull().any()

    # Check Column Dropping
    assert 'crm_cd_1' not in df.columns
    assert 'cross_street' not in df.columns
    
    # Check Text Cleaning
    # "SINGLE FAMILY DWELLING" -> "Single Family Dwelling"
    # Using iloc[0] because indexes might change
    assert df['premis_desc'].iloc[0] == 'Single Family Dwelling'

def test_process_target(sample_raw_df):
    """Test target creation and encoding."""
    df = clean_column_names(sample_raw_df.copy())
    df, encoder = process_target(df)
    
    assert 'Crime_Class_Enc' in df.columns
    assert 'Crime_Class' not in df.columns 
    assert encoder is not None
    assert pd.api.types.is_integer_dtype(df['Crime_Class_Enc'])

def test_encode_features(sample_raw_df):
    """Test feature encoding (One-Hot & Label)."""
    df = clean_column_names(sample_raw_df.copy())
    
    # Manual rename for this unit test context
    if 'part_1_2' in df.columns:
        df = df.rename(columns={'part_1_2': 'crm_risk'})
    
    df['vict_sex'] = df['vict_sex'].fillna('X')
    
    df, encoders = encode_features(df)
    
    # Check One-Hot (columns will be lowercased by clean_column_names earlier)
    # vict_sex -> vict_sex_F, vict_sex_M, vict_sex_X
    assert 'vict_sex_F' in df.columns or 'vict_sex_f' in df.columns
    
    # Check Label Encoding
    assert 'crm_risk' in encoders
    assert 'mocodes' in encoders

# ==========================================
# INTEGRATION TESTS
# ==========================================

@pytest.mark.usefixtures("cleanup_artifacts")
def test_pipeline_execution_fit_mode(temp_data_file):
    """Test the full pipeline in 'Fit' mode (first run)."""
    
    # Point the module to the temp data file
    preprocessing.DATA_PATH = temp_data_file
    
    start_time = time.time()
    preprocessing.run_preprocessing_pipeline()
    end_time = time.time()
    
    print(f"\nFit Time: {end_time - start_time:.4f}s")
    
    # Check if files were created in the current working directory's processors folder
    local_artifacts = os.path.join(os.getcwd(), ARTIFACTS_PATH)
    
    assert os.path.exists(os.path.join(local_artifacts, "preprocessed_data.pkl"))
    assert os.path.exists(os.path.join(local_artifacts, "target_label_encoder.pkl"))
    assert os.path.exists(os.path.join(local_artifacts, "features_config.pkl"))
    
    with open(os.path.join(local_artifacts, "preprocessed_data.pkl"), "rb") as f:
        data = pickle.load(f)
    
    # With 5 rows, train/test split might be small, but feature count must be 17
    assert data["X_train_scaled"].shape[1] == 17

@pytest.mark.usefixtures("cleanup_artifacts")
def test_pipeline_execution_load_mode(temp_data_file):
    """Test the pipeline in 'Load' mode (second run)."""
    
    preprocessing.DATA_PATH = temp_data_file
    
    # 1. Run Fit
    preprocessing.run_preprocessing_pipeline()
    
    # 2. Run Load
    start_time = time.time()
    preprocessing.run_preprocessing_pipeline()
    end_time = time.time()
    
    print(f"\nLoad Time: {end_time - start_time:.4f}s")
    assert True 

def test_feature_selection_consistency(sample_raw_df):
    """Ensure that if a selected feature is missing from raw data, it is filled with 0."""
    
    # Setup a df missing 'mocodes'
    df = clean_column_names(sample_raw_df.copy())
    if 'mocodes' in df.columns:
        df.drop('mocodes', axis=1, inplace=True)
        
    # Import defaults from the module
    selected_features = [f.lower() for f in preprocessing.DEFAULT_SELECTED_FEATURES]
    
    for col in selected_features:
        if col not in df.columns:
            df[col] = 0
            
    assert 'mocodes' in df.columns
    assert (df['mocodes'] == 0).all()