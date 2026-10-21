import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# SECTION 1 – LOAD RAW DATA

def load_raw(file_path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        print(f"Error: File not found — {file_path}")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


# SECTION 2 – COLUMN NORMALISATION
# Handles datasets where column names may differ across sources.

COLUMN_MAP = {
    'Exam_Final'   : 'Final_Exam',
    'Exam_Mid_Term': 'Midterm_Exam',
    'Gpa_Previous' : 'Previous_GPA',
}

def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {src: dst for src, dst in COLUMN_MAP.items()
              if src in df.columns and src != dst and dst not in df.columns}
    if rename:
        df = df.rename(columns=rename)
    return df


# SECTION 3 – DUPLICATE REMOVAL

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    id_col = next((c for c in ['Student_ID', 'Registration_No'] if c in df.columns), None)
    if id_col:
        df = df.drop_duplicates(subset=[id_col], keep='first')
    return df.reset_index(drop=True)


# SECTION 4 – MISSING VALUE HANDLING
# Numeric columns → median imputation
# Categorical columns → mode imputation

NUMERIC_SCORE_COLS = [
    'Final_Exam', 'Midterm_Exam', 'Average_Assignment',
    'Quiz_1', 'Quiz_2', 'Average_Quiz',
    'Lab_Work_1', 'Lab_Work_2', 'Average_Lab', 'Project_Score',
    'Assignment_1', 'Assignment_2', 'Assignment_3',
    'Attendance_Percentage', 'Study_Hours_Per_Week',
    'Library_Visits', 'Online_Course_Hours',
    'Class_Participation', 'Group_Work_Score',
    'Total_Score', 'CGPA', 'Previous_GPA',
]

CATEGORICAL_COLS = ['Gender', 'Course', 'Section', 'Sports_Activity', 'Club_Membership']

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_SCORE_COLS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


# SECTION 5 – DERIVED FEATURES

def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    if 'Quiz_1' in df.columns and 'Quiz_2' in df.columns:
        df['Average_Quiz'] = df[['Quiz_1', 'Quiz_2']].mean(axis=1)

    if all(c in df.columns for c in ['Assignment_1', 'Assignment_2', 'Assignment_3']):
        df['Average_Assignment'] = df[['Assignment_1', 'Assignment_2', 'Assignment_3']].mean(axis=1)

    if 'Lab_Work_1' in df.columns and 'Lab_Work_2' in df.columns:
        df['Average_Lab'] = df[['Lab_Work_1', 'Lab_Work_2']].mean(axis=1)

    return df


# SECTION 6 – PERFORMANCE LABEL
# Weighted formula: Final(40%) + Midterm(20%) + Assignment(20%) + Quiz(20%)

def compute_performance(row: pd.Series) -> str:
    score = (
        row['Final_Exam']         * 0.4 +
        row['Midterm_Exam']       * 0.2 +
        row['Average_Assignment'] * 0.2 +
        row['Average_Quiz']       * 0.2
    )
    if score >= 85:   return 'Excellent'
    elif score >= 75: return 'Very Good'
    elif score >= 60: return 'Good'
    elif score >= 50: return 'Average'
    else:             return 'At-Risk'


def assign_performance(df: pd.DataFrame) -> pd.DataFrame:
    required = ['Final_Exam', 'Midterm_Exam', 'Average_Assignment', 'Average_Quiz']
    if all(c in df.columns for c in required):
        df['Performance'] = df.apply(compute_performance, axis=1)
    return df


# SECTION 7 – MAIN PIPELINE

def run_pipeline(file_path: str = 'student_data.csv'):
    print("=" * 50)
    print("   PHASE 1: DATA PREPROCESSING & ML SPLITTING")
    print("=" * 50)

    # Load
    df = load_raw(file_path)
    if df is None:
        return
    print(f"  Dataset loaded       : {len(df)} records, {len(df.columns)} features")

    # Clean
    df = normalise_columns(df)
    df = remove_duplicates(df)
    print(f"  After deduplication  : {len(df)} records")

    df = handle_missing(df)
    print(f"  Missing values       : {df.isnull().sum().sum()} remaining")

    # Engineer
    df = derive_features(df)
    df = assign_performance(df)

    # Save processed file
    df.to_csv('processed_student_data.csv', index=False)

    # Train / Test split
    features = ['Final_Exam', 'Midterm_Exam', 'Average_Assignment', 'Average_Quiz']
    X = df[features]
    y = df['Performance']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv',   index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv',   index=False)

    print(f"  Train set            : {len(X_train)} records (80%)")
    print(f"  Test set             : {len(X_test)} records (20%)")
    print(f"  Performance classes  : {sorted(y.unique())}")
    print("=" * 50)
    print("  Phase 1 completed successfully.")
    print("=" * 50)


if __name__ == "__main__":
    run_pipeline('student_data.csv')