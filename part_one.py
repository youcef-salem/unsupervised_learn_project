# Part 1 – Data Exploration
# Libraries: pandas, numpy, matplotlib, seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-whitegrid')


def load_dataset(filepath):
    """Task 1: Load the Heart Disease dataset (0.25pt)"""
    print("=" * 60)
    print("1. LOADING DATASET")
    print("=" * 60)
    
    df = pd.read_csv(filepath)
    print(f"Dataset loaded from: {filepath}")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def display_dataset_info(df):
    """Task 2: Display dataset information (0.5pt)"""
    print("\n" + "=" * 60)
    print("2. DATASET INFORMATION")
    print("=" * 60)
    
    print("\n--- First 5 rows ---")
    print(df.head())
    
    print(f"\n--- Shape ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    print("\n--- Column Names and Data Types ---")
    print(df.dtypes)


def remove_target_column(df, target_column='target'):
    """Task 3: Remove the target column (0.25pt)"""
    print("\n" + "=" * 60)
    print("3. REMOVING TARGET COLUMN")
    print("=" * 60)
    
    print(f"Removing: '{target_column}'")
    print(f"Shape BEFORE: {df.shape}")
    df.drop(columns=[target_column], inplace=True)
    print(f"Shape AFTER: {df.shape}")
    return df


def compute_basic_statistics(df):
    """Task 4: Compute basic statistics (0.5pt)"""
    print("\n" + "=" * 60)
    print("4. BASIC STATISTICS")
    print("=" * 60)
    
    stats = df.describe().loc[['mean', 'std', 'min', 'max', 'count']]
    print(stats)
    return stats


def visualize_distributions(df):
    """Task 5: Visualize distributions (0.5pt) - Histograms, Box plots, Scatter plots"""
    print("\n" + "=" * 60)
    print("5. FEATURE DISTRIBUTION VISUALIZATIONS")
    print("=" * 60)
    
    plt.ion()
    
    # Histograms
    print("\nGenerating Histograms...")
    fig1, axes1 = plt.subplots(nrows=4, ncols=4, figsize=(14, 10), num='Histograms')
    axes1 = axes1.flatten()
    for i, col in enumerate(df.columns):
        if i < len(axes1):
            axes1[i].hist(df[col], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
            axes1[i].set_title(col, fontsize=10, fontweight='bold')
    for j in range(i + 1, len(axes1)):
        axes1[j].set_visible(False)
    fig1.suptitle('Histograms of Features', fontsize=14, fontweight='bold')
    fig1.tight_layout()
    fig1.savefig('histograms.png', dpi=150)
    
    # Box Plots
    print("Generating Box Plots...")
    fig2, axes2 = plt.subplots(nrows=4, ncols=4, figsize=(14, 10), num='Box Plots')
    axes2 = axes2.flatten()
    colors = sns.color_palette("husl", len(df.columns))
    for i, col in enumerate(df.columns):
        if i < len(axes2):
            sns.boxplot(y=df[col], ax=axes2[i], color=colors[i])
            axes2[i].set_title(col, fontsize=10, fontweight='bold')
    for j in range(i + 1, len(axes2)):
        axes2[j].set_visible(False)
    fig2.suptitle('Box Plots of Features', fontsize=14, fontweight='bold')
    fig2.tight_layout()
    fig2.savefig('boxplots.png', dpi=150)
    
    # Scatter Plots
    print("Generating Scatter Plots...")
    selected = ['age', 'trestbps', 'chol', 'thalach']
    selected = [f for f in selected if f in df.columns]
    if len(selected) >= 2:
        fig3, axes3 = plt.subplots(nrows=2, ncols=3, figsize=(14, 9), num='Scatter Plots')
        axes3 = axes3.flatten()
        plot_idx = 0
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                if plot_idx < len(axes3):
                    axes3[plot_idx].scatter(df[selected[i]], df[selected[j]], alpha=0.6, s=30)
                    axes3[plot_idx].set_xlabel(selected[i])
                    axes3[plot_idx].set_ylabel(selected[j])
                    axes3[plot_idx].set_title(f'{selected[i]} vs {selected[j]}', fontweight='bold')
                    plot_idx += 1
        for k in range(plot_idx, len(axes3)):
            axes3[k].set_visible(False)
        fig3.suptitle('Scatter Plots of Selected Features', fontsize=14, fontweight='bold')
        fig3.tight_layout()
        fig3.savefig('scatterplots.png', dpi=150)
    
    plt.show(block=True)


def handle_missing_values(df):
    """Task 6: Identify and remove missing values (0.25pt)"""
    print("\n" + "=" * 60)
    print("6. MISSING VALUES")
    print("=" * 60)
    
    missing = df.isnull().sum()
    print(f"Missing values per column:\n{missing}")
    print(f"Total missing: {missing.sum()}")
    
    if missing.sum() > 0:
        rows_before = df.shape[0]
        df.dropna(inplace=True)
        print(f"Rows removed: {rows_before - df.shape[0]}")
    else:
        print("No missing values found!")
    return df


def remove_irrelevant_columns(df):
    """Task 7: Remove irrelevant columns (0.25pt)"""
    print("\n" + "=" * 60)
    print("7. REMOVING IRRELEVANT COLUMNS")
    print("=" * 60)
    
    # Check for patient identifiers or irrelevant columns
    irrelevant = ['id', 'patient_id', 'name', 'index']
    cols_to_remove = [col for col in irrelevant if col in df.columns]
    
    if cols_to_remove:
        print(f"Removing: {cols_to_remove}")
        df.drop(columns=cols_to_remove, inplace=True)
    else:
        print("No irrelevant columns found (no patient identifiers)")
    print(f"Remaining columns: {df.columns.tolist()}")
    return df


def check_duplicates(df):
    """Task 8: Check for duplicates and remove them (0.25pt)"""
    print("\n" + "=" * 60)
    print("8. CHECKING DUPLICATES")
    print("=" * 60)
    
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows found: {duplicates}")
    
    if duplicates > 0:
        print(f"Shape BEFORE: {df.shape}")
        df.drop_duplicates(inplace=True)
        print(f"Shape AFTER: {df.shape}")
    else:
        print("No duplicates to remove")
    return df


def encode_categorical(df):
    """Task 9: Encode categorical variables with One Hot Encoding (0.25pt)"""
    print("\n" + "=" * 60)
    print("9. ONE HOT ENCODING")
    print("=" * 60)
    
    # Identify categorical columns (object type or low cardinality integers)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Also check for categorical-like integer columns (cp, restecg, slope, ca, thal)
    cat_like = ['cp', 'restecg', 'slope', 'ca', 'thal', 'sex', 'fbs', 'exang']
    cat_like = [c for c in cat_like if c in df.columns]
    
    print(f"Categorical columns identified: {cat_like}")
    
    if cat_like:
        print(f"Shape BEFORE encoding: {df.shape}")
        df = pd.get_dummies(df, columns=cat_like, drop_first=False)
        print(f"Shape AFTER encoding: {df.shape}")
    else:
        print("No categorical columns to encode")
    
    return df


def select_features(df):
    """Task 10: Select features for clustering (0.25pt)"""
    print("\n" + "=" * 60)
    print("10. FEATURE SELECTION")
    print("=" * 60)
    
    # All remaining columns are used for clustering
    features = df.columns.tolist()
    print(f"Selected {len(features)} features for clustering:")
    print(f"  {features}")
    return df


def normalize_features(df):
    """Task 11: Normalize numerical features (0.25pt)"""
    print("\n" + "=" * 60)
    print("11. NORMALIZING FEATURES")
    print("=" * 60)
    
    from sklearn.preprocessing import StandardScaler
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Normalizing {len(numerical_cols)} numerical features using StandardScaler")
    
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    print("Normalization complete (mean=0, std=1)")
    return df, scaler


def save_cleaned_data(df, output_filepath='heart_cleaned.csv'):
    """Save cleaned dataset"""
    print("\n" + "=" * 60)
    print("SAVING CLEANED DATA")
    print("=" * 60)
    df.to_csv(output_filepath, index=False)
    print(f"Saved to: '{output_filepath}'")
    return output_filepath


def run_part_one(filepath='heart.csv'):
    """Main function to run all Part 1 tasks"""
    # Task 1: Load dataset
    df = load_dataset(filepath)
    original_shape = df.shape
    
    # Task 2: Display info
    display_dataset_info(df)
    
    # Task 3: Remove target
    df = remove_target_column(df, 'target')
    
    # Task 4: Statistics
    compute_basic_statistics(df)
    
    # Task 5: Visualizations
    visualize_distributions(df)
    
    # Task 6: Missing values
    df = handle_missing_values(df)
    
    # Task 7: Remove irrelevant columns
    df = remove_irrelevant_columns(df)
    
    # Task 8: Duplicates
    df = check_duplicates(df)
    
    # Task 9: One Hot Encoding
    df = encode_categorical(df)
    
    # Task 10: Feature selection
    df = select_features(df)
    
    # Task 11: Normalize (note: will be done again in Part 2 properly)
    # df, scaler = normalize_features(df)  # Commented: normalization done in Part 2
    
    # Save
    save_cleaned_data(df, 'heart_cleaned.csv')
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - Part 1 Complete!")
    print("=" * 60)
    print(f"Original: {original_shape[0]} rows, {original_shape[1]} columns")
    print(f"Cleaned: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df
