# Part 1 – Data Exploration (Functions Module)
# Libraries needed: pandas, numpy, matplotlib, seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options for better output
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-whitegrid')


def load_dataset(filepath):
    """
    Task 1: Load the Heart Disease dataset (0.25pt)
    """
    print("=" * 60)
    print("1. LOADING DATASET")
    print("=" * 60)
    
    df = pd.read_csv(filepath)
    print(f"Dataset loaded successfully from: {filepath}")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df


def display_dataset_info(df):
    """
    Task 2: Display dataset information (0.5pt)
    - First rows of the dataset
    - Dataset shape (number of rows and columns)
    - Column names and data types
    """
    print("\n" + "=" * 60)
    print("2. DATASET INFORMATION")
    print("=" * 60)
    
    # First rows of the dataset
    print("\n--- First 5 rows of the dataset ---")
    print(df.head())
    
    # Dataset shape
    print(f"\n--- Dataset Shape ---")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    
    # Column names and data types
    print("\n--- Column Names and Data Types ---")
    print(df.dtypes)


def remove_target_column(df, target_column='target'):
    """
    Task 3: Remove the target column (0.25pt)
    Modifies the dataframe in-place and returns it
    """
    print("\n" + "=" * 60)
    print("3. REMOVING TARGET COLUMN")
    print("=" * 60)
    
    print(f"Target column to be deleted: '{target_column}'")
    print(f"Shape BEFORE removing target: {df.shape}")
    
    # Remove the target column in-place
    df.drop(columns=[target_column], inplace=True)
    
    print(f"Shape AFTER removing target: {df.shape}")
    print(f"Remaining columns: {df.columns.tolist()}")
    
    return df


def compute_basic_statistics(df):
    """
    Task 4: Compute basic statistics (0.5pt)
    - mean, standard deviation, minimum, maximum
    """
    print("\n" + "=" * 60)
    print("4. BASIC STATISTICS")
    print("=" * 60)
    
    # mean is the average value
    # std is the standard deviation
    # min is the minimum value
    # max is the maximum value
    # count is the number of non-missing values
    print("\n--- Basic Statistics (mean, std, min, max, count) ---")
    stats = df.describe().loc[['mean', 'std', 'min', 'max', 'count']]
    print(stats)
    
    return stats


def visualize_distributions(df):
    """
    Task 5: Analyze feature distributions and visualize them (0.5pt)
    - Histograms
    - Box plots
    - Scatter plots (for selected features)
    Opens 3 interactive windows simultaneously
    """
    print("\n" + "=" * 60)
    print("5. FEATURE DISTRIBUTION VISUALIZATIONS")
    print("=" * 60)
    
    # Use interactive backend for better visualization with zoom/scroll
    plt.ion()  # Turn on interactive mode
    
    # --- Figure 1: Histograms ---
    print("\nGenerating Histograms (Window 1)...")
    fig1, axes1 = plt.subplots(nrows=4, ncols=4, figsize=(14, 10), num='Histograms - Window 1')
    axes1 = axes1.flatten()
    
    for i, col in enumerate(df.columns):
        if i < len(axes1):
            axes1[i].hist(df[col], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
            axes1[i].set_title(f'{col}', fontsize=10, fontweight='bold')
            axes1[i].set_xlabel(col, fontsize=8)
            axes1[i].set_ylabel('Frequency', fontsize=8)
            axes1[i].tick_params(axis='both', labelsize=7)
    
    # Hide empty subplots
    for j in range(i + 1, len(axes1)):
        axes1[j].set_visible(False)
    
    fig1.suptitle('Histograms of Features (Use toolbar to zoom/pan)', fontsize=14, fontweight='bold')
    fig1.tight_layout()
    fig1.savefig('histograms.png', dpi=150)
    
    # --- Figure 2: Box Plots ---
    print("Generating Box Plots (Window 2)...")
    fig2, axes2 = plt.subplots(nrows=4, ncols=4, figsize=(14, 10), num='Box Plots - Window 2')
    axes2 = axes2.flatten()
    
    colors = sns.color_palette("husl", len(df.columns))
    for i, col in enumerate(df.columns):
        if i < len(axes2):
            sns.boxplot(y=df[col], ax=axes2[i], color=colors[i])
            axes2[i].set_title(f'{col}', fontsize=10, fontweight='bold')
            axes2[i].set_ylabel('')
            axes2[i].tick_params(axis='both', labelsize=7)
    
    # Hide empty subplots
    for j in range(i + 1, len(axes2)):
        axes2[j].set_visible(False)
    
    fig2.suptitle('Box Plots of Features (Use toolbar to zoom/pan)', fontsize=14, fontweight='bold')
    fig2.tight_layout()
    fig2.savefig('boxplots.png', dpi=150)
    
    # --- Figure 3: Scatter Plots (for selected features) ---
    print("Generating Scatter Plots (Window 3)...")
    
    # Select some important features for scatter plots
    selected_features = ['age', 'trestbps', 'chol', 'thalach']
    selected_features = [f for f in selected_features if f in df.columns]
    
    if len(selected_features) >= 2:
        fig3, axes3 = plt.subplots(nrows=2, ncols=3, figsize=(14, 9), num='Scatter Plots - Window 3')
        axes3 = axes3.flatten()
        
        scatter_colors = sns.color_palette("viridis", 6)
        plot_idx = 0
        for i in range(len(selected_features)):
            for j in range(i + 1, len(selected_features)):
                if plot_idx < len(axes3):
                    axes3[plot_idx].scatter(df[selected_features[i]], 
                                            df[selected_features[j]], 
                                            alpha=0.6, edgecolors='white', linewidth=0.3,
                                            c=[scatter_colors[plot_idx]], s=30)
                    axes3[plot_idx].set_xlabel(selected_features[i], fontsize=9)
                    axes3[plot_idx].set_ylabel(selected_features[j], fontsize=9)
                    axes3[plot_idx].set_title(f'{selected_features[i]} vs {selected_features[j]}', 
                                              fontsize=10, fontweight='bold')
                    axes3[plot_idx].tick_params(axis='both', labelsize=8)
                    plot_idx += 1
        
        # Hide empty subplots
        for k in range(plot_idx, len(axes3)):
            axes3[k].set_visible(False)
        
        fig3.suptitle('Scatter Plots of Selected Features (Use toolbar to zoom/pan)', fontsize=14, fontweight='bold')
        fig3.tight_layout()
        fig3.savefig('scatterplots.png', dpi=150)
    
    # Show all 3 windows simultaneously
    print("\n" + "-" * 60)
    print("3 VISUALIZATION WINDOWS ARE NOW OPEN!")
    print("-" * 60)
    print("Tips for interacting with the plots:")
    print("  - Use the toolbar at the bottom of each window")
    print("  - Pan tool: Click and drag to move around")
    print("  - Zoom tool: Draw a rectangle to zoom in")
    print("  - Home button: Reset to original view")
    print("  - Save button: Save the current view")
    print("-" * 60)
    
    plt.show(block=True)  # Block to keep all windows open


def handle_missing_values(df):
    """
    Task 6: Identify missing values and remove them if found (0.25pt)
    Modifies the dataframe in-place and returns it
    """
    print("\n" + "=" * 60)
    print("6. MISSING VALUES ANALYSIS")
    print("=" * 60)
    
    # Check for missing values
    print("\n--- Missing Values per Column ---")
    missing_values = df.isnull().sum()
    print(missing_values)
    
    print(f"\n--- Total Missing Values: {missing_values.sum()} ---")
    
    # Remove missing values if found (in-place modification)
    if missing_values.sum() > 0:
        print(f"\nShape BEFORE removing missing values: {df.shape}")
        rows_before = df.shape[0]
        
        # Remove rows with missing values
        df.dropna(inplace=True)
        
        print(f"Shape AFTER removing missing values: {df.shape}")
        print(f"Rows removed: {rows_before - df.shape[0]}")
    else:
        print("\nNo missing values found in the dataset!")
    
    return df


def save_cleaned_data(df, output_filepath='heart_cleaned.csv'):
    """
    Save the cleaned dataset to a new CSV file
    """
    print("\n" + "=" * 60)
    print("SAVING CLEANED DATA")
    print("=" * 60)
    
    df.to_csv(output_filepath, index=False)
    print(f"Cleaned dataset saved to: '{output_filepath}'")
    
    return output_filepath


def print_summary(original_shape, df):
    """
    Print summary of Part 1 exploration
    """
    print("\n" + "=" * 60)
    print("SUMMARY - Part 1 Complete!")
    print("=" * 60)
    print(f"Original dataset: heart.csv ({original_shape[0]} rows, {original_shape[1]} columns)")
    print(f"Cleaned dataset: heart_cleaned.csv ({df.shape[0]} rows, {df.shape[1]} columns)")
    print("\nModifications made to the data:")
    print("  ✓ Target column removed")
    print("  ✓ Missing values removed (if any)")
    print("\nVisualization files saved:")
    print("  - histograms.png")
    print("  - boxplots.png")
    print("  - scatterplots.png")
    print("\nThe cleaned data is now ready for Part 2!")


def run_part_one(filepath='heart.csv'):
    """
    Main function to run all Part 1 tasks
    Returns the cleaned dataframe
    """
    # Task 1: Load dataset
    df = load_dataset(filepath)
    original_shape = df.shape
    
    # Task 2: Display dataset info
    display_dataset_info(df)
    
    # Task 3: Remove target column
    df = remove_target_column(df, target_column='target')
    
    # Task 4: Compute statistics
    compute_basic_statistics(df)
    
    # Task 5: Visualize distributions
    visualize_distributions(df)
    
    # Task 6: Handle missing values
    df = handle_missing_values(df)
    
    # Save cleaned data
    save_cleaned_data(df, 'heart_cleaned.csv')
    
    # Print summary
    print_summary(original_shape, df)
    
    return df
