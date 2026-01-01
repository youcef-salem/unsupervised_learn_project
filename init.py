# Main execution file for Heart Disease Analysis
# This file imports and runs functions from part_one.py, part_two.py, and part_three.py

from part_one import run_part_one
from part_two import run_part_two, load_cleaned_data
from part_three import run_part_three

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("        HEART DISEASE ANALYSIS PROJECT")
    print("=" * 60)
    
    # PART 1 - DATA EXPLORATION
    print("\n>>> Starting Part 1: Data Exploration...\n")
    df_cleaned = run_part_one('heart.csv')
    
    # PART 2 - MODEL TRAINING
    print("\n>>> Starting Part 2: Model Training...\n")
    kmeans, train_labels, test_labels, X_train_scaled, X_test_scaled, scaler = run_part_two('heart_cleaned.csv')
    
    # PART 3 - EVALUATION
    print("\n>>> Starting Part 3: Evaluation...\n")
    df = load_cleaned_data('heart_cleaned.csv')
    feature_names = df.columns.tolist()
    metrics_df = run_part_three(kmeans, train_labels, X_train_scaled, feature_names)

