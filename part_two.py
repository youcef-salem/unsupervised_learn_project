# Part 2 – Model Training (Functions Module)
# Libraries needed: pandas, numpy, matplotlib, sklearn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set display options
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-whitegrid')


def load_cleaned_data(filepath='heart_cleaned.csv'):
    """
    Load the cleaned dataset from Part 1
    """
    print("=" * 60)
    print("LOADING CLEANED DATA")
    print("=" * 60)
    
    df = pd.read_csv(filepath)
    print(f"Dataset loaded from: {filepath}")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df


def split_dataset(df, test_size=0.2, random_state=42):
    """
    Task 1: Split the dataset into training (80%) and test (20%) sets (0.5pt)
    Also explains why splitting is useful in unsupervised learning
    """
    print("\n" + "=" * 60)
    print("1. SPLITTING DATASET (80% Train / 20% Test)")
    print("=" * 60)
    
    # Split the data
    X_train, X_test = train_test_split(df, test_size=test_size, random_state=random_state)
    
    print(f"\nOriginal dataset size: {len(df)}")
    print(f"Training set size: {len(X_train)} ({100*(1-test_size):.0f}%)")
    print(f"Test set size: {len(X_test)} ({100*test_size:.0f}%)")
    
    # Explanation of why splitting is useful in unsupervised learning
    print("\n" + "-" * 60)
    print("WHY SPLITTING IS USEFUL IN UNSUPERVISED LEARNING:")
    print("-" * 60)
    print("""
    Even though unsupervised learning has no labels, splitting is useful for:
    
    1. GENERALIZATION TESTING:
       - Check if clusters found in training data apply to unseen test data
       - Ensures the model doesn't overfit to specific data patterns
    
    2. MODEL VALIDATION:
       - Evaluate clustering stability and consistency
       - Compare cluster assignments between train and test sets
    
    3. HYPERPARAMETER TUNING:
       - Use training set to determine optimal k (number of clusters)
       - Validate the choice on the test set
    
    4. AVOIDING DATA LEAKAGE:
       - Preprocessing (like scaling) should be fit on training data only
       - Then applied to test data to simulate real-world scenarios
    
    5. QUALITY METRICS:
       - Calculate metrics (silhouette score, inertia) on test set
       - Provides unbiased evaluation of clustering quality
    """)
    
    return X_train, X_test


def standardize_data(X_train, X_test):
    """
    Standardize the features (important for K-Means)
    Fit scaler on training data, transform both train and test
    """
    print("\n" + "=" * 60)
    print("STANDARDIZING DATA")
    print("=" * 60)
    
    scaler = StandardScaler()
    
    # Fit on training data only, then transform both
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Data standardized using StandardScaler")
    print("  - Scaler fitted on training data only")
    print("  - Both train and test data transformed")
    print(f"  - Training set shape: {X_train_scaled.shape}")
    print(f"  - Test set shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, scaler


def apply_elbow_method(X_train_scaled, k_range=range(1, 11)):
    """
    Task 2: Apply the Elbow Method to choose optimal k 
    """
    print("\n" + "=" * 60)
    print("2. ELBOW METHOD - Finding Optimal Number of Clusters")
    print("=" * 60)
    
    inertias = []
    k_values = list(k_range)
    
    print("\nCalculating inertia for different k values...")
    print("-" * 40)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_train_scaled)
        inertias.append(kmeans.inertia_)
        print(f"  k = {k:2d} | Inertia = {kmeans.inertia_:.2f}")
    
    # Plot the Elbow curve
    plt.figure(figsize=(10, 6), num='Elbow Method')
    plt.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Inertia (Within-cluster Sum of Squares)', fontsize=12)
    plt.title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
    plt.xticks(k_values)
    plt.grid(True, alpha=0.3)
    
    # Mark the elbow point (typically k=3 or k=4 for heart disease data)
    # Calculate the rate of change to find elbow
    differences = np.diff(inertias)
    second_diff = np.diff(differences)
    elbow_idx = np.argmax(second_diff) + 2  # +2 because of double diff
    optimal_k = k_values[elbow_idx]
    
    plt.axvline(x=optimal_k, color='r', linestyle='--', linewidth=2, label=f'Elbow at k={optimal_k}')
    plt.scatter([optimal_k], [inertias[elbow_idx]], color='red', s=200, zorder=5, edgecolors='black')
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('elbow_method.png', dpi=150)
    plt.show(block=False)
    
    # Explanation
    print("\n" + "-" * 60)
    print("HOW TO CHOOSE k USING ELBOW METHOD:")
    print("-" * 60)
    print("""
    The Elbow Method works by:
    
    1. PLOTTING INERTIA vs k:
       - Inertia = sum of squared distances from points to their cluster centers
       - As k increases, inertia always decreases
    
    2. FINDING THE "ELBOW":
       - Look for the point where inertia decrease slows dramatically
       - This creates an "elbow" shape in the curve
       - The elbow represents diminishing returns from adding more clusters
    
    3. INTERPRETING THE CURVE:
       - Before elbow: Adding clusters significantly reduces inertia
       - After elbow: Adding clusters provides minimal improvement
       - The elbow point balances complexity vs. clustering quality
    """)
    
    print(f"\n>>> OPTIMAL k SELECTED: {optimal_k} clusters")
    print(f"    (Based on elbow point analysis)")
    
    return optimal_k, inertias


def train_kmeans(X_train_scaled, n_clusters):
    """
    Task 3: Import K-Means and train using training set 
    """
    print("\n" + "=" * 60)
    print(f"3. TRAINING K-MEANS MODEL (k={n_clusters})")
    print("=" * 60)
    
    # Initialize K-Means algorithm
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',      # Smart initialization
        n_init=10,             # Number of times to run with different seeds
        max_iter=300,          # Maximum iterations per run
        random_state=42        # For reproducibility
    )
    
    print("\nK-Means Parameters:")
    print(f"  - n_clusters: {n_clusters}")
    print(f"  - init: 'k-means++' (smart centroid initialization)")
    print(f"  - n_init: 10 (runs with different seeds)")
    print(f"  - max_iter: 300")
    print(f"  - random_state: 42")
    
    # Train the model on training data
    print("\nTraining K-Means on training set...")
    kmeans.fit(X_train_scaled)
    
    print(f"\n>>> Model trained successfully!")
    print(f"    Final inertia: {kmeans.inertia_:.2f}")
    print(f"    Iterations to converge: {kmeans.n_iter_}")
    
    return kmeans


def get_cluster_labels(kmeans, X_train_scaled, X_test_scaled):
    """
    Task 4: Obtain cluster labels for training and test data 
    """
    print("\n" + "=" * 60)
    print("4. OBTAINING CLUSTER LABELS")
    print("=" * 60)
    
    # Get labels for training data (already computed during fit)
    train_labels = kmeans.labels_
    
    # Predict labels for test data
    test_labels = kmeans.predict(X_test_scaled)
    
    print("\n--- Training Set Labels ---")
    print(f"Total samples: {len(train_labels)}")
    unique_train, counts_train = np.unique(train_labels, return_counts=True)
    for cluster, count in zip(unique_train, counts_train):
        percentage = 100 * count / len(train_labels)
        print(f"  Cluster {cluster}: {count} samples ({percentage:.1f}%)")
    
    print("\n--- Test Set Labels ---")
    print(f"Total samples: {len(test_labels)}")
    unique_test, counts_test = np.unique(test_labels, return_counts=True)
    for cluster, count in zip(unique_test, counts_test):
        percentage = 100 * count / len(test_labels)
        print(f"  Cluster {cluster}: {count} samples ({percentage:.1f}%)")
    
    return train_labels, test_labels


def display_cluster_centers(kmeans, feature_names):
    """
    Task 5: Display the cluster centers 
    """
    print("\n" + "=" * 60)
    print("5. CLUSTER CENTERS")
    print("=" * 60)
    
    # Get cluster centers
    centers = kmeans.cluster_centers_
    
    # Create a DataFrame for better visualization
    centers_df = pd.DataFrame(
        centers,
        columns=feature_names,
        index=[f'Cluster {i}' for i in range(len(centers))]
    )
    
    print("\n--- Cluster Centers (Standardized Values) ---")
    print(centers_df.round(3).to_string())
    
    # Visualize cluster centers as a heatmap
    plt.figure(figsize=(14, 6), num='Cluster Centers Heatmap')
    
    # Create heatmap
    im = plt.imshow(centers, cmap='RdYlBu_r', aspect='auto')
    plt.colorbar(im, label='Standardized Value')
    
    # Add labels
    plt.yticks(range(len(centers)), [f'Cluster {i}' for i in range(len(centers))], fontsize=11)
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right', fontsize=9)
    
    # Add values on heatmap
    for i in range(len(centers)):
        for j in range(len(feature_names)):
            text_color = 'white' if abs(centers[i, j]) > 0.5 else 'black'
            plt.text(j, i, f'{centers[i, j]:.2f}', ha='center', va='center', 
                    color=text_color, fontsize=8)
    
    plt.title('Cluster Centers Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cluster_centers.png', dpi=150)
    plt.show(block=False)
    
    # Interpretation
    print("\n" + "-" * 60)
    print("INTERPRETING CLUSTER CENTERS:")
    print("-" * 60)
    print("""
    Each row represents a cluster centroid in the feature space.
    Values are standardized (mean=0, std=1 for original data).
    
    - Positive values: Above average for that feature
    - Negative values: Below average for that feature
    - Near zero: Close to the average
    
    Use these centers to characterize each cluster's profile.
    """)
    
    return centers_df


def run_part_two(filepath='heart_cleaned.csv'):
    """
    Main function to run all Part 2 tasks
    Returns the trained model and labels
    """
    # Load cleaned data from Part 1
    df = load_cleaned_data(filepath)
    feature_names = df.columns.tolist()
    
    # Task 1: Split dataset
    X_train, X_test = split_dataset(df, test_size=0.2)
    
    # Standardize data (important for K-Means)
    X_train_scaled, X_test_scaled, scaler = standardize_data(X_train, X_test)
    
    # Task 2: Apply Elbow Method
    optimal_k, inertias = apply_elbow_method(X_train_scaled)
    
    # Task 3: Train K-Means
    kmeans = train_kmeans(X_train_scaled, n_clusters=optimal_k)
    
    # Task 4: Get cluster labels
    train_labels, test_labels = get_cluster_labels(kmeans, X_train_scaled, X_test_scaled)
    
    # Task 5: Display cluster centers
    centers_df = display_cluster_centers(kmeans, feature_names)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - Part 2 Complete!")
    print("=" * 60)
    print(f"✓ Dataset split: 80% train, 20% test")
    print(f"✓ Optimal k found using Elbow Method: {optimal_k}")
    print(f"✓ K-Means model trained successfully")
    print(f"✓ Cluster labels obtained for train and test sets")
    print(f"✓ Cluster centers displayed and visualized")
    print("\nFiles saved:")
    print("  - elbow_method.png")
    print("  - cluster_centers.png")
    print("\nReady for Part 3 evaluation!")
    
    # Keep plots open
    plt.show(block=True)
    
    return kmeans, train_labels, test_labels, X_train_scaled, X_test_scaled, scaler


if __name__ == "__main__":
    run_part_two()