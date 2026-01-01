# Part 2 – Model Training
# Libraries: pandas, numpy, matplotlib, sklearn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-whitegrid')


def load_cleaned_data(filepath='heart_cleaned.csv'):
    """Load the cleaned dataset from Part 1"""
    print("=" * 60)
    print("LOADING CLEANED DATA")
    print("=" * 60)
    
    df = pd.read_csv(filepath)
    print(f"Dataset loaded from: {filepath}")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def split_dataset(df, test_size=0.2, random_state=42):
    """Task 1: Split dataset into training (80%) and test (20%) sets (0.5pt)"""
    print("\n" + "=" * 60)
    print("1. SPLITTING DATASET (80% Train / 20% Test)")
    print("=" * 60)
    
    X_train, X_test = train_test_split(df, test_size=test_size, random_state=random_state)
    
    print(f"Original size: {len(df)}")
    print(f"Training set: {len(X_train)} ({100*(1-test_size):.0f}%)")
    print(f"Test set: {len(X_test)} ({100*test_size:.0f}%)")
    
    return X_train, X_test


def standardize_data(X_train, X_test):
    """Standardize features using StandardScaler"""
    print("\n" + "=" * 60)
    print("STANDARDIZING DATA")
    print("=" * 60)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Data standardized (fit on train, transform both)")
    print(f"Training shape: {X_train_scaled.shape}")
    print(f"Test shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, scaler


def apply_elbow_method(X_train_scaled, k_range=range(1, 11)):
    """Task 2: Apply Elbow Method to choose optimal k (0.75pt)"""
    print("\n" + "=" * 60)
    print("2. ELBOW METHOD")
    print("=" * 60)
    
    inertias = []
    k_values = list(k_range)
    
    print("\nCalculating inertia for k = 1 to 10...")
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_train_scaled)
        inertias.append(kmeans.inertia_)
        print(f"  k = {k:2d} | Inertia = {kmeans.inertia_:.2f}")
    
    # Plot Elbow curve
    plt.figure(figsize=(10, 6), num='Elbow Method')
    plt.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Inertia', fontsize=12)
    plt.title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
    plt.xticks(k_values)
    plt.grid(True, alpha=0.3)
    
    # Find elbow point
    differences = np.diff(inertias)
    second_diff = np.diff(differences)
    elbow_idx = np.argmax(second_diff) + 2
    optimal_k = k_values[elbow_idx]
    
    plt.axvline(x=optimal_k, color='r', linestyle='--', linewidth=2, label=f'Elbow at k={optimal_k}')
    plt.scatter([optimal_k], [inertias[elbow_idx]], color='red', s=200, zorder=5, edgecolors='black')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('elbow_method.png', dpi=150)
    plt.show(block=False)
    
    print(f"\n>>> OPTIMAL k: {optimal_k}")
    return optimal_k, inertias


def train_kmeans(X_train_scaled, n_clusters):
    """Task 3: Train K-Means using sklearn (0.75pt)"""
    print("\n" + "=" * 60)
    print(f"3. TRAINING K-MEANS (k={n_clusters})")
    print("=" * 60)
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=10,
        max_iter=300,
        random_state=42
    )
    
    print(f"Parameters: n_clusters={n_clusters}, init='k-means++', n_init=10")
    kmeans.fit(X_train_scaled)
    print(f"Training complete! Inertia: {kmeans.inertia_:.2f}, Iterations: {kmeans.n_iter_}")
    
    return kmeans


def get_cluster_labels(kmeans, X_train_scaled, X_test_scaled):
    """Task 4: Obtain cluster labels for train and test data (0.5pt)"""
    print("\n" + "=" * 60)
    print("4. CLUSTER LABELS")
    print("=" * 60)
    
    train_labels = kmeans.labels_
    test_labels = kmeans.predict(X_test_scaled)
    
    print("\n--- Training Set ---")
    unique_train, counts_train = np.unique(train_labels, return_counts=True)
    for cluster, count in zip(unique_train, counts_train):
        print(f"  Cluster {cluster}: {count} ({100*count/len(train_labels):.1f}%)")
    
    print("\n--- Test Set ---")
    unique_test, counts_test = np.unique(test_labels, return_counts=True)
    for cluster, count in zip(unique_test, counts_test):
        print(f"  Cluster {cluster}: {count} ({100*count/len(test_labels):.1f}%)")
    
    return train_labels, test_labels


def display_cluster_centers(kmeans, feature_names):
    """Task 5: Display cluster centers (0.5pt)"""
    print("\n" + "=" * 60)
    print("5. CLUSTER CENTERS")
    print("=" * 60)
    
    centers = kmeans.cluster_centers_
    centers_df = pd.DataFrame(centers, columns=feature_names,
                               index=[f'Cluster {i}' for i in range(len(centers))])
    
    print("\n--- Cluster Centers (Standardized) ---")
    print(centers_df.round(3).to_string())
    
    # Heatmap
    plt.figure(figsize=(14, 6), num='Cluster Centers')
    im = plt.imshow(centers, cmap='RdYlBu_r', aspect='auto')
    plt.colorbar(im, label='Standardized Value')
    plt.yticks(range(len(centers)), [f'Cluster {i}' for i in range(len(centers))])
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right', fontsize=9)
    
    for i in range(len(centers)):
        for j in range(len(feature_names)):
            color = 'white' if abs(centers[i, j]) > 0.5 else 'black'
            plt.text(j, i, f'{centers[i, j]:.2f}', ha='center', va='center', color=color, fontsize=8)
    
    plt.title('Cluster Centers Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cluster_centers.png', dpi=150)
    plt.show(block=False)
    
    return centers_df


def run_part_two(filepath='heart_cleaned.csv'):
    """Main function to run all Part 2 tasks"""
    df = load_cleaned_data(filepath)
    feature_names = df.columns.tolist()
    
    # Task 1: Split
    X_train, X_test = split_dataset(df, test_size=0.2)
    
    # Standardize
    X_train_scaled, X_test_scaled, scaler = standardize_data(X_train, X_test)
    
    # Task 2: Elbow Method
    optimal_k, inertias = apply_elbow_method(X_train_scaled)
    
    # Task 3: Train K-Means
    kmeans = train_kmeans(X_train_scaled, n_clusters=optimal_k)
    
    # Task 4: Get labels
    train_labels, test_labels = get_cluster_labels(kmeans, X_train_scaled, X_test_scaled)
    
    # Task 5: Cluster centers
    centers_df = display_cluster_centers(kmeans, feature_names)
    
    print("\n" + "=" * 60)
    print("SUMMARY - Part 2 Complete!")
    print("=" * 60)
    print(f"✓ Split: 80% train, 20% test")
    print(f"✓ Optimal k: {optimal_k}")
    print(f"✓ K-Means trained")
    print(f"✓ Labels obtained")
    print(f"✓ Centers displayed")
    
    plt.show(block=True)
    return kmeans, train_labels, test_labels, X_train_scaled, X_test_scaled, scaler


if __name__ == "__main__":
    run_part_two()