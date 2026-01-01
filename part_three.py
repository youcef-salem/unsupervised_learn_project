# Part 3 – Evaluation (Functions Module)
# Libraries needed: pandas, numpy, matplotlib, sklearn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
from sklearn.decomposition import PCA

# Set display options
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-whitegrid')


def evaluate_clustering_metrics(X_train_scaled, kmeans, train_labels):
    """
    Task 1: Evaluate clustering using appropriate metrics 
    - Inertia
    - Silhouette Score
    - Davies-Bouldin Index
    """
    print("=" * 60)
    print("1. CLUSTERING EVALUATION METRICS")
    print("=" * 60)
    
    # Calculate metrics
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_train_scaled, train_labels)
    davies_bouldin = davies_bouldin_score(X_train_scaled, train_labels)
    
    print("\n--- Evaluation Metrics ---")
    print(f"\n1. INERTIA (Within-Cluster Sum of Squares): {inertia:.2f}")
    print(f"2. SILHOUETTE SCORE: {silhouette:.4f}")
    print(f"3. DAVIES-BOULDIN INDEX: {davies_bouldin:.4f}")
    
    # Explanation of metrics
    print("\n" + "-" * 60)
    print("WHAT THESE METRICS MEAN:")
    print("-" * 60)
    print("""
    1. INERTIA (WCSS - Within-Cluster Sum of Squares):
       - Sum of squared distances from each point to its cluster center
       - Lower values indicate tighter, more compact clusters
       - Always decreases as k increases (use with Elbow Method)
    
    2. SILHOUETTE SCORE:
       - Measures how similar a point is to its own cluster vs other clusters
       - Range: [-1, 1]
         * +1: Points are well matched to their cluster
         *  0: Points are on cluster boundaries
         * -1: Points may be assigned to wrong cluster
       - Higher is better (> 0.5 is generally good)
    
    3. DAVIES-BOULDIN INDEX:
       - Ratio of within-cluster distances to between-cluster distances
       - Lower values indicate better separation between clusters
       - Range: [0, ∞) - no upper bound
       - Values < 1 indicate good clustering
    """)
    
    return inertia, silhouette, davies_bouldin


def compare_metrics_for_different_k(X_train_scaled, k_range=range(2, 11)):
    """
    Task 2: Compare metric values for different choices of k (0.75pt)
    """
    print("\n" + "=" * 60)
    print("2. COMPARING METRICS FOR DIFFERENT k VALUES")
    print("=" * 60)
    
    # Store metrics for each k
    k_values = list(k_range)
    inertias = []
    silhouettes = []
    davies_bouldins = []
    
    print("\nCalculating metrics for k = 2 to 10...")
    print("-" * 70)
    print(f"{'k':^5} | {'Inertia':^12} | {'Silhouette':^12} | {'Davies-Bouldin':^14}")
    print("-" * 70)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_train_scaled)
        
        inertia = kmeans.inertia_
        silhouette = silhouette_score(X_train_scaled, labels)
        davies_bouldin = davies_bouldin_score(X_train_scaled, labels)
        
        inertias.append(inertia)
        silhouettes.append(silhouette)
        davies_bouldins.append(davies_bouldin)
        
        print(f"{k:^5} | {inertia:>12.2f} | {silhouette:>12.4f} | {davies_bouldin:>14.4f}")
    
    print("-" * 70)
    
    # Find best k for each metric
    best_k_silhouette = k_values[np.argmax(silhouettes)]
    best_k_db = k_values[np.argmin(davies_bouldins)]
    
    print(f"\n--- Best k based on metrics ---")
    print(f"Best k for Silhouette Score (highest): k = {best_k_silhouette} ({max(silhouettes):.4f})")
    print(f"Best k for Davies-Bouldin (lowest): k = {best_k_db} ({min(davies_bouldins):.4f})")
    
    # Create comparison DataFrame
    metrics_df = pd.DataFrame({
        'k': k_values,
        'Inertia': inertias,
        'Silhouette': silhouettes,
        'Davies-Bouldin': davies_bouldins
    })
    
    # Visualize metrics comparison
    plt.ion()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), num='Metrics Comparison - Window 1')
    
    # Plot Inertia
    axes[0].plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Clusters (k)', fontsize=11)
    axes[0].set_ylabel('Inertia', fontsize=11)
    axes[0].set_title('Inertia vs k\n(Lower is Better)', fontsize=12, fontweight='bold')
    axes[0].set_xticks(k_values)
    axes[0].grid(True, alpha=0.3)
    
    # Plot Silhouette Score
    axes[1].plot(k_values, silhouettes, 'go-', linewidth=2, markersize=8)
    axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Good threshold (0.5)')
    axes[1].set_xlabel('Number of Clusters (k)', fontsize=11)
    axes[1].set_ylabel('Silhouette Score', fontsize=11)
    axes[1].set_title('Silhouette Score vs k\n(Higher is Better)', fontsize=12, fontweight='bold')
    axes[1].set_xticks(k_values)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    # Plot Davies-Bouldin Index
    axes[2].plot(k_values, davies_bouldins, 'ro-', linewidth=2, markersize=8)
    axes[2].axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Good threshold (1.0)')
    axes[2].set_xlabel('Number of Clusters (k)', fontsize=11)
    axes[2].set_ylabel('Davies-Bouldin Index', fontsize=11)
    axes[2].set_title('Davies-Bouldin Index vs k\n(Lower is Better)', fontsize=12, fontweight='bold')
    axes[2].set_xticks(k_values)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Comparison of Clustering Metrics for Different k Values', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=150)
    
    print("\n✓ Metrics comparison plot saved as 'metrics_comparison.png'")
    
    return metrics_df, k_values, inertias, silhouettes, davies_bouldins


def interpret_cluster_quality(metrics_df, optimal_k, inertia, silhouette, davies_bouldin):
    """
    Task 3: Interpret what these metrics say about cluster quality (1pt)
    """
    print("\n" + "=" * 60)
    print("3. INTERPRETATION OF CLUSTER QUALITY")
    print("=" * 60)
    
    print(f"\n--- Analysis for k = {optimal_k} clusters ---")
    
    # Interpret Inertia
    print("\n1. INERTIA INTERPRETATION:")
    print(f"   Value: {inertia:.2f}")
    print("   • Inertia alone is hard to interpret without comparison")
    print("   • The elbow method helps identify where adding more clusters")
    print("     provides diminishing returns")
    print("   • A good k is where inertia drop starts to level off")
    
    # Interpret Silhouette Score
    print("\n2. SILHOUETTE SCORE INTERPRETATION:")
    print(f"   Value: {silhouette:.4f}")
    if silhouette > 0.7:
        quality_sil = "EXCELLENT - Strong cluster structure"
    elif silhouette > 0.5:
        quality_sil = "GOOD - Reasonable cluster structure"
    elif silhouette > 0.25:
        quality_sil = "FAIR - Weak cluster structure, possible overlap"
    else:
        quality_sil = "POOR - Clusters may be artificial or overlapping"
    print(f"   Assessment: {quality_sil}")
    
    # Interpret Davies-Bouldin Index
    print("\n3. DAVIES-BOULDIN INDEX INTERPRETATION:")
    print(f"   Value: {davies_bouldin:.4f}")
    if davies_bouldin < 0.5:
        quality_db = "EXCELLENT - Very well-separated clusters"
    elif davies_bouldin < 1.0:
        quality_db = "GOOD - Well-separated clusters"
    elif davies_bouldin < 1.5:
        quality_db = "FAIR - Moderate cluster separation"
    else:
        quality_db = "POOR - Clusters are not well separated"
    print(f"   Assessment: {quality_db}")
    
    # Overall assessment
    print("\n" + "-" * 60)
    print("OVERALL CLUSTER QUALITY ASSESSMENT:")
    print("-" * 60)
    
    # Calculate overall score
    overall_score = 0
    if silhouette > 0.5:
        overall_score += 2
    elif silhouette > 0.25:
        overall_score += 1
    
    if davies_bouldin < 1.0:
        overall_score += 2
    elif davies_bouldin < 1.5:
        overall_score += 1
    
    if overall_score >= 3:
        overall = "GOOD CLUSTERING"
        recommendation = "The clustering is meaningful and well-structured."
    elif overall_score >= 2:
        overall = "ACCEPTABLE CLUSTERING"
        recommendation = "Clusters exist but have some overlap. Consider feature engineering."
    else:
        overall = "WEAK CLUSTERING"
        recommendation = "Clusters are not well-defined. Try different k or preprocessing."
    
    print(f"\n   >>> {overall} <<<")
    print(f"\n   Recommendation: {recommendation}")
    
    print("\n" + "-" * 60)
    print("KEY INSIGHTS:")
    print("-" * 60)
    print("""
    • Higher Silhouette Score = points are well-matched to their clusters
    • Lower Davies-Bouldin Index = clusters are more distinct from each other
    • These metrics help validate whether the chosen k makes sense
    • If metrics are poor, consider:
      - Different number of clusters
      - Different features or feature scaling
      - Different clustering algorithm (DBSCAN, hierarchical, etc.)
    """)


def visualize_clustering_results(X_train_scaled, kmeans, train_labels, feature_names):
    """
    Task 4: Visualize clustering results using matplotlib plots (1pt)
    - Color data points according to their assigned cluster
    - Visualize cluster centroids
    """
    print("\n" + "=" * 60)
    print("4. CLUSTERING VISUALIZATION")
    print("=" * 60)
    
    n_clusters = kmeans.n_clusters
    centers = kmeans.cluster_centers_
    
    print(f"\nVisualizing {n_clusters} clusters...")
    print(f"Number of features: {X_train_scaled.shape[1]}")
    print("Using PCA for 2D visualization of high-dimensional data")
    
    # Apply PCA for 2D visualization
    pca = PCA(n_components=2)
    X_train_2d = pca.fit_transform(X_train_scaled)
    centers_2d = pca.transform(centers)
    
    explained_var = pca.explained_variance_ratio_
    print(f"\nPCA Explained Variance:")
    print(f"  PC1: {explained_var[0]:.2%}")
    print(f"  PC2: {explained_var[1]:.2%}")
    print(f"  Total: {sum(explained_var):.2%}")
    
    # Define colors for clusters
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
    
    # --- Figure 2: Main Clustering Visualization ---
    fig2, ax = plt.subplots(figsize=(12, 9), num='Clustering Results - Window 2')
    
    # Plot data points colored by cluster
    for i in range(n_clusters):
        cluster_mask = train_labels == i
        ax.scatter(X_train_2d[cluster_mask, 0], X_train_2d[cluster_mask, 1],
                   c=[colors[i]], label=f'Cluster {i} (n={sum(cluster_mask)})',
                   s=60, alpha=0.6, edgecolors='white', linewidth=0.5)
    
    # Plot cluster centroids with larger markers
    ax.scatter(centers_2d[:, 0], centers_2d[:, 1],
               c='black', marker='X', s=300, linewidths=2,
               edgecolors='yellow', label='Centroids', zorder=10)
    
    # Add centroid labels
    for i, (cx, cy) in enumerate(centers_2d):
        ax.annotate(f'C{i}', (cx, cy), fontsize=12, fontweight='bold',
                    color='white', ha='center', va='center')
    
    ax.set_xlabel(f'Principal Component 1 ({explained_var[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'Principal Component 2 ({explained_var[1]:.1%} variance)', fontsize=12)
    ax.set_title(f'K-Means Clustering Results (k={n_clusters})\nPCA 2D Projection', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('clustering_2d.png', dpi=150)
    print("✓ 2D clustering plot saved as 'clustering_2d.png'")
    
    # --- Figure 3: Silhouette Analysis ---
    fig3, ax3 = plt.subplots(figsize=(10, 8), num='Silhouette Analysis - Window 3')
    
    silhouette_vals = silhouette_samples(X_train_scaled, train_labels)
    y_lower = 10
    
    for i in range(n_clusters):
        cluster_silhouette_vals = silhouette_vals[train_labels == i]
        cluster_silhouette_vals.sort()
        
        cluster_size = len(cluster_silhouette_vals)
        y_upper = y_lower + cluster_size
        
        ax3.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                          facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
        ax3.text(-0.05, y_lower + 0.5 * cluster_size, f'Cluster {i}', fontsize=10)
        
        y_lower = y_upper + 10
    
    avg_silhouette = silhouette_vals.mean()
    ax3.axvline(x=avg_silhouette, color='red', linestyle='--', linewidth=2,
                label=f'Average Silhouette: {avg_silhouette:.3f}')
    
    ax3.set_xlabel('Silhouette Coefficient', fontsize=12)
    ax3.set_ylabel('Cluster', fontsize=12)
    ax3.set_title('Silhouette Analysis for Each Cluster', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.set_yticks([])
    ax3.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('silhouette_analysis.png', dpi=150)
    print("✓ Silhouette analysis plot saved as 'silhouette_analysis.png'")
    
    # --- Figure 4: Cluster Distribution ---
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6), num='Cluster Distribution - Window 4')
    
    # Cluster sizes bar chart
    unique, counts = np.unique(train_labels, return_counts=True)
    bars = axes4[0].bar([f'Cluster {i}' for i in unique], counts, color=colors, 
                         edgecolor='black', linewidth=1.5)
    
    for bar, count in zip(bars, counts):
        axes4[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                      f'{count}\n({100*count/len(train_labels):.1f}%)',
                      ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    axes4[0].set_ylabel('Number of Samples', fontsize=12)
    axes4[0].set_xlabel('Cluster', fontsize=12)
    axes4[0].set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')
    axes4[0].grid(True, alpha=0.3, axis='y')
    
    # Pie chart
    axes4[1].pie(counts, labels=[f'Cluster {i}' for i in unique], colors=colors,
                 autopct='%1.1f%%', startangle=90, explode=[0.02]*n_clusters,
                 shadow=True, textprops={'fontsize': 11})
    axes4[1].set_title('Cluster Proportion', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Cluster Distribution Analysis (k={n_clusters})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cluster_distribution.png', dpi=150)
    print("✓ Cluster distribution plot saved as 'cluster_distribution.png'")
    
    # --- Figure 5: Cluster Centers Heatmap ---
    fig5, ax5 = plt.subplots(figsize=(14, 6), num='Cluster Centers - Window 5')
    
    im = ax5.imshow(centers, cmap='RdYlBu_r', aspect='auto')
    plt.colorbar(im, ax=ax5, label='Standardized Value')
    
    ax5.set_yticks(range(n_clusters))
    ax5.set_yticklabels([f'Cluster {i}' for i in range(n_clusters)], fontsize=11)
    ax5.set_xticks(range(len(feature_names)))
    ax5.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=9)
    
    # Add values on heatmap
    for i in range(n_clusters):
        for j in range(len(feature_names)):
            text_color = 'white' if abs(centers[i, j]) > 0.7 else 'black'
            ax5.text(j, i, f'{centers[i, j]:.2f}', ha='center', va='center',
                     color=text_color, fontsize=8)
    
    ax5.set_title('Cluster Centers Feature Values', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cluster_centers_heatmap.png', dpi=150)
    print("✓ Cluster centers heatmap saved as 'cluster_centers_heatmap.png'")
    
    print("\n" + "-" * 60)
    print("5 VISUALIZATION WINDOWS ARE NOW OPEN!")
    print("-" * 60)
    print("Tips for interacting with the plots:")
    print("  - Use the toolbar at the bottom of each window")
    print("  - Pan tool: Click and drag to move around")
    print("  - Zoom tool: Draw a rectangle to zoom in")
    print("  - Home button: Reset to original view")
    print("-" * 60)
    
    plt.show(block=True)


def run_part_three(kmeans, train_labels, X_train_scaled, feature_names):
    """
    Main function to run all Part 3 tasks
    """
    # Task 1: Evaluate clustering metrics
    inertia, silhouette, davies_bouldin = evaluate_clustering_metrics(
        X_train_scaled, kmeans, train_labels
    )
    
    # Task 2: Compare metrics for different k
    metrics_df, k_values, inertias, silhouettes, davies_bouldins = compare_metrics_for_different_k(
        X_train_scaled
    )
    
    # Task 3: Interpret cluster quality
    optimal_k = kmeans.n_clusters
    interpret_cluster_quality(metrics_df, optimal_k, inertia, silhouette, davies_bouldin)
    
    # Task 4: Visualize clustering results
    visualize_clustering_results(X_train_scaled, kmeans, train_labels, feature_names)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - Part 3 Complete!")
    print("=" * 60)
    print(f"✓ Evaluation metrics calculated (Inertia, Silhouette, Davies-Bouldin)")
    print(f"✓ Metrics compared for k = 2 to 10")
    print(f"✓ Cluster quality interpreted")
    print(f"✓ Clustering results visualized")
    print("\nVisualization files saved:")
    print("  - metrics_comparison.png")
    print("  - clustering_2d.png")
    print("  - silhouette_analysis.png")
    print("  - cluster_distribution.png")
    print("  - cluster_centers_heatmap.png")
    print("\n>>> PROJECT COMPLETE! <<<")
    
    return metrics_df


if __name__ == "__main__":
    # For standalone testing - requires data from Part 2
    print("Part 3 should be run from init.py after Part 1 and Part 2")
    print("Run: python init.py")
