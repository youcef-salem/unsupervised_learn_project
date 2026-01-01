# Part 3 – Evaluation
# Libraries: pandas, numpy, matplotlib, sklearn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-whitegrid')


def evaluate_clustering_metrics(X_train_scaled, kmeans, train_labels):
    """Task 1: Evaluate clustering using metrics (0.75pt)"""
    print("=" * 60)
    print("1. CLUSTERING EVALUATION METRICS")
    print("=" * 60)
    
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_train_scaled, train_labels)
    davies_bouldin = davies_bouldin_score(X_train_scaled, train_labels)
    
    print(f"\n1. INERTIA: {inertia:.2f}")
    print(f"2. SILHOUETTE SCORE: {silhouette:.4f}")
    print(f"3. DAVIES-BOULDIN INDEX: {davies_bouldin:.4f}")
    
    return inertia, silhouette, davies_bouldin


def compare_metrics_for_different_k(X_train_scaled, k_range=range(2, 11)):
    """Task 2: Compare metrics for different k values (0.75pt)"""
    print("\n" + "=" * 60)
    print("2. COMPARING METRICS FOR DIFFERENT k")
    print("=" * 60)
    
    k_values = list(k_range)
    inertias, silhouettes, davies_bouldins = [], [], []
    
    print(f"\n{'k':^5} | {'Inertia':^12} | {'Silhouette':^12} | {'Davies-Bouldin':^14}")
    print("-" * 50)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_train_scaled)
        
        inertia = kmeans.inertia_
        silhouette = silhouette_score(X_train_scaled, labels)
        db = davies_bouldin_score(X_train_scaled, labels)
        
        inertias.append(inertia)
        silhouettes.append(silhouette)
        davies_bouldins.append(db)
        
        print(f"{k:^5} | {inertia:>12.2f} | {silhouette:>12.4f} | {db:>14.4f}")
    
    best_k_sil = k_values[np.argmax(silhouettes)]
    best_k_db = k_values[np.argmin(davies_bouldins)]
    print(f"\nBest k (Silhouette): {best_k_sil} ({max(silhouettes):.4f})")
    print(f"Best k (Davies-Bouldin): {best_k_db} ({min(davies_bouldins):.4f})")
    
    # Plot metrics
    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), num='Metrics Comparison')
    
    axes[0].plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('k')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Inertia vs k (Lower=Better)', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(k_values, silhouettes, 'go-', linewidth=2, markersize=8)
    axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('k')
    axes[1].set_ylabel('Silhouette')
    axes[1].set_title('Silhouette vs k (Higher=Better)', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(k_values, davies_bouldins, 'ro-', linewidth=2, markersize=8)
    axes[2].axhline(y=1.0, color='green', linestyle='--', alpha=0.7)
    axes[2].set_xlabel('k')
    axes[2].set_ylabel('Davies-Bouldin')
    axes[2].set_title('Davies-Bouldin vs k (Lower=Better)', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=150)
    
    metrics_df = pd.DataFrame({'k': k_values, 'Inertia': inertias, 
                               'Silhouette': silhouettes, 'Davies-Bouldin': davies_bouldins})
    return metrics_df, k_values, inertias, silhouettes, davies_bouldins


def interpret_cluster_quality(metrics_df, optimal_k, inertia, silhouette, davies_bouldin):
    """Task 3: Interpret cluster quality (1pt)"""
    print("\n" + "=" * 60)
    print("3. INTERPRETATION OF CLUSTER QUALITY")
    print("=" * 60)
    
    print(f"\n--- Analysis for k = {optimal_k} ---")
    print(f"Inertia: {inertia:.2f}")
    
    # Silhouette interpretation
    if silhouette > 0.7:
        sil_quality = "EXCELLENT"
    elif silhouette > 0.5:
        sil_quality = "GOOD"
    elif silhouette > 0.25:
        sil_quality = "FAIR"
    else:
        sil_quality = "POOR"
    print(f"Silhouette: {silhouette:.4f} - {sil_quality}")
    
    # Davies-Bouldin interpretation
    if davies_bouldin < 0.5:
        db_quality = "EXCELLENT"
    elif davies_bouldin < 1.0:
        db_quality = "GOOD"
    elif davies_bouldin < 1.5:
        db_quality = "FAIR"
    else:
        db_quality = "POOR"
    print(f"Davies-Bouldin: {davies_bouldin:.4f} - {db_quality}")


def visualize_clustering_results(X_train_scaled, kmeans, train_labels, feature_names):
    """Task 4: Visualize clustering with colored points and centroids (1pt)"""
    print("\n" + "=" * 60)
    print("4. CLUSTERING VISUALIZATION")
    print("=" * 60)
    
    n_clusters = kmeans.n_clusters
    centers = kmeans.cluster_centers_
    
    # PCA for 2D visualization
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_train_scaled)
    centers_2d = pca.transform(centers)
    explained_var = pca.explained_variance_ratio_
    
    print(f"PCA: PC1={explained_var[0]:.1%}, PC2={explained_var[1]:.1%}")
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
    
    # 2D Clustering plot
    fig2, ax = plt.subplots(figsize=(12, 9), num='Clustering Results')
    for i in range(n_clusters):
        mask = train_labels == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors[i]], 
                   label=f'Cluster {i} (n={sum(mask)})', s=60, alpha=0.6)
    ax.scatter(centers_2d[:, 0], centers_2d[:, 1], c='black', marker='X', 
               s=300, edgecolors='yellow', label='Centroids', zorder=10)
    ax.set_xlabel(f'PC1 ({explained_var[0]:.1%})')
    ax.set_ylabel(f'PC2 ({explained_var[1]:.1%})')
    ax.set_title(f'K-Means Clustering (k={n_clusters})', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('clustering_2d.png', dpi=150)
    
    # Silhouette plot
    fig3, ax3 = plt.subplots(figsize=(10, 8), num='Silhouette Analysis')
    silhouette_vals = silhouette_samples(X_train_scaled, train_labels)
    y_lower = 10
    for i in range(n_clusters):
        cluster_vals = silhouette_vals[train_labels == i]
        cluster_vals.sort()
        y_upper = y_lower + len(cluster_vals)
        ax3.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_vals, 
                          facecolor=colors[i], alpha=0.7)
        y_lower = y_upper + 10
    ax3.axvline(x=silhouette_vals.mean(), color='red', linestyle='--', 
                label=f'Avg: {silhouette_vals.mean():.3f}')
    ax3.set_xlabel('Silhouette Coefficient')
    ax3.set_title('Silhouette Analysis', fontweight='bold')
    ax3.legend()
    ax3.set_yticks([])
    plt.tight_layout()
    plt.savefig('silhouette_analysis.png', dpi=150)
    
    # Cluster distribution
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6), num='Cluster Distribution')
    unique, counts = np.unique(train_labels, return_counts=True)
    axes4[0].bar([f'Cluster {i}' for i in unique], counts, color=colors, edgecolor='black')
    for i, count in enumerate(counts):
        axes4[0].text(i, count + 1, f'{count}\n({100*count/len(train_labels):.1f}%)',
                      ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes4[0].set_ylabel('Number of Samples', fontsize=12)
    axes4[0].set_xlabel('Cluster', fontsize=12)
    axes4[0].set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')
    axes4[0].grid(True, alpha=0.3, axis='y')
    
    axes4[1].pie(counts, labels=[f'Cluster {i}' for i in unique], colors=colors,
                 autopct='%1.1f%%', startangle=90, explode=[0.02]*n_clusters,
                 shadow=True, textprops={'fontsize': 11})
    axes4[1].set_title('Cluster Proportion', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Cluster Distribution Analysis (k={n_clusters})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cluster_distribution.png', dpi=150)
    
    # Cluster centers heatmap
    fig5, ax5 = plt.subplots(figsize=(14, 6), num='Cluster Centers')
    im = ax5.imshow(centers, cmap='RdYlBu_r', aspect='auto')
    plt.colorbar(im, ax=ax5, label='Standardized Value')
    
    ax5.set_yticks(range(n_clusters))
    ax5.set_yticklabels([f'Cluster {i}' for i in range(n_clusters)], fontsize=11)
    ax5.set_xticks(range(len(feature_names)))
    ax5.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=9)
    
    for i in range(n_clusters):
        for j in range(len(feature_names)):
            text_color = 'white' if abs(centers[i, j]) > 0.7 else 'black'
            ax5.text(j, i, f'{centers[i, j]:.2f}', ha='center', va='center',
                     color=text_color, fontsize=8)
    
    ax5.set_title('Cluster Centers Feature Values', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cluster_centers_heatmap.png', dpi=150)
    
    print("✓ Saved: clustering_2d.png, silhouette_analysis.png, cluster_distribution.png, cluster_centers_heatmap.png")
    plt.show(block=True)


def run_part_three(kmeans, train_labels, X_train_scaled, feature_names):
    """Main function to run all Part 3 tasks"""
    # Task 1: Metrics
    inertia, silhouette, davies_bouldin = evaluate_clustering_metrics(X_train_scaled, kmeans, train_labels)
    
    # Task 2: Compare for different k
    metrics_df, k_values, inertias, silhouettes, davies_bouldins = compare_metrics_for_different_k(X_train_scaled)
    
    # Task 3: Interpret
    interpret_cluster_quality(metrics_df, kmeans.n_clusters, inertia, silhouette, davies_bouldin)
    
    # Task 4: Visualize
    visualize_clustering_results(X_train_scaled, kmeans, train_labels, feature_names)
    
    print("\n" + "=" * 60)
    print("SUMMARY - Part 3 Complete!")
    print("=" * 60)
    print("✓ Metrics calculated")
    print("✓ Metrics compared for k=2 to 10")
    print("✓ Quality interpreted")
    print("✓ Visualizations saved")
    
    return metrics_df


if __name__ == "__main__":
    print("Run from init.py: python init.py")
