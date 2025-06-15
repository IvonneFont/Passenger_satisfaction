# Unsupervised Learning Analysis of Airline Passenger Satisfaction

## Project Overview

In this project, I applied unsupervised machine learning techniques to analyze airline passenger satisfaction data. The goal was to identify meaningful customer segments and uncover patterns in satisfaction levels using clustering and dimensionality reduction.

## Dataset

- **Source:** Maven Analytics
- **File:** `airline_passenger_satisfaction.csv`
- **Size:** 120,000+ passenger records
- **Features include:**  
  - Passenger demographics  
  - Flight details (e.g. flight delays)  
  - Type of travel  
  - Customer satisfaction ratings across multiple dimensions (cleanliness, comfort, service, etc.)

## Methodology

### Data Preparation

- One-hot encoding of categorical variables
- Standardization using `RobustScaler` to handle outliers
- Labels were removed to ensure purely unsupervised analysis

### Clustering Techniques

- **KMeans clustering:**
  - Initial analysis with default parameters
  - Optimized cluster count using Elbow method and Silhouette score

- **HDBSCAN clustering:**
  - Tuned minimum cluster size and minimum samples on a random sample of 12,000 passengers
  - Selected model based on noise levels and Silhouette score
  - Final model applied to entire dataset

### Dimensionality Reduction (PCA)

- Applied PCA to reduce dimensionality
- Optimized number of components based on Scree plot analysis
- Used to visualize clustering results and uncover underlying data structures

## Key Findings

- **Flight delays emerged as the most significant factor** driving variance across all models.
- PCA revealed two additional latent factors not fully captured by the clustering:
  - An overall service satisfaction component
  - A tech and comfort satisfaction component

- Clusters showed significant overlap in PCA space, reflecting the continuous nature of real-world customer experiences.

- Notable passenger patterns:
  - For shorter delays (low PC1), satisfaction levels are polarized: some passengers are highly satisfied (good flight + good service), while others are strongly dissatisfied.
  - As flight delays increase, satisfaction levels converge toward neutrality, especially for tech and comfort aspects.
  - Many passengers exhibit a tradeoff between overall service and tech/comfort satisfaction, with two main groups:
    - Positive overall service but negative tech/comfort satisfaction
    - Negative overall service and negative tech/comfort satisfaction

## Tools Used

- Python (Pandas, NumPy, Scikit-learn, HDBSCAN, Matplotlib, Seaborn)
- KMeans Clustering
- HDBSCAN Clustering
- Principal Component Analysis (PCA)
- Silhouette Score, Elbow Method
- RobustScaler for data normalization

## Future Work

- Perform cluster-level correlation heatmaps to better profile passenger segments
- Explore dominance of customer types (business vs personal, returning vs new) within each cluster
- Analyze cluster centroids for deeper passenger insights
- Re-run the analysis excluding delay features to detect other hidden patterns
- Apply soft clustering methods (e.g. Gaussian Mixture Models) to better model overlapping or transitional clusters
