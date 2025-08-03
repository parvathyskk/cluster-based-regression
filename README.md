# Clustering Based Linear Regression

## Project Overview

This project explores an advanced approach to predictive modeling by integrating clustering techniques with linear regression to enhance prediction accuracy. Traditional linear regression often struggles to capture the complexities of real-world data due as it uses a single linear relationship across the entire dataset. To overcome this limitation, this project proposes dividing the dataset into meaningful, homogeneous groups (clusters) and then applying separate linear regression models to each cluster. This method, known as cluster-based linear regression, aims to create more subtle differences and a more accurate predictive framework.

The project investigates the application of three distinct clustering methods—K-means, Hierarchical, and DBSCAN—in conjunction with linear regression and Random Forest models.

## Problem Statement

Traditional linear regression models assume a single linear relationship across the entire dataset. In real-world scenarios, data often exhibits complex, non-linear, and heterogeneous patterns that a single global model cannot effectively capture. This limitation can lead to suboptimal prediction accuracy and a lack of insight into localized data behaviors.

## Solution Approach

To address the limitations of traditional linear regression, this project implements a clustering-based linear regression approach. The core idea is to:

1.  **Divide the data:** Utilize clustering algorithms to partition the dataset into subgroups where data points within each group exhibit more similar relationships.
2.  **Train localized models:** Fit separate linear regression (or Random Forest) models to each identified cluster.
3.  **Improve Accuracy:** By capturing distinct relationships inside each cluster, this method aims to enhance overall predicting performance.

## Key Features

* **Integration of Clustering and Regression:** Combines the strengths of data segmentation with predictive modeling.
* **Three Clustering Algorithms Explored:**
    * **K-means Clustering:** An unsupervised learning algorithm that divides data into 'k' non-overlapping clusters with the goal of reducing variance within each cluster. The Elbow Method is used to determine the optimal number of clusters.
    * **Hierarchical Clustering:** Creates a hierarchy of clusters structured as a dendrogram, eliminating the need to predetermine the number of clusters.
    * **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):** A density-based algorithm that identifies clusters by measuring the density of data points and can handle arbitrary shapes and noise.
* **Regression Models Used:** Linear Regression and Random Forest.
* **Comprehensive Data Preprocessing:** Includes steps like removing duplicates, standardizing, handling empty records, normalization, standardization, categorical variable encoding, feature engineering, and dimensionality reduction.
* **Evaluation Metrics:** R-squared and Mean Squared Error for regression performance, and Silhouette Score for cluster quality.

## Datasets Used

The effectiveness of the clustering-based regression approach was validated on three distinct real-world datasets:

* **House Price Dataset:** Contains 20,640 entries and 10 columns of housing-related metrics, including geographical coordinates, and `medianhousevalue`. The 'total bedrooms' column contains some missing values.
* **Auto-MPG (Miles Per Gallon) Dataset:** Contains 397 entries with 9 columns representing automobile data, including `mpg`, `cylinders`, `horsepower`, and `weight`. The 'horsepower' column has some missing or invalid entries.
* **Wine Quality Dataset:** Contains 1,599 entries with 12 columns, detailing various chemical properties and a `quality` rating for wine. All columns are fully populated.

## Results and Findings

The project demonstrates that clustering-based linear regression significantly improves prediction accuracy compared to traditional linear regression, especially for datasets exhibiting varied patterns.

* **K-means Performance:** K-means clustering, when combined with Linear Regression, effectively models linear trends within subgroups, leading to better overall accuracy. K-means generally outperforms Hierarchical clustering in regression analysis due to its iterative centroid updates and refinement.
* **Random Forest:** While Random Forest is a powerful non-linear model on its own, its combination with clustering (K-means and Hierarchical) further enhanced accuracy by capturing unique patterns and complex relationships within each cluster.
* **DBSCAN Limitations:** DBSCAN, while good at identifying arbitrary shapes and handling noise, may not perform as well when dealing with clusters of varied densities or non-spherical shapes, potentially leading to lower regression accuracy if cluster assumptions are violated. This is evident in the negative R2 scores for DBSCAN with Linear Regression on AutoMPG and Wine Quality datasets.
* **Overall Improvement:** For both AutoMPG and House Price datasets, clustering-based regression models (K-means and Hierarchical) achieved higher prediction accuracy than regular linear regression. For Wine Quality, Random Forest combined with K-means and hierarchical clustering showed superior performance.

## How to Run (General Guidance)

This section provides general guidance based on typical machine learning projects. Specific code implementation details would be required for a more precise guide.

### Prerequisites

* Python (3.x recommended)
* Jupyter Notebook or a Python IDE (VS Code, PyCharm, etc.)
* Required Python libraries:
    * `pandas`
    * `numpy`
    * `scikit-learn` (for Linear Regression, Random Forest, K-means, Hierarchical, DBSCAN, preprocessing tools, model selection)
    * `matplotlib`
    * `seaborn` (for visualizations like histograms, box plots, and heatmaps)

### Installation

```bash
# It's recommended to use a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`

pip install pandas numpy scikit-learn matplotlib seaborn
