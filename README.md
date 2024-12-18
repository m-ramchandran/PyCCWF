# Cross-Cluster Weighted Forests

A Python implementation of the Cross-Cluster Weighted Forests method for ensemble learning across multiple clusters or datasets. Full details can be found here: https://arxiv.org/abs/2105.07610 


## Installation
```bash
pip install PyCCWF
```
To install from source: 
```bash
git clone https://github.com/m-ramchandran/PyCCWF.git
cd PyCCWF
pip install -e .
```

## Quick Start

### Running the Example
Currently, only regression is implemented. Classification will be included in a later release. 

```python
from PyCCWF import CrossClusterForest, sim_data, evaluate_model, plot_results, interpret_results

# Generate example data
data = sim_data(nclusters=15, ncoef=20, ntest=5,
                multimodal=True, n_samples=200, clustszind=2)
clusters_list = data['clusters_list']

# Split into train and test
train_clusters = clusters_list[:-5]
test_clusters = clusters_list[-5:]

# Initialize and fit model
model = CrossClusterForest(
  ntree=100,  # trees for cluster models
  merged_ntree=500,  # trees for merged model
  outcome_col='y',  # name of target variable
  k=10,  # number of clusters
  cluster_ind=1  # use k-means clustering
)

# Fit the model
model.fit(train_clusters, ncoef=20)

# Evaluate
eval_mod = evaluate_model(model, test_clusters)
predictions, improvements, performance = eval_mod['predictions'], eval_mod['improvements'], eval_mod['performance']
# Dataframe of predictions across all test clusters (concatenated):
print(predictions)
#Average percent change in RMSE from baseline Random Forest across all test clusters, for each weighting method
print(improvements)
#Dataframe of average RMSEs per method (column) per test cluster (rows)
print(performance)

#plot improvements:
plot_results(improvements)

#interpret results
interpret_results(improvements)

```

### Using Your Own Data - Pre-clustered Version
If you have multiple pre-existing clusters (e.g., multiple studies or datasets):
In this case, there is the option to use k-means to create new clusters from the concatenated inputted training clusters (cluster_ind = 1), or just use the pre-existing clusters (cluster_ind = 0)
```python
# Prepare your clusters
import pandas as pd
clusters = [
    pd.DataFrame(...),  # Cluster 1
    pd.DataFrame(...),  # Cluster 2
    # ...
]

# Split into train and test
train_clusters = clusters[:-n_test]
test_clusters = clusters[-n_test:]

# Initialize and fit
model = CrossClusterForest(
    ntree=100,
    merged_ntree=500,
    outcome_col='target_column_name',
    k=10,
    cluster_ind=1
)

# Fit
model.fit(train_clusters, ncoef=n_features)

# Predict using different methods
predictions = model.predict(new_data, method='stack_ridge')
```

### Using Your Own Data - Single Dataset Version
If you have a single dataset that you want to fit CCWF to:

```python
from PyCCWF import SingleDatasetForest

# Initialize
model = SingleDatasetForest(
  ntree=100,
  merged_ntree=500,
  outcome_col='target_column_name',
  k=10  # number of clusters
)

# Fit (two options):
# Option 1: X includes the outcome column
model.fit(X)

# Option 2: Separate X and y
model.fit(X, y)

# Predict
predictions = model.predict(X_test, method='stack_ridge')
```

## Methods Available

The package implements four key methods:
- `merged`: Single random forest on merged data
- `unweighted`: Simple average of cluster-specific models
- `stack_ridge`: Ridge regression stacking
- `stack_lasso`: Lasso regression stacking

## Parameters

### CrossClusterForest
- `ntree`: Number of trees in cluster-specific models (default: 100)
- `merged_ntree`: Number of trees in merged model (default: 500)
- `outcome_col`: Name of target variable column
- `k`: Number of clusters for k-means
- `cluster_ind`: Whether to use k-means clustering (1) or not (0)

### SingleDatasetForest
- `ntree`: Number of trees in cluster-specific models (default: 100)
- `merged_ntree`: Number of trees in merged model (default: 500)
- `outcome_col`: Name of target variable column
- `k`: Number of clusters for k-means

## Additional functionality

See more comprehensive examples at https://github.com/m-ramchandran/PyCCWF/tree/main/examples

## Citation

If you use this package, please cite:
@article{ramchandran2021cross,
  title={Cross-cluster weighted forests},
  author={Ramchandran, Maya and Mukherjee, Rajarshi and Parmigiani, Giovanni},
  journal={arXiv preprint arXiv:2105.07610},
  year={2024}
}

## License

MIT License
