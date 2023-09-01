# thesis

#### Author: Ida Thrane (idth@itu.dk)

This repository contains all code to the thesis project "Modelling Street Suitability for Tree Planting". Data is not stored here, but is obtainable through OpenStreetMap (OSM) and Opendata DK (link to dataset of trees in Copenhagen Municipality: https://www.opendata.dk/city-of-copenhagen/gadetraeer)

## Files in repository 
The files contained in this repository are the following: 
- **intersections.ipynb**: Notebook to create the spatial footprint features. Notebook used to create polygons around each street in Copenhagen Municipality, and then intersect them with all points and polygons which describe Copenhagen Municipality obtained as a download of Denmark in .shp-files from Geofabrik (https://download.geofabrik.de/europe/denmark.html) and preprocessed in QGIS to only contain points in Copenhagen Municipality.
- **preprocess.ipynb**: Preprocess the tree dataset from Opendata DK, streets from OSM and the spatial footprint features created in intersections.ipynb. Additionally create distance_from_center feature. Creates the final datasets for training and testing for each experiment.
- **models.py**: Script to initialize random classifier, decision tree, DNN and CNN.
- **decision_tree_optimization.ipynb**: Optimization of hyperparameters of the decision tree model.
- **nn_optimization.ipynb**: Optimization of hyperparameters of the DNN model.
- **cnn_optimization.ipynb**: Optimization of hyperparameters of the CNN model.
- **model_testing.ipynb**: Training and testing of each model. Running the three experiments mentioned in the report.
- **shap.ipynb**: SHAP-analysis of the predictions.

## Data:
- **spatial_features/**: spatial feature files used for preprocessing.
- **graphs/**: graph of Copenhagen Municipality, street network and Geodataframe used for preprocessing. 
- **data/**: preprocessed data used as machine learning input for the three experiments.
- **data/predictions/**: predictions made


### Visualization notebooks: 
- **correlation_matrix.ipynb**: Create correlation matrix used in report.
- **prediction_insights.ipynb**: Heatmaps and stacked bar plots used in report.
- **viz.ipynb**: Additional visualizations.
- **misclassifications.ipynb**: misclassification analysis of the predictions.

