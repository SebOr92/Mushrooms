Mushroom Classification using TPOT

In this repository I classified mushrooms by their edibility.
The focus lies on using automated machine learning using the TPOT library.
Therefore, there is only little EDA and preprocessing in the project.

The data doesn't contain missing values and consists purely of categorical data.
Also, the data is almost perfectly balanced.

The features are encoded using scikit-learns OneHotEncoder and LabelEncoder.
Running TPOT takes several minutes. The performance is evaluated using accuracy score and
a confusion matrix.

The best classifier is exported in a serparate file, which contains the steps necessary to reproduce
the results.

The pipeline scores 100% accuracy, therefore there aren't any false negatives or positives in 
the confusion matrix. The confusion matrix is important for evaluation in this task, as 
falsely classifying a mushroom as edible although it actually is poisonous may have bad
consequences for anybody eating it.