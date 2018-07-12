#Question 2
**Answer:**

**Support Vector Machines**
- SVMs are useful for classifying tokenized TF-IDF features derived from text.
- They are effective in highly dimensional domains, particularly where there is linear separability in the data.
- SVMs are slow when using large sample sizes, so they are typically used when samples sizes are under 100k.
- The sample size is only 36k so is well within SVMs capability. There are many features to be considered and SVM can handle the high dimensionality.

**Random Forest**
- Real world example, identifying fruit based on features such as color, size, shape.
- Any decision tree application can be enhanced by using RF, as an RF is multiple decision trees with different initial conditions allowing for a more robust prediction.
- The RF is susceptible to overfitting as it continues to split data completely. This needs to be addressed with pruning or limiting the number of split levels.
- RF is always a good classifier to include in initial analysis as it is simple to implement and easy to interpret.

**K Nearest Neighbors**
- Real world example, identifying customer segments based on purchase data to improve business decisions.
- KNN groups similar data points, allowing for robust pattern identification within the dataset.
- If there is overlap in the data classification it is difficult/impossible for KNN to correctly classify points in the overlapped region.
- KNN is a standard and common classification method and is useful to include in initial analysis.
