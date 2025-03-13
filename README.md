# Neural Network Model Optimization Report

---
## Overview of the Analysis
The purpose of this model analysis report is to outline the reasoning used in the optimization of the to predicting successful campaigns at a 75% or higher accuracy. I used a csv file containing data for more than 34,000 organizations that received funding from the nonprofit foundation Alphabet Soup.

---
## Results

---
### Data Preprocessing

- **Target Variable:** This model is trying to predict successful campaigns by using the `IS_SUCCESSFUL` column found in the dataset, which indicates whether an applicant was successful or not.
    
- **Feature Variables:** The feature variables are all other columns in the pandas DataFrame after removing the required non-variable columns. These features include various applicant attributes such as `APPLICATION_TYPE`, `ASK_AMT`, `CLASSIFICATION`, and `INCOME_AMT`.
    
- **Removed Variables:** Based on the value counts of all columns in the dataset, I was able to identify and remove the non-useful identification columns and remove them from the DataFrame. In the end, only the columns `EIN` and `NAME`.
  
---
### Compiling, Training, and Evaluating the Model

- **Neurons, Layers, and Activation Functions:**
    
    - When I initially ran the model, there was only a basic architecture with three layers. However, when optimizing the model, additional layers and neurons were added to enhance model performance.
        
    - The activation functions used included ReLU for hidden layers and sigmoid for the output layer to handle the binary classification problem. ReLU was chosen to introduce non-linearity and mitigate vanishing gradients, while sigmoid was used for its probabilistic output, aligning with the classification task. Furthermore, I stuck to using the `adam` optimizer when building my models. 
      
- **Model Performance:**
    
    - The initial model did not achieve the desired accuracy. Multiple different approaches were taken in hopes of increasing the prediction accuracy of the model. iterations of feature selection, data balancing, and architectural changes were made in an attempt to improve performance.
      
- **Optimization Attempts:**
    
	1. **Adding Additional Neural Network Layers and Node on top of Feature Selection:** I ran a random forest model on to predict the `IS_SUCCESSFUL` target. I then used [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) to cross validate the models. After doing that, I trained the best-performing model and evaluated the accuracy score. Then, I analyzed which features influenced predictions the most. I then removed low-importance features in an attempt to reduce excessive noise in the dataset. I classified low-importance features as those that contributed less than .05% to the model training. 
        
    2. **SMOTE Oversampling:** Due to the slight imbalance of the target classes, I tried to implement a over-sampling technique to see if that improved the model's accuracy. I used [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) to balance the target variable distribution, but this did not significantly improve accuracy. Since SMOTE did not yield meaningful performance gains, it was removed in subsequent model refinements to simplify preprocessing.
        
    3. **ASK_AMT Binarization:** Due to **extreme** imbalance in `ASK_AMT` where the value 5000 was the only instance of a value count over 5, the column was converted into a binary feature where values were either 5000 or other. The model was then retrained without SMOTE, but accuracy improvements remained marginal. 
      
---
## Summary

---
Despite multiple optimization attempts, including feature selection, oversampling, and feature engineering, the final model did not achieve the target accuracy. The transformation of `ASK_AMT` into a binary variable did not yield significant improvement, and SMOTE failed to enhance performance. Given that SMOTE did not contribute to model improvement, it was removed from later iterations to focus on other optimization strategies.

**Recommendation:** Given the large class imbalances in several features, replacing the neural network with a Random Forest classifier could yield better results. Random Forests can handle imbalanced data more effectively by adjusting class weights. This approach can improve classification performance without the need for artificial data balancing techniques like SMOTE, which did not provide meaningful gains in accuracy.

----
## Citations

---

1. I used [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) in association with [ChatGPT](https://chatgpt.com/) to cross validate the random forest model. This allowed me to find the optimal model and find out the features that most influenced the model's predictions. 
2. I used [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) in association with [ChatGPT](https://chatgpt.com/) to over-sample my imbalanced target variable in an attempt to improve model accuracy.
