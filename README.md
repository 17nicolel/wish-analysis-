# wish-analysis
Forecasting product units sold on Wish using Machine Learning. 


# Goal
To predict which products will have the highest potential to be sold in relation to tags used. To investigate if the correlation between units sold and tags is more about the specific tags used instead of quantity used. This information would be valuable because a business could use the model to decide if they see the potential in a product so that they could choose to investigate advertisements. 

# Method 
A logistic regression model is created to analyze the correlation between tags and units sold per item. My hypothesis is that: the occurrence of certain keywords will have certain contributions to units sold. A logistic regression model is appropriate here because the x variables (tags) are categorical, and the y variable (if the product could become popular) is also categorical. 

Due to the fact that the data was originally not cleaned and organized, the tags column (x variables) for the products table is first split into different columns, each containing a separate tag, then transformed into dummy variables, which in total consists of 529 dummy variables that each represent a different tag. The y variable is defined to be whether a product could be sold above 100 units, with True as sold above 1000 units and False as sold below 1000 units. 

<img src="https://user-images.githubusercontent.com/65926359/101327034-fa0afe00-3822-11eb-9e98-1c8a270cf361.png" width="90%"></img> 

(The above screenshot shows the output of the columns, after transforming string variables into dummy variables. A 0 represents the tag not present, and a 1 represents the tag present)

After the data is split to 70% training, and 30% testing, the data is trained according to the training set, and it produces an accuracy score of 0.6292372881355932 for the testing set. Although this accuracy level is not significantly high, it is higher than a 50% guess.

Confusion matrix:

<img src="https://user-images.githubusercontent.com/65926359/101327079-0727ed00-3823-11eb-8d12-9db0f827b55e.png" width="45%"></img> 

The confusion matrix shows the labels that could provide further insight on the accuracy rate. The diagonal from top left to bottom right shows when the prediction is right and the opposite diagonal shows when the prediction is wrong. It is observed that a high portion of inaccuracies is due to the False Positive label (23%). This means that there might be some products that will not perform as well as predicted. However, the True positive rate is also high with a 48%, which means that business will be less likely to miss out potential products that will perform as predicted. Therefore, this model could be qualified to use as a reference for business decisions. 


# Outcome
For the implementation of this model, a list could be created to represent the present or non-present of tags used for a product. The list should contain 529 numbers, with 0 representing a tag not being present, and 1 representing a tag being present. 

<img src="https://user-images.githubusercontent.com/65926359/101327144-232b8e80-3823-11eb-9faf-82d349f93ea9.png" width="90%"></img> 
(The example above shows a random generated list that could represent a possible tags used on a product, and for this hypothetical product, the prediction is that it could be sold above 1000 units.)

Nevertheless, I also acknowledge that there may be some potential incompetence of this model. First of all, due to the fact that there are only 1573 rows of data available in the data set, the size of the data set might not be large enough to train a very accurate model. Secondly, because the goal of this model is to examine the relationship between specific tags used and the potential popularity of the product, I did not take into account other variables, such as the product’s price, discount, ads used on the product and etc. While specific tags used alone may not be causal to the result, a more complete model should take into account more variables. 

