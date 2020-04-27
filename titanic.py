
###############################################################
#														      #
# TITANIC Machine learning from disaster       				  # 
# Kaggle competion, following CodeAcademy suggestions		  #
# by Fanny Ummelen											  #
#															  #	
###############################################################

# Load python modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data
passengers = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
submission_file = pd.read_csv("gender_submission.csv")
# Update sex column to numerical
passengers.Sex = passengers.Sex.apply(lambda row: 0 if row == 'male' else 1)

# Fill the nan values in the age column
passengers.Age.loc[passengers.Age.isna()] = passengers.Age.mean()

# Create a first class column
passengers['FirstClass'] = 0
passengers.FirstClass.loc[passengers.Pclass == 1] = 1

# Create a second class column
passengers['SecondClass'] = 0
passengers.FirstClass.loc[passengers.Pclass == 2] = 1

# Select the desired features
features = passengers[['Sex','Age','FirstClass','SecondClass', 'SibSp']]
survival=passengers.Survived

# Perform train, test, split
x_train, x_test, y_train, y_test = train_test_split(features, survival)

# Scale the feature data so it has mean = 0 and standard deviation = 1
scaler = StandardScaler()
scaler.fit_transform(x_train)
scaler.transform(x_test)
# Create and train the model
model = LogisticRegression(solver = 'liblinear')
model.fit(x_train, y_train)

# Score the model on the train data
print('Model score on the training data:')
print(model.score(x_train, y_train))

# Score the model on the test data
print('Model score on the validation data:')
print(model.score(x_test, y_test))

# Analyze the coefficients
print('Model coefficients:')
print(model.coef_)

# Retrain model on complete training set
model.fit(features, survival)

# Prepare the test data
test_data.Sex = test_data.Sex.apply(lambda row: 0 if row == 'male' else 1)
test_data.Age.loc[test_data.Age.isna()] = passengers.Age.mean()
test_data['FirstClass'] = 0
test_data.FirstClass.loc[test_data.Pclass == 1] = 1
test_data['SecondClass'] = 0
test_data.FirstClass.loc[test_data.Pclass == 2] = 1
features_test = test_data[['Sex','Age','FirstClass','SecondClass', 'SibSp']]
scaler.transform(features_test)

# Make preditions for the test data
survival_predicted = model.predict(features_test)

# Put the predictions into the submission file and save it
submission_file.Survived = survival_predicted
submission_file.to_csv('titanic_predictions.csv', index = False)

# added change using github desktop


