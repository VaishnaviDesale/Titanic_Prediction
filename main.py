#Import relevant libraries.
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
#Read csv file into a dataframe and select relevant columns.
df = pd.read_csv("train.csv", header=0)
df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]

#Assign a numerical value to the feature ‘gender’
df['Sex'] = df['Sex'].astype('category')
df["Sex"] = df['Sex'].cat.codes

# For Pclass do one hot encoding
p = pd.get_dummies(df["Pclass"], prefix="class")
df["Pclass_1"] = p["class_1"]
df["Pclass_2"] = p["class_2"]
df["Pclass_3"] = p["class_3"]
df = df.drop('Pclass', axis=1)

# From SibSp and Parch create familysize column.
for i, data in df.iterrows():
    df.at[i, 'FamilySize'] = data['SibSp'] + data['Parch'] + 1
df['FamilySize'] = df['FamilySize'].astype(int)
df = df.drop(['SibSp', "Parch"], axis=1)

# Missing values for age
std_age = df['Age'].std()
avg_age = df['Age'].mean()

for i, data in df.iterrows():
    if pd.isnull(data['Age']):
        r = np.random.uniform(avg_age - std_age, avg_age + std_age)
        r = np.round(r, 0)
        df.at[i, 'Age'] = r

df['Age'].isnull().sum()
print(df)


# Test train splitting
X = df.drop("Survived", axis=1)
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=25)

# Do std scaler operation.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Now divide your data into Training, Test data and apply Naive Bayes classifier on train data and generate the classifier model. use pickle to store classifier model and load that model.

# Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

# Create a Gaussian Classifier
model = GaussianNB()
# Train the model using the training sets
model.fit(X, y)

#lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

# for learning_rate in lr_list:
#     model = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0)
#     model.fit(X_train, y_train)
#
#     print("Learning rate: ", learning_rate)
#     print("Accuracy score (training): {0:.3f}".format(model.score(X_train, y_train)))
#     print("Accuracy score (validation): {0:.3f}".format(model.score(X_test, y_test)))
# model = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_features=2, max_depth=2, random_state=0)
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)
# print("Accuracy score (validation): {0:.3f}".format(model.score(X_test, y_test)))

pickle.dump(model, open('model.pkl', 'wb'))  # write binary

loaded_model = pickle.load(open('model.pkl', 'rb'))



# Generate UI
def predict_input():
    st.header("Survival Prediction")  # display big size font  but smaller than title for Testing
    st.title("Titanic")  # display big size font for Testing
    st.dataframe(df)  # view dataframe in streamlit
    sex = st.radio('Gender', ['Male', 'Female'])  # creates radio button
    pclass = st.selectbox('Passenger Class', [1, 2, 3])  # creates combo box
    age = st.slider('Age', min_value=1, max_value=100)  # creates slider within 10 and 20 value
    fs = st.text_input('Enter Family Size')  # creates a text input box for Name
    fare = st.number_input("Enter Fare")  # creates a number input box for Age
    predict = st.button('Predict')  # creates button called Predict
    try:
        if predict:
            if pclass == 1:
                pclass_1 = 1
                pclass_2 = 0
                pclass_3 = 0
            elif pclass == 2:
                pclass_1 = 0
                pclass_2 = 1
                pclass_3 = 0
            elif pclass == 3:
                pclass_1 = 0
                pclass_2 = 0
                pclass_3 = 1
            if sex == 'Male':
                sex = 1
            elif sex == 'Female':
                sex = 0
            testdata = np.array([[sex, age, fare, pclass_1, pclass_2, pclass_3, fs]])
            classindx = loaded_model.predict(testdata)[0]
            if df.Survived[classindx] == 0:
                st.info("Did Not Survive")
            elif df.Survived[classindx] == 1:
                st.info("Survived")
    except:
        st.error("Invalid Input")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=9000)
