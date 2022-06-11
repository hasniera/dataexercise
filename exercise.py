import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

st.write("""
# Sample of student enroll private school
This app predicts the number of student enroll private school!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    accept = st.sidebar.slider('accept', 100, 1000, 10000)
    enroll = st.sidebar.slider('enroll', 50, 500, 5000)
    f_undergrad = st.sidebar.slider('f_undergrad', 100, 1000, 10000)
    p_undergrad = st.sidebar.slider('p_undergrad', 0, 500, 5000)
    phd = st.sidebar.slider('phd', 20, 50, 100)
    data = {'accept': accept,
            'enroll': enroll,
            'f_undergrad': f_undergrad,
            'p_undergrad': p_undergrad,
            'phd': phd}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

url = "https://raw.githubusercontent.com/hasniera/dataexercise/main/data%20(1).csv"
private = pd.read_csv(url)

X = private[['private', 'apps', 'accept', 'enroll', 'top10per', 'top25per', 'f_undergrad', 'p_undergrad', 'outstate', 'room_board', 'books', 'personal', 'phd', 'terminal', 's_f_ratio', 'perc_alumni', 'expend']]
Y = private['grad_rate']

pss = RandomForestClassifier()
pss.fit(X, Y)

prediction = pss.predict(df)
prediction_proba = pss.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(private['grad_rate'].unique())
#st.write(['Iris-setosa','Iris-versicolor','Iris-virginica')]

st.subheader('Prediction')
#st.write(iris.target_names[prediction])
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
