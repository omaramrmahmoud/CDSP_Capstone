import streamlit as st
import numpy as np
import sklearn
import joblib

st.title('CDSP Final Project')
st.title('Healthcare Cost Prediction')

model = joblib.load('model.h5')
scaler = joblib.load('standard_scaler.h5')

# df=pd.read_csv('health_costs.csv')
# X = df.drop('charges', axis=1)
#
# features=X.columns


st.write(f'We used the Healthcare data to predict the cost based on the following independant.')

st.sidebar.write('This is the final project of the CDSP Diploma with Epsilon AI')
st.sidebar.image('logo.jfif')

with st.form(key='my_form'):
    age = st.slider('How old are you?', 0, 130, 25)

    sex = st.selectbox('Select Gender:', ('Male','Female'))

    weight = st.number_input('Insert weight in kgs: ', min_value=1.0, step=0.1)
    height = st.number_input('Insert height in cm', min_value=1.0, step=0.1)

    bmi = round((weight / (height ** 2)) * 10000, 1)

    children = st.slider('How many children do you have?', 0, 10, 3)

    smoker = st.selectbox('Do you smoke?', ('Yes','No'))

    region = st.selectbox('Select Region:', ('Southeast','Southwest','Northeast','Northwest'))

    predict_button = st.form_submit_button(label='Predict')


    def predict_hc(age, sex, bmi, children, smoker, region):

        if (sex.lower() == 'male'):
            sex_male = 1
        else:
            sex_male = 0

        if (smoker.lower() == 'yes'):

            smoker_yes = 1
        else:

            smoker_yes = 0

        if (region.lower() == 'southeast'):

            region_northwest = 0
            region_southeast = 1
            region_southwest = 0

        elif (region.lower() == 'southwest'):
            region_northwest = 0
            region_southeast = 0
            region_southwest = 1

        elif (region.lower() == 'northwest'):
            region_northwest = 1
            region_southeast = 0
            region_southwest = 0


        else:
            region_northwest = 0
            region_southeast = 0
            region_southwest = 0

        scaler.transform([[age, bmi]])

        int_features = np.array(
            [age, bmi, children, sex_male, smoker_yes, region_northwest, region_southeast, region_southwest])
        final_features = int_features.reshape(1, -1)

        health_costs = int(round(model.predict(final_features)[0], 0))

        return (f'Health costs are {health_costs} EGP')

    if predict_button:
        st.write(predict_hc(age,sex,bmi,children,smoker,region))





