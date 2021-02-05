import streamlit as st
import pickle
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
st.title('Employee Churn Prediction')
im = Image.open("employee_churn.jpg")
st.image(im, width=700)
model = pickle.load(open("final_rf_model", "rb"))
st.sidebar.header("Select the Employee Features:")
satisfaction_level = st.sidebar.slider("Please select the satisfaction level of employee", 0.0, 1.0, 0.5, 0.01)
last_evaluation = st.sidebar.slider("Please select the last evaluation level of employee", 0.0, 1.0, 0.5, 0.01)
average_monthly_hour = st.sidebar.slider("Please select the number of hours in average an employee work in a month",
                                         140, 320, 160, 1)
time_spend_company = st.sidebar.slider("Please select the number of years an employee work in company", 1, 15, 5, 1)
project_number = st.sidebar.slider("Please select the number of projects assigned to employee", 1, 10, 5, 1)
productivity = project_number / time_spend_company
work_accident = st.sidebar.slider("Please select whether an employee had an accident or not", 0, 1, 0, 1)
salary_list = ['low', 'medium', 'high']
salary = st.sidebar.selectbox("Please select salary level of employee", salary_list)
my_dict = {"satisfaction_level": satisfaction_level,
           "last_evaluation": last_evaluation,
           "average_monthly_hour": average_monthly_hour,
           "time_spend_company": time_spend_company,
           "productivity": productivity,
           'work_accident': work_accident,
           'salary': salary}
df = pd.DataFrame.from_dict([my_dict])
if salary == 'low':
    df[['salary']] = 0
elif salary == 'medium':
    df[['salary']] = 1
else:
    df[['salary']] = 2
def single_customer():
    df_table = df
    st.write('')
    st.dataframe(data=df_table, width=700, height=400)
    st.write('')
single_customer()
if st.button("Submit"):
    import time
    with st.spinner("Random Forest Model is loading..."):
        my_bar = st.progress(0)
        for p in range(0, 101, 10):
            my_bar.progress(p)
            time.sleep(0.1)
    churn_probability = model.predict_proba(df)
    st.markdown(churn_probability)
    st.success(f'The Probability of the Employee Churn is %{round(churn_probability[0][1] * 100, 1)}')
    if round(churn_probability[0][1] * 100, 1) > 50:
        st.warning("The Employee is Left")
    else:
        st.success("The Employee is NOT Left")

