import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

#DATA
file_path = 'social_media_usage.csv'
s = pd.read_csv(file_path)
def clean_sm(x):
    return np.where(x == 1, 1, 0)
ss = s.copy()
ss['sm_li'] = clean_sm(ss['web1h'])
features = ['income', 'educ2', 'par', 'marital', 'gender', 'age']
ss = ss[features + ['sm_li']]
ss['gender'] = np.where(ss['gender'] == 2, 1, 0)
ss['par'] = np.where(ss['par'] == 1, 1, 0)
ss['marital'] = np.where(ss['marital'] == 1, 1, 0)
ss['educ2'] = np.where((ss['educ2'] >= 1) & (ss['educ2'] <= 8), ss['educ2'], np.nan)
ss['income'] = np.where((ss['income'] >= 1) & (ss['income'] <= 9), ss['income'], np.nan)
ss = ss.dropna()
#Train - Test Split
y = ss['sm_li']
X = ss[['income', 'educ2', 'par', 'marital', 'gender', 'age']]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=876)

# LR Model
lr = LogisticRegression(class_weight='balanced', random_state=36)
lr.fit(X_train, y_train)
# Streamlit
st.title("Are you a LinkedIn User?")
st.markdown("""
    ## LinkedIn Usage Prediction App
    This app is designed to predict whether an individual uses LinkedIn based on certain features. 
    Click on the arrow in the top left corner to reveal the sidebar,  select options and click "Predict" to see the result!
""")
st.markdown('See your results below')



income_mapping = {1: 'Less than $10,000', 2: '$10,000 - $19,999', 3: '$20,000 - $29,999', 
                  4: '$30,000 - $39,999', 5: '$40,000 - $49,999', 6: '$50,000 - $74,999', 
                  7: '$75,000 - $99,999', 8: '$100,000 - $149,999', 9: '$150,000 or more'}

education_mapping = {1: 'Less than high school', 2: 'High school incomplete', 3: 'High school graduate', 
                     4: 'Some college, no degree', 5: 'Two-year associate degree', 
                     6: 'Four-year college or university degree', 7: 'Some postgraduate or professional schooling', 
                     8: 'Postgraduate or professional degree'}

marital_mapping = {1: 'Married', 2: 'Living with a partner', 3: 'Divorced', 4: 'Separated', 
                   5: 'Widowed', 6: 'Never been married'}
gender_mapping = {1: 'Female', 2: 'Male'} 
parent_mapping = {1: 'No', 0: 'Yes'}
st.sidebar.title("Input your Information here!")
#Mapping Connector
income = st.sidebar.selectbox('Income:', list(income_mapping.keys()), format_func=lambda x: income_mapping[x])
education = st.sidebar.selectbox('Education:', list(education_mapping.keys()), format_func=lambda x: education_mapping[x])
parent = st.sidebar.radio('Parent (Yes or No):', list(parent_mapping.keys()), format_func=lambda x: parent_mapping[x])
marital = st.sidebar.selectbox('Marital Status:', list(marital_mapping.keys()), format_func=lambda x: marital_mapping[x])
gender = st.sidebar.radio('Gender (Female or Male):', list(gender_mapping.keys()), format_func=lambda x: gender_mapping[x])
age = st.sidebar.slider('Age:', 1, 98, 25)

# Predict Button
predict_button = st.sidebar.button("Predict", key="prediction_button")

if predict_button:
    income = int(income)
    education = int(education)
    parent = int(parent)
    marital = int(marital)
    gender = int(gender)
    age = int(age)

    new_data = np.array([income, education, parent, marital, gender, age]).reshape(1, -1)
    prediction = lr.predict(new_data)
    probability = lr.predict_proba(new_data)[:, 1]

    st.header("Prediction Results")
    
    #Results
    st.write("Probability:", f"{round(probability[0] * 100, 2)}%")
    st.write("LinkedIn User:", 'Yes' if prediction[0] == 1 else 'No')

    #Display User Input Under Result
    st.subheader("User Details:")
    st.write(f"- **Income:** {income_mapping[income]}")
    st.write(f"- **Education:** {education_mapping[education]}")
    st.write(f"- **Parent:** {parent_mapping[parent]}")
    st.write(f"- **Marital Status:** {marital_mapping[marital]}")
    st.write(f"- **Gender:** {gender_mapping[gender]}")
    st.write(f"- **Age:** {age} years old")

st.markdown('---')


st.header('Analysis plots: Check out which factors have an impact on LinkedIn usage')

st.markdown('---')


st.subheader("Marital Status vs. Income with LinkedIn Usage")
#Plot 1
fig, ax1 = plt.subplots()
grouped_data = ss.groupby('sm_li').agg({'income': 'mean', 'educ2': 'mean'})
sns.lineplot(data=grouped_data, x=grouped_data.index, y='income', marker='o', label='Average Income')
ax2 = ax1.twinx()
sns.lineplot(data=grouped_data, x=grouped_data.index, y='educ2', marker='s', color='blue', label='Average Education')
ax1.set_xlabel('LinkedIn User (0: No, 1: Yes)')
ax1.set_ylabel('Average Income', color='black')
ax2.set_ylabel('Average Education', color='black')
plt.title('Comparison of Average Income and Education by LinkedIn Usage')
plt.legend(loc='upper left')
st.pyplot(fig)

st.subheader("Income usage vs. LinkedIn Usage with Age")

#Plot 2
#Age Filter
age_filter = st.slider('Filter by Age:', 18, 90, 25)
filtered_ss = ss[(ss['age'] >= age_filter) & (ss['age'] <= 90)]


# Mapping
income_mapping = {1: 'Less than $10,000', 2: '$10,000 - $19,999', 3: '$20,000 - $29,999',
                  4: '$30,000 - $39,999', 5: '$40,000 - $49,999', 6: '$50,000 - $74,999',
                  7: '$75,000 - $99,999', 8: '$100,000 - $149,999', 9: '$150,000 or more'}
# Income Mapping
filtered_ss['income'] = filtered_ss['income'].map(income_mapping)

fig, ax = plt.subplots()
sns.countplot(x='income', hue='sm_li', data=filtered_ss, palette=['#3A1078', '#2F58CD'], ax=ax)
ax.set_xlabel('Income')
ax.set_ylabel('Count')
plt.title(f'Distribution of LinkedIn Usage by Income (Age: {age_filter}-90)')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
st.pyplot(fig)


#FEEDBACK
feedback_list = []
# User feedback 
user_feedback = st.text_area("Feedback:", "Share your thoughts...")
# Submit Feedback button
if st.button("Submit Feedback"):
    feedback_list.append(user_feedback)
    st.success("Thank you for your feedback!")
#Developer
@st.cache(allow_output_mutation=True)
def clear_feedback(developer_code):
    if developer_code == "secret_code" and st.button("Clear Feedback (Developer Only)"):
        feedback_list.clear()
        st.success("Feedback has been cleared.")
#Developer Center
developer_code = st.text_input("Developer Code:", type="1234")
clear_feedback(developer_code)
# Display Reviews
st.subheader("User Feedback:")
for feedback in feedback_list:
    st.write(feedback)
