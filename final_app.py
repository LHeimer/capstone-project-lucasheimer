#Import necessary libraries
from scipy import stats as st
import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
import statsmodels as sm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
## this sets some style parameters
sns.set()
st.set_page_config(layout="wide")
#demographics: Age, Gender, Ethnicity, Education Level
#lifestyle: BMI, Smoking, Alcohol Consumption, Physical Activity, DietQuality, Sleep quality
#medical history: family history, cardiovascular disease, diabetes, depression, headinjury, hypertension
#clinical measurements:systolic bp, diastolic bp, cholesterol total,  cholesterol ldl, cholesterol hdl, cholesterol triglycerides
#cognitive and functional: mmse, functional, memory, behavioral, adl
#symptoms: confusion, disorientation, personalitychanges, difficultycompleting tasks, forgetfulness



#Read in the data and clean
df = pd.read_csv('https://raw.githubusercontent.com/LHeimer/final/refs/heads/main/alzheimers_disease_data.csv')
df_copy = df.drop(['PatientID','DoctorInCharge'], axis = 1)

#Function for normalizing the data
def normalize(df, column_list):
  for column in column_list:
    df[column] = (df[column] - np.mean(df[column])) / np.std(df[column])
  return df

#Normalize the dataset
normalize(df_copy, ["Age", "Ethnicity", "EducationLevel", "BMI", "AlcoholConsumption", "PhysicalActivity", "DietQuality", "SleepQuality", "SystolicBP", "DiastolicBP", "CholesterolTotal", "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides","MMSE", "FunctionalAssessment","ADL"])

#Divide the dataset into subcategories
demo_df = df_copy[['Age', 'Gender', 'Ethnicity', 'EducationLevel', 'Diagnosis']].copy()
lifestyle_df = df_copy[['BMI', 'Smoking', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality', 'Diagnosis']].copy()
med_history_df = df_copy[['FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes', 'Depression', 'HeadInjury', 'Hypertension', 'Diagnosis']].copy()
measure_df = df_copy[['SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 'Diagnosis']].copy()
assessment_df = df_copy[['MMSE', 'FunctionalAssessment', 'MemoryComplaints', 'BehavioralProblems', 'ADL', 'Diagnosis']].copy()
symptoms_df = df_copy[['Confusion', 'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks', 'Forgetfulness', 'Diagnosis']].copy()

#Create a function that passes the dataframes through the neural network
def evaluate_df(df):
  column_names = df.columns
  column_names = column_names.to_list()
  column_names.remove('Diagnosis')
  num_inputs = len(column_names)
  #Create a test and a training set
  trainFraction = .75

  sample = np.random.uniform(size = 2149) < trainFraction
  trainingDat = df[sample]
  testingDat = df[~sample]
  #Define the predictors and outputs
  test_var = 'Diagnosis ~ '
  for i in range(len(column_names)):
    if i == len(column_names)-1:
      test_var = test_var + column_names[i]
    else:
      test_var = test_var + column_names[i] + ' + '
  #Statsmodel for logistic regression
  fit = smf.logit(test_var, data = trainingDat).fit()


  yhat = 1 / (1 + np.exp(-fit.fittedvalues))
  #Format the data to be used in pytorch
  n = df.shape[0]
  training_inputs_list = [0]*num_inputs
  testing_inputs_list = [0]*num_inputs
  for i in range(num_inputs):
    training_inputs_list[i] = torch.from_numpy(trainingDat[column_names[i]].values)
    testing_inputs_list[i] = torch.from_numpy(testingDat[column_names[i]].values)
  for j in range(len(training_inputs_list)):
    training_inputs_list[j] = training_inputs_list[j].float()
    testing_inputs_list[j] = testing_inputs_list[j].float()
    training_inputs_list[j] = training_inputs_list[j].unsqueeze(1)
    testing_inputs_list[j] = testing_inputs_list[j].unsqueeze(1)
  training_output_list = torch.from_numpy(trainingDat['Diagnosis'].values)
  testing_output_list = torch.from_numpy(testingDat['Diagnosis'].values)
  training_output_list = training_output_list.float()
  testing_output_list = testing_output_list.float()
  training_output_list = training_output_list.unsqueeze(1)
  testing_output_list = testing_output_list.unsqueeze(1)
    
  #Define the neural network model
  class LogisticRegression(torch.nn.Module):
     def __init__(self):
        super(LogisticRegression, self).__init__()
        self.fc1 = torch.nn.Linear(num_inputs, 3, bias = True)
        self.fc2 = torch.nn.Linear(3, 1, bias=True)
     def forward(self, predict_list):
        xyz = torch.cat(predict_list, 1)
        xyz = F.relu(self.fc1(xyz))
        y_pred = torch.sigmoid(self.fc2(xyz))
        return y_pred

  model = LogisticRegression()

  ## BCE is the loss function
  loss_fn = torch.nn.BCELoss()

  ## Set the optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

  ## Loop over iterations
  for t in range(500):

    ## Forward propagation
    y_pred = model(training_inputs_list)

    ## the loss for this iteration
    loss = loss_fn(y_pred, training_output_list)

    ## Zero out the gradients before adding them up
    optimizer.zero_grad()

    ## Backprop
    loss.backward()

    ## Optimization step
    optimizer.step()
  
  #Steps used for verification during development
  #ytest = model(training_inputs_list)
  #ytest = ytest.detach().numpy().reshape(-1)
  #graph = plt.plot(yhat, ytest,  ".")
  #plt.plot([0, 1], [0, 1], linewidth=2)
  
  return fit, model
#Pass each dataset through the neural network
overall = evaluate_df(df_copy)
demograph = evaluate_df(demo_df)
measurements = evaluate_df(measure_df)
lifestyle = evaluate_df(lifestyle_df)
med_history = evaluate_df(med_history_df)
assessment = evaluate_df(assessment_df)
symptoms = evaluate_df(symptoms_df)


#Function for transforming dataframes to tensors. Will be used on the input dataset
def convert_tensor(df):
    column_names = df.columns
    column_names = column_names.to_list()
    num_inputs = len(column_names)
    n = df.shape[0]
    training_inputs_list = [0]*num_inputs
    for i in range(num_inputs):
      training_inputs_list[i] = torch.from_numpy(df[column_names[i]].values)
    for j in range(len(training_inputs_list)):
      training_inputs_list[j] = training_inputs_list[j].float()
      training_inputs_list[j] = training_inputs_list[j].unsqueeze(0)
    return training_inputs_list

#Build the app framework
st.title("Alzheimer's Prediction App")
overall_metric = st.container()

col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)
col1.subheader("Demographics")
col2.subheader("Lifestyle Factors")
col3.subheader("Family Medical History")
col4.subheader("Clinical Measurements")
col5.subheader("Cognitive and Functional Assessments")
col6.subheader("Symptoms")
with col1:
    age = st.number_input("Age", min_value = 10, max_value = 90, value=50)
    gender = st.selectbox("Gender", ("Male", "Female"))
    if(gender == "Male"):
        gender_int = 0
    elif(gender == "Female"):
        gender_int = 1
    ethnicity = st.selectbox("Ethnicity", ("Caucasian", "African American", "Asian", "Other"))
    if(ethnicity == "Caucasian"):
        eth_int = 0
    elif(ethnicity == "African American"):
        eth_int = 1
    elif(ethnicity == "Asian"):
        eth_int = 2
    elif(ethnicity == "Other"):
        eth_int = 3
    education = st.selectbox("Highest Education Level", ("None", "High School", "Bachelor's", "Higher"), index = 1)
    if(education == "None"):
        edu_int = 0
    elif(education == "High School"):
        edu_int = 1
    elif(education == "Bachelor's"):
        edu_int = 2
    elif(education == "Higher"):
        edu_int = 3
    demo_metric = st.container(border=True)
def presence(var):
    if(var == "Yes"):
        var_int = 1
    elif(var == "No"):
        var_int = 0
    return var_int
with col2:
    bmi = st.number_input("BMI", min_value = 15, max_value = 40, value=22)
    smoking = st.radio("Do you smoke?", ("Yes", "No"), index=1)
    smoker = presence(smoking)
    alcohol = st.number_input("Average Number of Alcoholic Drinks consumed per week", min_value = 0, max_value = 20)
    activity = st.number_input("Avergae number of hours of physicial activity per week", min_value = 0, max_value = 20, value=3)
    diet = st.number_input("Diet Quality Score", min_value = 0, max_value = 10, value=5)
    sleep = st.number_input("Sleep Quality score", min_value = 4, max_value = 10, value=8)
    lifestyle_metric = st.container(border=True)
    
with col3:  
    family_history = st.radio("Presence of Alzheimers in family history", ("Yes", "No"), index = 1)
    fam_hist = presence(family_history)
    cardiovascular = st.radio("Diagnosis of Cardiovascular Disease", ("Yes", "No"), index = 1)
    cardio = presence(cardiovascular)
    diabetes = st.radio("Diagnosis of Diabetes", ("Yes", "No"), index = 1)
    diab = presence(diabetes)
    depression = st.radio("Diagnosis of Depression", ("Yes", "No"), index = 1)
    depr = presence(depression)
    head_injury = st.radio("History of head injuries", ("Yes", "No"), index = 1)
    inj = presence(head_injury)
    hypertension = st.radio("Diagnosis of hypertension", ("Yes", "No"), index = 1)
    hyper = presence(hypertension)
    med_metric = st.container(border=True)
    
with col4:    
    systolic = st.number_input("Systolic BP", min_value = 90, max_value = 180, value=110)
    diastolic = st.number_input("Diastolic BP", min_value = 60, max_value = 120, value=70)
    chol_total = st.number_input("Total Cholesterol", min_value = 125, max_value = 300, value=150)
    ldl = st.number_input("LDL Cholesterol", min_value = 20, max_value = 250, value=50)
    hdl = st.number_input("HDL Cholesterol", min_value = 20, max_value = 100, value=80)
    triglyc = st.number_input("Triglycerides", min_value = 20, max_value = 500, value=100)
    measure_metric = st.container(border=True)
    
with col5:  
    mmse = st.number_input("MMSE Score", min_value = 0, max_value = 30, value=25)
    function_assess = st.number_input("Functional Assessment", min_value = 0, max_value = 10, value=8)
    memory = st.radio("Presence of memory loss", ("Yes", "No"), index = 1)
    mem_loss = presence(memory)
    behavior = st.radio("Behavioral Problems", ("Yes", "No"), index = 1)
    behav_prob = presence(behavior)
    adl = st.number_input("Daily Living Score", min_value = 0, max_value = 10, value=6)
    assessment_metric = st.container(border=True)
    
with col6:    
    confusion = st.radio("Presence of confusion", ("Yes", "No"), index = 1)
    conf = presence(confusion)
    disorientation = st.radio("Presence of disorientation", ("Yes", "No"), index = 1)
    disorient = presence(disorientation)
    personal_changes = st.radio("Presence of personality changes", ("Yes", "No"), index = 1)
    pers_change = presence(personal_changes)
    diff_tasks = st.radio("Difficulty completing tasks", ("Yes", "No"), index = 1)
    tasks = presence(diff_tasks)
    forgetfulness = st.radio("Presence of forgetfulness", ("Yes", "No"), index = 1)
    forget = presence(forgetfulness)
    symptoms_metric = st.container(border=True)

#Create the input dataframe
column_names = df.columns
column_names = column_names.to_list()
column_names.remove('PatientID')
column_names.remove('Diagnosis')
column_names.remove('DoctorInCharge')
user_data_list = [age, gender_int, eth_int, edu_int, bmi, smoker, alcohol, activity, diet, sleep, fam_hist, cardio, diab, depr, inj, hyper, systolic, diastolic, chol_total, ldl, hdl, triglyc, mmse, function_assess, mem_loss, behav_prob, adl, conf, disorient, pers_change, tasks, forget]
user_data = pd.DataFrame([user_data_list],columns = column_names)

#Function to normalize the input dataframe with the training dataset
def normalize_input(input_df, df, column_list):
  for column in column_list:
    input_df[column] = (input_df[column] - np.mean(df[column])) / np.std(df[column])
  return input_df

#Normalize the inputs
user_data = normalize_input(user_data, df, ["Age", "Ethnicity", "EducationLevel", "BMI", "AlcoholConsumption", "PhysicalActivity", "DietQuality", "SleepQuality", "SystolicBP", "DiastolicBP", "CholesterolTotal", "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides","MMSE", "FunctionalAssessment","ADL"])

#Divide the user inputs into subcategories
user_demo_df = user_data[['Age', 'Gender', 'Ethnicity', 'EducationLevel']].copy()
user_lifestyle_df = user_data[['BMI', 'Smoking', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality']].copy()
user_med_history_df = user_data[['FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes', 'Depression', 'HeadInjury', 'Hypertension']].copy()
user_measure_df = user_data[['SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides']].copy()
user_assessment_df = user_data[['MMSE', 'FunctionalAssessment', 'MemoryComplaints', 'BehavioralProblems', 'ADL']].copy()
user_symptoms_df = user_data[['Confusion', 'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks', 'Forgetfulness']].copy()

#Convert the user inputs into tensors
user_overall_tensor = convert_tensor(user_data)
user_demo_tensor = convert_tensor(user_demo_df)
user_lifestyle_tensor = convert_tensor(user_lifestyle_df)
user_med_tensor = convert_tensor(user_med_history_df)
user_measure_tensor = convert_tensor(user_measure_df)
user_assessment_tensor = convert_tensor(user_assessment_df)
user_symptoms_tensor = convert_tensor(user_symptoms_df)

#Obtain the neural network model for each category
overall_model = overall[1]
demo_model = demograph[1]
life_model = lifestyle[1]
med_model = med_history[1]
measure_model = measurements[1]
assessment_model = assessment[1]
symptoms_model = symptoms[1]

#Obtain the predicted output by passing each input tensor through the respective model
overall_prediction = overall_model(user_overall_tensor)
overall_prediction = overall_prediction.detach().numpy()*100
overall_prediction = str(np.round(float(overall_prediction[0]))) + "%"
demo_prediction = demo_model(user_demo_tensor)
demo_prediction = demo_prediction.detach().numpy()*100
demo_prediction = str(np.round(float(demo_prediction[0]))) + "%"
life_prediction = life_model(user_lifestyle_tensor)
life_prediction = life_prediction.detach().numpy()*100
life_prediction = str(np.round(float(life_prediction[0]))) + "%"
med_prediction = med_model(user_med_tensor)
med_prediction = med_prediction.detach().numpy()*100
med_prediction = str(np.round(float(med_prediction[0]))) + "%"
measure_prediction = measure_model(user_measure_tensor)
measure_prediction = measure_prediction.detach().numpy()*100
measure_prediction = str(np.round(float(measure_prediction[0]))) + "%"
assessment_prediction = assessment_model(user_assessment_tensor)
assessment_prediction = assessment_prediction.detach().numpy()*100
assessment_prediction = str(np.round(float(assessment_prediction[0]))) + "%"
symptoms_prediction = symptoms_model(user_symptoms_tensor)
symptoms_prediction = symptoms_prediction.detach().numpy()*100
symptoms_prediction = str(np.round(float(symptoms_prediction[0]))) + "%"

#Obtain the coefficients from each model
demo_coef = demograph[0].params
life_coef = lifestyle[0].params
measure_coef = measurements[0].params
med_coef = med_history[0].params
assess_coef = assessment[0].params
symptoms_coef = symptoms[0].params

#Function to convert the logits to probabilities
def logittoprob(data, coef):
  df = pd.DataFrame(coef).reset_index()
  df.columns = ['Input', 'Beta']
  df['X'] = 0
  df['Probability'] = 0
  for column in data:
    df.loc[df['Input'] == column, 'X'] = data[column].values[0]
  beta0 = df.iloc[0,1]
  for i in range(len(df)-1):
    beta1 = df.iloc[i+1,1]
    x = df.iloc[i+1,2]
    prob = 1/(1+np.exp(-(beta0+beta1*x)))
    prob = prob*100
    df.iloc[i+1,3] = prob
  df.drop(0, inplace = True)
  return df

#Get a dataframe showing the probabilities for each factor within a subcategory
demo_prob = logittoprob(user_demo_df, demo_coef)
life_prob = logittoprob(user_lifestyle_df, life_coef)
measure_prob = logittoprob(user_measure_df, measure_coef)
med_prob = logittoprob(user_med_history_df, med_coef)
assess_prob = logittoprob(user_assessment_df, assess_coef)
symptoms_prob = logittoprob(user_symptoms_df, symptoms_coef)

#Function to create plots of probabilities
def plot_prob(df):
  fig = go.Figure()
  fig.add_trace(go.Bar(y = df.Input, x = df.Probability, orientation='h'))
  fig.update_layout(title='Individual Factors', xaxis_title='Probability (%)', yaxis_title='Factor')
  return fig

#Plot the probabilities for each factor in each subcategory
demo_fig = plot_prob(demo_prob)
life_fig = plot_prob(life_prob)
measure_fig = plot_prob(measure_prob)
med_fig = plot_prob(med_prob)
assess_fig = plot_prob(assess_prob)
symptoms_fig = plot_prob(symptoms_prob)

#Display the subcategory probability and the plots
with overall_metric:
    st.metric("Probability of Alzheimer's Diagnosis", overall_prediction)
with demo_metric:
    st.metric("Probability of Alzheimer's Diagnosis", demo_prediction)
    st.plotly_chart(demo_fig)
with lifestyle_metric:
    st.metric("Probability of Alzheimer's Diagnosis", life_prediction)
    st.plotly_chart(life_fig)
with med_metric:
    st.metric("Probability of Alzheimer's Diagnosis", med_prediction)
    st.plotly_chart(med_fig)
with measure_metric:
    st.metric("Probability of Alzheimer's Diagnosis", measure_prediction)
    st.plotly_chart(measure_fig)
with assessment_metric:
    st.metric("Probability of Alzheimer's Diagnosis", assessment_prediction)
    st.plotly_chart(assess_fig)
with symptoms_metric:
    st.metric("Probability of Alzheimer's Diagnosis", symptoms_prediction)
    st.plotly_chart(symptoms_fig)

