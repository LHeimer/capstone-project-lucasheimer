# Capstone Project: Alzheimer's Diagnosis Prediction App
## Rationale
Alzheimer's Disease is the most common form of dementia among the elderly population worldwide. It is a neurodegenerative disease that often results in memory loss, difficulty completing daily tasks, and personality changes. Diagnosis of Alzheimer's usally requires anything from neurological exams to brain imaging. Although the exact cause of the disease is unknown research has found there are many lifestyle factors beyond increased age that affect the potential onset of the disease. The goal of this app is to combine data from both lifestyle factors and clinical testing to predict the likelihood of an Alzheimer's Disease diagnosis. Knowledge about an individual's risk for developing this form of dementia may allow him/her to make necessary lifestyle changes to prevent or slow the progression of its onset.
## Description
### Model
This app was created to predict the probability of a Alzheimer's Diagnosis using standard biometric and demographic data. The data used for this model was synthetically developed, however many of the assessments and parameters are common standards used to gauge the onset of Alzheimer's. Prior to inserting it into the model, the data was cleaned by removing irrelevant information and then it was normalized. A logistic regression model was defined by passing the inputs through a linear neural network layer followed by applying a sigmoid function to classify the outcome as either a positive or negative diagnosis. In addition to modeling an overall probability of diagnosis using all the metrics as inputs, the data was divided into smaller groups to examine the how each subcategory influenced the overall outcome. A new dataframe was created for each subgroup and the same logistic regression model was used to create a diagnosis prediction. The app also displays a bar chart outlining the individual probabilities for each variable within the subcategory. This allows the user to alter the variables and observe in realtime how an increase or decrease in certain factors may affect the likelihood of a positive diagnosis. The breakout of subcategories is outlined in the following table:
|Demographic Information   |Lifestyle Factors    |Family Medical History    |Clinical Measurements   |Cognitive and Functional Assessments   |Symptoms  |
| ---- | ---- | ---- | ---- | ---- | ---- |
|Age| BMI| Alzheimer's| Systolic BP| MMSE| Confusion|
|Gender| Smoking| Cardiovascular Disease| Diastolic BP| Functional Assessment| Disorientation|
|Ethnicity| Weekly Alcohol Consumption| Diabetes| Total Cholesterol| Memory Complaints| Personality Changes|
|Education Level| Weekly Hours of Activity| Depression| LDL Cholesterol| Behavioral Problems| Difficulty Completing Tasks|
|     |  Diet Quality| Head Injury| HDL Cholesterol| ADL| Forgetfulness
|    |   Sleep Quality| Hypertension| Triglycerides Cholesterol|    | |

### Instruction
The app interface is easy to use. The user simply fills out each respective field and the app will dynamically update the probabilities. By default, the data is set to a value that is considered on average to be a "healthy" level for a given parameter.

## Limitations and Improvements
As mentioned previously the dataset used to train the model is synthetic so there are limitations to improve upon. One main example of this is the age range for the data is limited to 60-90 years old so it would be important to include additional data which broadens the range. Since the purpose of this app would be to predict the likelihood of onset of Alzheimer's, many users would likely want to know their probabilities at an age younger than 60. This improvement applies to several other categories within the dataset which do not have the full range of potential values represented. Additional data will help further train the neural network to provide accurate predictions.
## Links and References
Alzheimer’s disease - Diagnosis and treatment - Mayo Clinic. (n.d.). https://www.mayoclinic.org/diseases-conditions/alzheimers-disease/diagnosis-treatment/drc-20350453

Alzheimer’s disease - Symptoms and causes. (n.d.). Mayo Clinic. https://www.mayoclinic.org/diseases-conditions/alzheimers-disease/symptoms-causes/syc-20350447

Rabie El Kharoua. (2024). 🧠 Alzheimer's Disease Dataset 🧠 [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/8668279
