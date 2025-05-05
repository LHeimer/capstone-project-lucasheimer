# Capstone Project: Alzheimer's Diagnosis Prediction App
## Rationale
## Description
### Model
This app was created to predict the probability of a Alzheimer's Diagnosis using standard biometric and demographic data. The data used for this model was synthetically developed, however many of the assessments and parameters used are common standards used to gauge the onset of Alzheimer's. Prior to inserting it into the model, the data was cleaned by removing irrelevant information and then it was normalized. A logistic regression model was defined by passing the inputs through a linear neural network layer followed by applying a sigmoid function to classify the outcome as either a positive or negative diagnosis. In addition to modeling an overall probability of diagnosis using all the metrics as inputs, the data was divided into smaller groups to examine the how each subcategory influenced the overall outcome. A new dataframe was created for each subgroup and the same logistic regression model was used to create a diagnosis prediction. The app also displays a bar chart outlining the individual probabilities for each variable within the subcategory. This allows the user to alter the variables and observe in realtime how an increase or decrease in certain factors may affect the likelihood of a positive diagnosis. The breakout of subcategories is outlined in the following table:
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
## Links and References
