import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
import numpy as np
#loading saved models
diabetes_model=pickle.load(open('diabetes.sav','rb'))

heart_model=pickle.load(open('heart.sav','rb'))

parkinsons_model=pickle.load(open('parkinsons.sav','rb'))

scaler=StandardScaler()

#sidebar for navigation

with st.sidebar:
  selected=option_menu("Multiple Disease Prediction System",
                       
                       ['Diabetes Prediction',
                        'Heart Disease Prediction',
                        'Parkinsons prediction'],
                       #bootstrap icons
                       icons=['activity','heart','person-wheelchair'],
                       
                       default_index=0
                       )

if (selected=='Diabetes Prediction'):
  st.title("Diabetes Prediction System")
  
  col1,col2,col3=st.columns(3)
  
  with col1:
    Pregnancies=st.text_input('Number of Pregnancies')
  with col2:
    Glucose=st.text_input('Glucose Level')
  with col3:
    BloodPressure=st.text_input('BP')
  
  with col1:
    SkinThickness=st.text_input('Skin Thickness value')
  with col2:
    Insulin=st.text_input('Insulin Level')
  with col3:
    BMIvalue=st.text_input('BMI')
  
  with col1:
    Diabetespedegreefn=st.text_input('Diabetes Pedegree Function')
  with col2:
    Age=st.text_input('Age')
  
  diab_diagnose=''
  
  
  if st.button('result'):
    diab_prediction=diabetes_model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMIvalue,Diabetespedegreefn,Age]])
    
    if diab_prediction[0]==1:
      diab_diagnose="You are Diabetic!!!"
    else:
      diab_diagnose="You are not diabetic"
  
  st.success(diab_diagnose)
 
  
  
  
  
  
  
if (selected=='Heart Disease Prediction'):
  st.title("Heart Disease Prediction System")

  col1,col2=st.columns(2)
  
  with col1:
    age=st.text_input('Your Age')
    
  with col2:
    sex=st.text_input('Gender (Male-1 Female-0)')
  
  cp=st.text_input('Chest Pain Type(4 values)')
  
  trestbps=st.text_input('resting blood pressure')
    
  chol=st.text_input('serum cholestoral in mg/dl')
    
  fbs=st.text_input('fasting blood sugar > 120 mg/d (Yes-1 No-0)')
    
  restecg=st.text_input('resting electrocardiographic results (values 0,1,2)')
    
  thalach=st.text_input('maximum heart rate achieved')
  
  exang=st.text_input('exercise induced angina(Yes-1 No-0)')
  
  oldpeak=st.text_input(' ST depression induced by exercise relative to rest')
  
  slope=st.text_input('the slope of the peak exercise ST segment')
  
  ca=st.text_input('number of major vessels (0-3) colored by flourosopy')
  
  thal=st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
  
  
  
  heart_diagnose=''
  
  
  if st.button('result'):
    inputs=[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
    num_in=[float(x) for x in inputs]
    heart_prediction=heart_model.predict([num_in])
    
    if heart_prediction[0]==1:
      heart_diagnose="You have a unhealthy heart!!!"
    else:
      heart_diagnose="You have a healthy heart"
  
  st.success(heart_diagnose)  
  
  
  
  
  

if (selected=='Parkinsons prediction'):
  st.title("Parkinsons prediction System")
  
  col1,col2,col3,col4=st.columns(4)
  
  with col1:
    fho=st.text_input('MDVP Fo(Hz)')
  with col2:
    fhi=st.text_input('MDVP Fhi(Hz)')
  with col3:
    flo=st.text_input('MDVP Flo(Hz)')
  with col4:
    jitter=st.text_input('MDVP Jitter(%)')
  
  with col1:
    jittera=st.text_input('MDVP Jitter(Abs)')
  with col2:
    rap=st.text_input('MDVP RAP')
  with col3:
    ppq=st.text_input('MDVP PPQ')
  with col4:
    jitterddp=st.text_input('Jitter DDP')
    
  with col1:
    shimmer=st.text_input('MDVP Shimmer')
  with col2:
    shimmerdb=st.text_input('MDVP Shimmer(dB)')
  with col3:
    shimmerap3=st.text_input('Shimmer APQ3')
  with col4:
    shimmerap5=st.text_input('Shimmer APQ5')
  
  with col1:
    mdvpaqp=st.text_input('MDVP APQ')
  with col2:
    shimmerdda=st.text_input('Shimmer DDA')
  with col3:
    nhr=st.text_input('NHR')
  with col4:
    hnr=st.text_input('HNR')
  
  with col1:
    rpde=st.text_input('RPDE')
  with col2:
    dfa=st.text_input('DFA')
  with col3:
    spread1=st.text_input('Spread1')
  with col4:
    spread2=st.text_input('Spread2')
    
  with col1:
    d2=st.text_input('D2')
  with col2:
    ppe=st.text_input('PPE')
  
  
  parkinsons_diagnose=''

  
  if st.button('result'):
    inputs=[fho,fhi,flo,jitter,jittera,rap,ppq,jitterddp,shimmer,shimmerdb,shimmerap3,shimmerap5,mdvpaqp,shimmerdda,nhr,hnr,rpde,dfa,spread1,spread2,d2,ppe]
    inputs_f=[float(x) for x in inputs]
    input_np=np.asarray(inputs_f)
    input_reshaped=input_np.reshape(1,-1)
    std_in=scaler.fit_transform(input_reshaped)
    
    
    parkinsons_prediction=parkinsons_model.predict(std_in)
    
    if parkinsons_prediction[0]==1:
      parkinsons_diagnose="You have parkinsons disease!!!"
    else:
      parkinsons_diagnose="You do not have parkinsons"
  
  st.success(parkinsons_diagnose)
  