from django.shortcuts import render
from .forms import UserInput
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import pickle
import pandas as pd

# Create your views here.
def home(request):
    if request.method == 'POST':
        form = UserInput(request.POST)
        if form.is_valid():
            #user_symptoms from form
            user_symptoms = form.cleaned_data['user_symptoms']
            while len(user_symptoms) < 17:
                user_symptoms.append('No Symptom')

            #Load encoder
            label_encoder = LabelEncoder()
            label_encoder.classes_ = np.load(os.path.abspath('models/encoder_symptoms.npy'))

            #Encode user symptoms into readeble for model format
            user_symptoms = label_encoder.transform(np.array(user_symptoms)).reshape(-1, 17)

            #Load model
            disease_forest = pickle.load(open(os.path.abspath('models/disease_forest.sav'), 'rb'))

            #Predicting top 5 possible diagnoses
            diagnose = disease_forest.predict_proba(user_symptoms)
            index_of_maximum_proba = diagnose.argsort()[0][-5:][::-1]
            predicted_classes = disease_forest.classes_[index_of_maximum_proba]

            #Load disease precautions and description
            disease_precaution_description = pd.read_csv(os.path.abspath('data/precaution_description.csv')).set_index('Disease').loc[predicted_classes].drop('Unnamed: 0', axis=1)

            #Save disease probability to dataframe
            disease_precaution_description['Probability'] = pd.Series(data=diagnose[0][index_of_maximum_proba],
                                                                 index=predicted_classes).apply(lambda probability: str(round(probability * 100, 1)) + '%')
            disease_precaution_description['Precautions'] = disease_precaution_description[['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].fillna('').agg(', '.join, axis=1)
            disease_precaution_description.drop(['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4'], axis=1)
            disease_precaution_description['Description'] = disease_precaution_description['Description'].fillna('No Description for this Disease Found')
            #Conver information about disease to dictionary
            precaution_description_dict = disease_precaution_description.T.to_dict()
            return render(request, 'prediction.html', {'results': precaution_description_dict})
    else:
        form = UserInput
    return render(request, 'user_symptoms.html', {'form': form})