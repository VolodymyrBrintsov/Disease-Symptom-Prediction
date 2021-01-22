from django import forms
from multiselectfield import MultiSelectField
import numpy as np
import os
class UserInput(forms.Form):
    symptoms_np = np.load(os.path.abspath(".")+'/models/symptoms.npy').tolist()
    symptoms = tuple([(symptom, symptom.capitalize().replace('_', ' ')) for symptom in symptoms_np])
    user_symptoms = forms.MultipleChoiceField(widget=forms.CheckboxSelectMultiple, choices=symptoms, label='Select your symptoms:')

    def clean_user_symptoms(self):
        if len(self.cleaned_data['user_symptoms']) > 17:
            raise forms.ValidationError('Select no more than 17 symptoms.')
        return self.cleaned_data['user_symptoms']