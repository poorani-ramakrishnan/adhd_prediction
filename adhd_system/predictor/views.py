import joblib
import os
from django.shortcuts import render 

base = os.path.dirname(os.path.abspath(__file__))
scaler = joblib.load(os.path.join(base, 'scaler.pkl'))
label_encoders = joblib.load(os.path.join(base, 'labelEncoders.pkl'))
adhd_model = joblib.load(os.path.join(base, 'adhdModel.pkl'))

def predict(request):
    if(request.method == 'POST'):
        gender = label_encoders['Gender'].transform([request.POST.get('Gender')])[0]
        education = label_encoders['EducationStage'].transform([request.POST.get('EducationStage')])[0]
        medication = label_encoders['Medication'].transform([request.POST.get('Medication')])[0]
        school = label_encoders['SchoolSupport'].transform([request.POST.get('SchoolSupport')])[0]
        inattention = int(request.POST.get('InattentionScore', 0))
        impulsive = int(request.POST.get('ImpulsivityScore', 0))
        hyperactive = int(request.POST.get('HyperactivityScore', 0))
        symptomsum = inattention+hyperactive+impulsive

        features = [
            int(request.POST.get('age',0)),
            gender,education,inattention,impulsive,hyperactive,symptomsum,
            int(request.POST.get('DayDreaming', 0)),
            int(request.POST.get('RSD', 0)),
            float(request.POST.get('SleepHours', 0)),
            float(request.POST.get('ScreenTime', 0)),
            int(request.POST.get('ComorbidAnxiety', 0)),
            int(request.POST.get('ComorbidDepression', 0)),
            int(request.POST.get('FamilyHistoryADHD', 0)),
            medication,
            school,
            float(request.POST.get('AcademicScore', 0)),
            
        ]
        X = scaler.transform([features])
        prediction = adhd_model.predict(X)[0]

        result = "There are possible signs of ADHD from your answers." if prediction == 1 else "You might not have ADHD"
        return render(request, "result.html", {"result":result})
    return render(request, "index.html")

def game(request):
    return render(request, "game.html")
