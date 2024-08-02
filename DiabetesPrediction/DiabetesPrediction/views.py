from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the dataset and train the model when the module is first imported
diabetes_df = pd.read_csv(r'C:\Users\ameer\diabetes prediction system\diabetes.csv')

# Drop the 'Outcome' column from the dataframe
X = diabetes_df.drop(['Outcome'], axis=1)
y = diabetes_df['Outcome']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)

# Initialize SVC model and train it
svc_model = SVC()
svc_model.fit(X_train, y_train)

def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    if request.method == 'GET':
        try:
            val1 = float(request.GET.get('n1'))
            val2 = float(request.GET.get('n2'))
            val3 = float(request.GET.get('n3'))
            val4 = float(request.GET.get('n4'))
            val5 = float(request.GET.get('n5'))
            val6 = float(request.GET.get('n6'))
            val7 = float(request.GET.get('n7'))
        except (TypeError, ValueError):
            error_message = "Please enter valid values for all fields."
            return render(request, 'predict.html', {"error": error_message})

        # Create a list containing the feature values and reshape it to be 2D
        features = [[val1, val2, val3, val4, val5, val6, val7]]
        pred = svc_model.predict(features)

        result1 = "Positive" if pred == 1 else "Negative"

        if result1 == "Positive":
            return render(request, 'positive.html')
        else:
            return render(request, 'negative.html')

    # Handle cases where the method is not GET (if needed)
    return render(request, 'predict.html')

def about(request):
    return render(request, 'about.html')
