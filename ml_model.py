import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from .models import Expense

# Global models
mnb_model = None
svm_model = None
gnb_model = None
vectorizer = None

def get_data():
    expenses = Expense.objects.exclude(category='Others').values('description', 'category')
    return pd.DataFrame(expenses)

def train_models():
    global mnb_model, svm_model, gnb_model, vectorizer

    df = get_data()

    # Seed data to help early training
    sample_data = pd.DataFrame({
        'description': [
            'Domino Pizza', 'Uber Ride', 'Netflix Subscription', 'Bus Ticket',
            'Electricity Bill', 'Hospital Visit', 'Course Fee', 'Amazon Order',
            'Movie Night', 'Grocery Shopping', 'Gym Membership', 'Recharge',
            'Flight Ticket', 'Laptop Purchase', 'Water Bill', 'Medicine Purchase',
            'Mobile Accessories', 'Tuition Fee', 'Doctor Appointment', 'Snacks'
        ],
        'category': [
            'Food', 'Transport', 'Entertainment', 'Transport',
            'Utilities', 'Health', 'Education', 'Shopping',
            'Entertainment', 'Food', 'Health', 'Utilities',
            'Transport', 'Shopping', 'Utilities', 'Health',
            'Shopping', 'Education', 'Health', 'Food'
        ]
    })

    # Merge with real user data
    df = pd.concat([df, sample_data], ignore_index=True)

    # Make sure we have at least 2 classes to train
    if df['category'].nunique() < 2:
        print("[ML] Not enough unique categories to train.")
        return

    # Vectorize text
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['description'])
    y = df['category']

    # Train classifiers
    mnb_model = MultinomialNB()
    mnb_model.fit(X, y)

    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X, y)

    gnb_model = GaussianNB()
    gnb_model.fit(X.toarray(), y)

    print("[ML] Models trained successfully.")

def classify_description(description, model='mnb'):
    global vectorizer, mnb_model, svm_model, gnb_model

    if not vectorizer:
        print("[ML] No trained model. Training now...")
        train_models()

    if not vectorizer:
        return "Others"  # Fallback in case training failed

    X_desc = vectorizer.transform([description])

    if model == 'svm':
        return svm_model.predict(X_desc)[0]
    elif model == 'gnb':
        return gnb_model.predict(X_desc.toarray())[0]
    else:  # default to mnb
        return mnb_model.predict(X_desc)[0]

def predict_monthly_spending():
    import datetime
    from sklearn.linear_model import LinearRegression

    expenses = Expense.objects.all().order_by('date')
    if not expenses:
        return 0

    df = pd.DataFrame(list(expenses.values('date', 'amount')))
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M').astype(str)

    monthly_totals = df.groupby('month')['amount'].sum().reset_index()
    monthly_totals['month_num'] = range(1, len(monthly_totals) + 1)
    
    if len(monthly_totals) < 2:
        return monthly_totals['amount'].iloc[-1] if not monthly_totals.empty else 0

    model = LinearRegression()
    model.fit(monthly_totals[['month_num']], monthly_totals['amount'])

    next_month_num = monthly_totals['month_num'].max() + 1
    predicted = model.predict([[next_month_num]])[0]
    return round(predicted, 2)
