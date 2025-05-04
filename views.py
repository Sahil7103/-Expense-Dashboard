from django.shortcuts import render, redirect
from .forms import ExpenseForm
from .models import Expense
from .ml_model import classify_description, predict_monthly_spending, train_models
from django.utils.timezone import now

def add_expense(request):
    if request.method == 'POST':
        form = ExpenseForm(request.POST)
        if form.is_valid():
            exp = form.save(commit=False)

            # Only auto-classify if user selects 'Others'
            if exp.category == 'Others':
                exp.category = classify_description(exp.description)

            exp.save()
            train_models()
            return redirect('dashboard')
    else:
        form = ExpenseForm()
    
    return render(request, 'add_expense.html', {'form': form})

def dashboard(request):
    expenses = Expense.objects.all().order_by('-date')
    prediction = predict_monthly_spending()

    today = now().date()
    current_month_expenses = expenses.filter(date__month=today.month, date__year=today.year)
    current_month_total = sum(exp.amount for exp in current_month_expenses)

    return render(request, 'dashboard.html', {
        'expenses': expenses,
        'prediction': prediction,
        'current_month_total': current_month_total
    })

def delete_expense(request, id):
    Expense.objects.get(id=id).delete()
    return redirect('dashboard')

