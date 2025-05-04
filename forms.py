from django import forms
from .models import Expense

class ExpenseForm(forms.ModelForm):
    class Meta:
        model = Expense
        fields = ['date', 'description', 'amount', 'category']  # Include category
        widgets = {
            'date': forms.DateInput(attrs={'type': 'date'}),
        }