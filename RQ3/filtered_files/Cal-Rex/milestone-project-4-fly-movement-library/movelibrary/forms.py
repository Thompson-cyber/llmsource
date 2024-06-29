from .models import UserOneRepMax, User, UserNonAuthField
from django import forms
from allauth.account.forms import SignupForm


class OneRmForm(forms.ModelForm):

    class Meta:
        model = UserOneRepMax
        fields = ('one_rep_max',)


class CustomSignupForm(SignupForm):  # Form generated by ChatGPT
    first_name = forms.CharField(max_length=50, label='First Name')
    last_name = forms.CharField(max_length=50, label='Last Name')

    def save(self, request):
        user = super(CustomSignupForm, self).save(request)
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        user.save()
        return user


class NameEditForm(forms.ModelForm):

    class Meta:
        model = User
        fields = ('first_name', 'last_name',)
