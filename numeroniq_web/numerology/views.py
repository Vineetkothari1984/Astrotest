from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login as auth_login
from django.shortcuts import render, redirect
from django.views.decorators.cache import cache_page

from .core import calculate_life_path


def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            auth_login(request, user)
            return redirect('home')
    else:
        form = AuthenticationForm()
    return render(request, 'numerology/login.html', {'form': form})


@login_required
def home(request):
    return render(request, 'numerology/home.html')


@cache_page(60 * 15)
@login_required
def life_path(request):
    result = None
    if request.method == 'POST':
        number = int(request.POST.get('number'))
        result = calculate_life_path(number)
    return render(request, 'numerology/life_path.html', {'result': result})
