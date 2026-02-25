using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using System.Windows;

namespace EndoscopyApp.ViewModels
{
    public partial class LoginViewModel : ViewModelBase
    {
        private readonly MainViewModel _mainViewModel;

        [ObservableProperty]
        private string _password = "";

        [ObservableProperty]
        private string _errorMessage = "";

        [ObservableProperty]
        private bool _isPasswordVisible;

        public LoginViewModel(MainViewModel mainViewModel)
        {
            _mainViewModel = mainViewModel;
        }

        [RelayCommand]
        private void TogglePasswordVisibility()
        {
            IsPasswordVisible = !IsPasswordVisible;
        }

        [RelayCommand]
        private void Login()
        {
            // Simple hardcoded password for Phase 1 as requested
            if (Password == "admin123")
            {
                ErrorMessage = "";
                _mainViewModel.NavigateTo(new HomeViewModel(_mainViewModel));
            }
            else
            {
                ErrorMessage = "Invalid Password";
            }
        }
    }
}
