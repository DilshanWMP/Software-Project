using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using EndoscopyApp.Models;
using EndoscopyApp.Services;
using System.Windows;
using Microsoft.Win32;

namespace EndoscopyApp.ViewModels
{
    public partial class SettingsViewModel : ViewModelBase
    {
        private readonly MainViewModel _mainViewModel;
        private readonly SettingsService _settingsService;
        private AppSettings _currentSettings;

        [ObservableProperty]
        private string _currentPassword = "";

        [ObservableProperty]
        private string _newPassword = "";

        [ObservableProperty]
        private string _confirmPassword = "";

        [ObservableProperty]
        private bool _isCurrentPasswordVisible;

        [ObservableProperty]
        private bool _isNewPasswordVisible;

        [ObservableProperty]
        private bool _isConfirmPasswordVisible;

        [ObservableProperty]
        private string _mediaPath = "";

        [ObservableProperty]
        private int _cameraIndex;

        public SettingsViewModel(MainViewModel mainViewModel)
        {
            _mainViewModel = mainViewModel;
            _settingsService = new SettingsService();

            // Load current settings
            _currentSettings = _settingsService.LoadSettings();
            MediaPath = _currentSettings.MediaPath;
            CameraIndex = _currentSettings.CameraIndex;
        }

        [RelayCommand]
        private void BrowseFolder()
        {
            var folderDialog = new OpenFolderDialog
            {
                Title = "Select Media Save Folder",
                InitialDirectory = MediaPath
            };

            if (folderDialog.ShowDialog() == true)
            {
                MediaPath = folderDialog.FolderName;
            }
        }

        [RelayCommand]
        private void ToggleCurrentPasswordVisibility()
        {
            IsCurrentPasswordVisible = !IsCurrentPasswordVisible;
        }

        [RelayCommand]
        private void ToggleNewPasswordVisibility()
        {
            IsNewPasswordVisible = !IsNewPasswordVisible;
        }

        [RelayCommand]
        private void ToggleConfirmPasswordVisibility()
        {
            IsConfirmPasswordVisible = !IsConfirmPasswordVisible;
        }

        [RelayCommand]
        private void SaveSettings()
        {
            if (string.IsNullOrWhiteSpace(MediaPath))
            {
                MessageBox.Show("Media path cannot be empty.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            bool isPasswordChangeAttempted = !string.IsNullOrWhiteSpace(CurrentPassword) ||
                                             !string.IsNullOrWhiteSpace(NewPassword) ||
                                             !string.IsNullOrWhiteSpace(ConfirmPassword);

            if (isPasswordChangeAttempted)
            {
                if (CurrentPassword != _currentSettings.AdminPassword)
                {
                    MessageBox.Show("Current password is incorrect.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                    return;
                }

                if (string.IsNullOrWhiteSpace(NewPassword) || NewPassword != ConfirmPassword)
                {
                    MessageBox.Show("New passwords do not match or are empty.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                    return;
                }

                _currentSettings.AdminPassword = NewPassword;
            }

            _currentSettings.MediaPath = MediaPath;
            _currentSettings.CameraIndex = CameraIndex;

            try
            {
                _settingsService.SaveSettings(_currentSettings);
                MessageBox.Show("Settings saved successfully.", "Success", MessageBoxButton.OK, MessageBoxImage.Information);
                _mainViewModel.NavigateToHome();
            }
            catch (System.Exception ex)
            {
                MessageBox.Show($"Failed to save settings: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        [RelayCommand]
        private void GoBack()
        {
            _mainViewModel.NavigateToHome();
        }
    }
}
