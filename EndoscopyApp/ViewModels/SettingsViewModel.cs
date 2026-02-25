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
        private string _adminPassword = "";

        [ObservableProperty]
        private string _mediaPath = "";

        public SettingsViewModel(MainViewModel mainViewModel)
        {
            _mainViewModel = mainViewModel;
            _settingsService = new SettingsService();
            
            // Load current settings
            _currentSettings = _settingsService.LoadSettings();
            AdminPassword = _currentSettings.AdminPassword;
            MediaPath = _currentSettings.MediaPath;
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
        private void SaveSettings()
        {
            if (string.IsNullOrWhiteSpace(AdminPassword))
            {
                MessageBox.Show("Password cannot be empty.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }
            if (string.IsNullOrWhiteSpace(MediaPath))
            {
                MessageBox.Show("Media path cannot be empty.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            _currentSettings.AdminPassword = AdminPassword;
            _currentSettings.MediaPath = MediaPath;

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
