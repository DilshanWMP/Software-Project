using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using EndoscopyApp.Models;
using EndoscopyApp.Services;
using System.Windows;
using Microsoft.Win32;
using System.Collections.ObjectModel;
using System.Linq;

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

        // ---------------- Hardware push button (Arduino) ----------------

        public record PortOption(string Value, string Display);

        public ObservableCollection<PortOption> ButtonPorts { get; } = new();

        [ObservableProperty]
        private string _selectedButtonPort = ButtonService.AutoDetectSetting;

        [ObservableProperty]
        private string _buttonStatus = "";

        [ObservableProperty]
        private string _buttonLastMessage = "";

        [ObservableProperty]
        private bool _isButtonConnected;

        public SettingsViewModel(MainViewModel mainViewModel)
        {
            _mainViewModel = mainViewModel;
            _settingsService = new SettingsService();

            // Load current settings
            _currentSettings = _settingsService.LoadSettings();
            MediaPath = _currentSettings.MediaPath;
            CameraIndex = _currentSettings.CameraIndex;

            RefreshPorts();
            SelectedButtonPort = _currentSettings.ButtonPort ?? ButtonService.DisabledSetting;
            EnsurePortInList(SelectedButtonPort);

            ButtonService.Instance.StatusChanged += OnButtonStatusChanged;
            OnButtonStatusChanged();
        }

        private void OnButtonStatusChanged()
        {
            var service = ButtonService.Instance;
            ButtonStatus = service.StatusMessage;
            IsButtonConnected = service.State == ButtonConnectionState.Connected;
            ButtonLastMessage = string.IsNullOrEmpty(service.LastMessage)
                ? "No button press received yet"
                : $"Last message from device: {service.LastMessage}";
        }

        private void EnsurePortInList(string value)
        {
            if (!ButtonPorts.Any(p => p.Value == value))
                ButtonPorts.Add(new PortOption(value, $"{value} (not connected)"));
        }

        [RelayCommand]
        private void RefreshPorts()
        {
            string selected = SelectedButtonPort;
            ButtonPorts.Clear();
            ButtonPorts.Add(new PortOption(ButtonService.DisabledSetting, "Disabled (no hardware button)"));
            ButtonPorts.Add(new PortOption(ButtonService.AutoDetectSetting, "Auto-detect Arduino button"));
            foreach (var port in ButtonService.GetAvailablePorts())
                ButtonPorts.Add(new PortOption(port,
                    ButtonService.IsBluetoothPort(port) ? $"{port} (Bluetooth - not the Arduino)" : port));

            if (selected != null)
            {
                EnsurePortInList(selected);
                SelectedButtonPort = selected;
            }
        }

        /// <summary>Connect to the selected port right away so the button can be tested.</summary>
        [RelayCommand]
        private void ConnectButton()
        {
            ButtonService.Instance.Configure(SelectedButtonPort);
        }

        /// <summary>Called when leaving the screen.</summary>
        public void Cleanup()
        {
            ButtonService.Instance.StatusChanged -= OnButtonStatusChanged;
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
            _currentSettings.ButtonPort = SelectedButtonPort ?? ButtonService.DisabledSetting;

            try
            {
                _settingsService.SaveSettings(_currentSettings);
                ButtonService.Instance.Configure(_currentSettings.ButtonPort);
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
            // Discard a port that was only tested with "Connect" but not saved
            var saved = _settingsService.LoadSettings().ButtonPort;
            if (saved != ButtonService.Instance.CurrentSetting)
                ButtonService.Instance.Configure(saved);

            _mainViewModel.NavigateToHome();
        }
    }
}
