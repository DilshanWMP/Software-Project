using System.Windows.Controls;
using EndoscopyApp.ViewModels;

namespace EndoscopyApp.Views
{
    public partial class SettingsView : UserControl
    {
        public SettingsView()
        {
            InitializeComponent();
            this.DataContextChanged += SettingsView_DataContextChanged;
        }

        private void SettingsView_DataContextChanged(object sender, System.Windows.DependencyPropertyChangedEventArgs e)
        {
            if (e.OldValue is SettingsViewModel oldVm)
                oldVm.PropertyChanged -= ViewModel_PropertyChanged;
            if (e.NewValue is SettingsViewModel newVm)
                newVm.PropertyChanged += ViewModel_PropertyChanged;
        }

        private void ViewModel_PropertyChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
        {
            if (DataContext is SettingsViewModel vm)
            {
                if (e.PropertyName == nameof(SettingsViewModel.IsCurrentPasswordVisible) && !vm.IsCurrentPasswordVisible)
                    CurrentPasswordBox.Password = vm.CurrentPassword;
                    
                if (e.PropertyName == nameof(SettingsViewModel.IsNewPasswordVisible) && !vm.IsNewPasswordVisible)
                    NewPasswordBox.Password = vm.NewPassword;
                    
                if (e.PropertyName == nameof(SettingsViewModel.IsConfirmPasswordVisible) && !vm.IsConfirmPasswordVisible)
                    ConfirmPasswordBox.Password = vm.ConfirmPassword;
            }
        }

        private void CurrentPasswordBox_PasswordChanged(object sender, System.Windows.RoutedEventArgs e)
        {
            if (this.DataContext is SettingsViewModel viewModel)
                viewModel.CurrentPassword = ((PasswordBox)sender).Password;
        }

        private void NewPasswordBox_PasswordChanged(object sender, System.Windows.RoutedEventArgs e)
        {
            if (this.DataContext is SettingsViewModel viewModel)
                viewModel.NewPassword = ((PasswordBox)sender).Password;
        }

        private void ConfirmPasswordBox_PasswordChanged(object sender, System.Windows.RoutedEventArgs e)
        {
            if (this.DataContext is SettingsViewModel viewModel)
                viewModel.ConfirmPassword = ((PasswordBox)sender).Password;
        }
    }
}
