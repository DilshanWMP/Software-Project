using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using System.Windows;
using EndoscopyApp.Services;

namespace EndoscopyApp.ViewModels
{
    public partial class MainViewModel : ViewModelBase
    {
        [ObservableProperty]
        private ViewModelBase? _currentViewModel;

        [ObservableProperty]
        private string _pageTitle = "Endoscopy Capture";

        [ObservableProperty]
        private Visibility _sidebarVisibility = Visibility.Collapsed;

        public MainViewModel()
        {
            // Initial View
            CurrentViewModel = new LoginViewModel(this);
            SidebarVisibility = Visibility.Collapsed;
        }

        [RelayCommand]
        public void NavigateToHome()
        {
            if (CurrentViewModel is not HomeViewModel)
            {
                if (CurrentViewModel is LiveViewModel liveVm) liveVm.Cleanup();
                NavigateTo(new HomeViewModel(this));
                PageTitle = "Home";
            }
        }

        [RelayCommand]
        public void NavigateToPatients()
        {
            if (CurrentViewModel is not PatientsViewModel)
            {
                if (CurrentViewModel is LiveViewModel liveVm) liveVm.Cleanup();
                NavigateTo(new PatientsViewModel(this));
                PageTitle = "Patient Details";
            }
        }

        public void NavigateTo(ViewModelBase viewModel)
        {
             CurrentViewModel = viewModel;
              // Show sidebar if we are not on Login, Home or Patients page
              if (viewModel is LoginViewModel || viewModel is HomeViewModel || viewModel is PatientsViewModel)
              {
                  SidebarVisibility = Visibility.Collapsed;
              }
              else
              {
                  SidebarVisibility = Visibility.Visible;
              }
        }

        [RelayCommand]
        public void NavigateToLive()
        {
            if (CurrentViewModel is not LiveViewModel)
            {
                // Cleanup previous view if needed
                if (CurrentViewModel is LiveViewModel liveVm) liveVm.Cleanup();
                NavigateTo(new LiveViewModel());
                PageTitle = "Live View";
            }
        }

        [RelayCommand]
        public void NavigateToRecord()
        {
            if (CurrentViewModel is RecordViewModel) return;
            
            // Cleanup previous
            if (CurrentViewModel is LiveViewModel liveVm) liveVm.Cleanup();

            NavigateTo(new RecordViewModel());
            PageTitle = "Registration & Record";
        }

        [RelayCommand]
        public void NavigateToGallery()
        {
            if (CurrentViewModel is GalleryViewModel) return;

             if (CurrentViewModel is LiveViewModel liveVm) liveVm.Cleanup();

            NavigateTo(new GalleryViewModel());
            PageTitle = "Use Gallery";
        }

        [RelayCommand]
        public void NavigateToSettings()
        {
            if (CurrentViewModel is SettingsViewModel) return;

             if (CurrentViewModel is LiveViewModel liveVm) liveVm.Cleanup();

            NavigateTo(new SettingsViewModel(this));
            PageTitle = "System Settings";
        }

        [RelayCommand]
        public void Logout()
        {
             if (CurrentViewModel is LiveViewModel liveVm) liveVm.Cleanup();
             NavigateTo(new LoginViewModel(this));
        }
    }
}
