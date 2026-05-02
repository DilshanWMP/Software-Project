using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using EndoscopyApp.Models;
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

        [RelayCommand]
        public void NavigateToRecordedVideos()
        {
            if (CurrentViewModel is not RecordedVideosViewModel)
            {
                if (CurrentViewModel is LiveViewModel liveVm) liveVm.Cleanup();
                NavigateTo(new RecordedVideosViewModel(this));
                PageTitle = "Recorded Videos";
            }
        }

        public void NavigateToPatientMedia(Patient patient)
        {
            if (CurrentViewModel is LiveViewModel liveVm) liveVm.Cleanup();
            NavigateTo(new PatientMediaViewModel(this, patient));
            PageTitle = "Recorded Media";
        }

        public void NavigateToMediaViewer(MediaFileViewModel media, Patient patient)
        {
            if (CurrentViewModel is LiveViewModel liveVm) liveVm.Cleanup();
            NavigateTo(new MediaViewerViewModel(this, media, patient));
            PageTitle = "Media Viewer";
        }

        public void NavigateTo(ViewModelBase viewModel)
        {
            CurrentViewModel = viewModel;

            // Hide sidebar for most views to maximize workspace
            if (viewModel is LoginViewModel || viewModel is HomeViewModel ||
                viewModel is PatientsViewModel || viewModel is LiveViewModel ||
                viewModel is RecordViewModel || viewModel is SelectPatientViewModel ||
                viewModel is RecordedVideosViewModel || viewModel is PatientMediaViewModel ||
                viewModel is SettingsViewModel || viewModel is MediaViewerViewModel)
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
            if (CurrentViewModel is not SelectPatientViewModel)
            {
                // Cleanup previous view if needed
                if (CurrentViewModel is LiveViewModel liveVm) liveVm.Cleanup();
                NavigateTo(new SelectPatientViewModel(this));
                PageTitle = "Live Video";
            }
        }

        [RelayCommand]
        public void NavigateToRecord()
        {
            if (CurrentViewModel is SelectPatientViewModel) return;

            // Cleanup previous
            if (CurrentViewModel is LiveViewModel liveVm) liveVm.Cleanup();

            NavigateTo(new SelectPatientViewModel(this));
            PageTitle = "Live Video";
        }

        [RelayCommand]
        public void NavigateToGallery()
        {
            if (CurrentViewModel is not RecordViewModel)
            {
                if (CurrentViewModel is LiveViewModel liveVm) liveVm.Cleanup();
                NavigateTo(new RecordViewModel(this));
                PageTitle = "Live Recording";
            }
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
