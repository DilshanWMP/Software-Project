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
    
    // Combine both lists from both branches
    if (viewModel is LoginViewModel || viewModel is HomeViewModel || 
        viewModel is PatientsViewModel || viewModel is LiveViewModel || 
        viewModel is RecordViewModel || viewModel is SelectPatientViewModel ||
        viewModel is RecordedVideosViewModel || viewModel is PatientMediaViewModel)
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
        public void Logout()
        {
            if (CurrentViewModel is LiveViewModel liveVm) liveVm.Cleanup();
            NavigateTo(new LoginViewModel(this));
        }
    }
}
