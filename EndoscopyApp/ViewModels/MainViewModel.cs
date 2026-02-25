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

        public void NavigateTo(ViewModelBase viewModel)
        {
             CurrentViewModel = viewModel;
             // Show sidebar if we are not on Login page
             if (viewModel is LoginViewModel)
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
        public void Logout()
        {
             if (CurrentViewModel is LiveViewModel liveVm) liveVm.Cleanup();
             NavigateTo(new LoginViewModel(this));
        }
    }
}
