using CommunityToolkit.Mvvm.Input;

namespace EndoscopyApp.ViewModels
{
    public partial class HomeViewModel : ViewModelBase
    {
        private readonly MainViewModel _mainViewModel;

        public HomeViewModel(MainViewModel mainViewModel)
        {
            _mainViewModel = mainViewModel;
        }

        [RelayCommand]
        private void NavigateToPatientDetails()
        {
            _mainViewModel.NavigateToPatients();
        }

        [RelayCommand]
        private void NavigateToLive()
        {
            _mainViewModel.NavigateToLive();
        }

        [RelayCommand]
        private void NavigateToRecordedVideos()
        {
            _mainViewModel.NavigateToGallery();
        }

        [RelayCommand]
        private void Settings()
        {
            _mainViewModel.NavigateToSettings();
        }

        [RelayCommand]
        private void Logout()
        {
            _mainViewModel.Logout();
        }
    }
}
