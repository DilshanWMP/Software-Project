using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using EndoscopyApp.Models;

namespace EndoscopyApp.ViewModels
{
    public partial class MediaViewerViewModel : ViewModelBase
    {
        private readonly MainViewModel _mainViewModel;
        private readonly Patient _patient;

        [ObservableProperty]
        private MediaFileViewModel _media;

        public MediaViewerViewModel(MainViewModel mainViewModel, MediaFileViewModel media, Patient patient)
        {
            _mainViewModel = mainViewModel;
            _media = media;
            _patient = patient;
        }

        [RelayCommand]
        private void NavigateBack()
        {
            _mainViewModel.NavigateToPatientMedia(_patient);
        }
    }
}
