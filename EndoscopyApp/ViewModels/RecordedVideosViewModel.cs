using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using EndoscopyApp.Models;
using EndoscopyApp.Services;
using System.Collections.ObjectModel;

namespace EndoscopyApp.ViewModels
{
    public partial class RecordedVideosViewModel : ViewModelBase
    {
        private readonly MainViewModel _mainViewModel;
        private readonly DatabaseService _dbService;

        [ObservableProperty]
        private ObservableCollection<Patient> _patients = new();

        public RecordedVideosViewModel(MainViewModel mainViewModel)
        {
            _mainViewModel = mainViewModel;
            _dbService = new DatabaseService();
            LoadPatients();
        }

        private void LoadPatients()
        {
            var list = _dbService.GetAllPatients();
            Patients = new ObservableCollection<Patient>(list);
        }

        [RelayCommand]
        private void NavigateBack()
        {
            _mainViewModel.NavigateToHome();
        }

        [RelayCommand]
        private void SelectPatient(Patient patient)
        {
            _mainViewModel.NavigateToPatientMedia(patient);
        }
    }
}
