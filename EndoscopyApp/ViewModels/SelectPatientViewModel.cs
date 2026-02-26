using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using System.Collections.ObjectModel;
using EndoscopyApp.Models;
using EndoscopyApp.Services;

namespace EndoscopyApp.ViewModels
{
    public partial class SelectPatientViewModel : ViewModelBase
    {
        private readonly MainViewModel _mainViewModel;
        private readonly DatabaseService _dbService;

        [ObservableProperty]
        private ObservableCollection<Patient> _patients;

        public SelectPatientViewModel(MainViewModel mainViewModel)
        {
            _mainViewModel = mainViewModel;
            _dbService = new DatabaseService();
            var patientsList = _dbService.GetAllPatients();
            _patients = new ObservableCollection<Patient>(patientsList);
        }

        [RelayCommand]
        private void SelectPatient(Patient patient)
        {
            var recordVm = new RecordViewModel(_mainViewModel);
            recordVm.SetPatient(patient);
            _mainViewModel.NavigateTo(recordVm);
        }

        [RelayCommand]
        private void NavigateBack()
        {
            _mainViewModel.NavigateToHome();
        }
    }
}
