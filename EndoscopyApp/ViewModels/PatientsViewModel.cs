using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using System.Collections.ObjectModel;
using EndoscopyApp.Models;
using EndoscopyApp.Services;

namespace EndoscopyApp.ViewModels
{
    public partial class PatientsViewModel : ViewModelBase
    {
        private readonly MainViewModel _mainViewModel;
        private readonly DatabaseService _dbService;

        [ObservableProperty]
        private ObservableCollection<Patient> _patients;

        public PatientsViewModel(MainViewModel mainViewModel)
        {
            _mainViewModel = mainViewModel;
            _dbService = new DatabaseService();
            _patients = new ObservableCollection<Patient>(_dbService.GetAllPatients());
        }

        [RelayCommand]
        private void NavigateBack()
        {
            _mainViewModel.NavigateToHome();
        }

        [RelayCommand]
        private void AddPatient()
        {
            _mainViewModel.NavigateToRecord();
        }

        [RelayCommand]
        private void EditPatient(Patient patient)
        {
            // Edit logic placeholder
        }

        [RelayCommand]
        private void DeletePatient(Patient patient)
        {
            // Delete logic placeholder
        }
    }
}
