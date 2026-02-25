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

            // Add dummy data for demonstration if list is short
            if (_patients.Count < 5)
            {
                _patients.Add(new Patient { Id = 230984, Name = "Praveen Dilshan", Age = 21, Gender = "Male", Phone = "0239490283" });
                _patients.Add(new Patient { Id = 230985, Name = "Kasun Perera", Age = 25, Gender = "Male", Phone = "0771234567" });
                _patients.Add(new Patient { Id = 230986, Name = "Nimali Silva", Age = 30, Gender = "Female", Phone = "0719876543" });
                _patients.Add(new Patient { Id = 230987, Name = "Sunil Gamage", Age = 45, Gender = "Male", Phone = "0112233445" });
                _patients.Add(new Patient { Id = 230988, Name = "Gayani Fernando", Age = 28, Gender = "Female", Phone = "0755566778" });
                _patients.Add(new Patient { Id = 230989, Name = "Dilan Gunawardena", Age = 35, Gender = "Male", Phone = "0723344556" });
                _patients.Add(new Patient { Id = 230990, Name = "Ishara Madushanka", Age = 22, Gender = "Male", Phone = "0766677889" });
                _patients.Add(new Patient { Id = 230991, Name = "Samanthi Cooray", Age = 27, Gender = "Female", Phone = "0788899001" });
                _patients.Add(new Patient { Id = 230992, Name = "Ruwan Kumara", Age = 40, Gender = "Male", Phone = "0700011223" });
            }
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
