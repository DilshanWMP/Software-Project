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

        [ObservableProperty]
        private bool _isAddModalOpen;

        [ObservableProperty]
        private string _newPatientId = "";
        [ObservableProperty]
        private string _newPatientName = "";
        [ObservableProperty]
        private string _newPatientAge = "";
        [ObservableProperty]
        private string _newPatientGender = "Male";
        [ObservableProperty]
        private string _newPatientPhone = "";
        [ObservableProperty]
        private string _newPatientNotes = "";

        public ObservableCollection<string> GenderOptions { get; } = new() { "Male", "Female", "Other" };

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
        private void OpenAddModal()
        {
            IsAddModalOpen = true;
            // Clear fields when opening
            NewPatientId = "";
            NewPatientName = "";
            NewPatientAge = "";
            NewPatientGender = "Male";
            NewPatientPhone = "";
            NewPatientNotes = "";
        }

        [RelayCommand]
        private void CloseAddModal()
        {
            IsAddModalOpen = false;
        }

        [RelayCommand]
        private void AddPatient()
        {
            if (string.IsNullOrWhiteSpace(NewPatientName)) return;

            int.TryParse(NewPatientAge, out int age);

            var patient = new Patient
            {
                Name = NewPatientName,
                Age = age,
                Gender = NewPatientGender,
                Phone = NewPatientPhone,
                Notes = NewPatientNotes,
                CreatedAt = System.DateTime.Now
            };

            _dbService.AddPatient(patient);
            Patients.Add(patient);
            IsAddModalOpen = false;
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
