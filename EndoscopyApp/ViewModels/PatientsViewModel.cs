using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using System.Collections.ObjectModel;
using System.Text.RegularExpressions;
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

        [ObservableProperty]
        private bool _isEditing;

        [ObservableProperty]
        private string _modalTitle = "Add New Patient";

        [ObservableProperty]
        private string _modalButtonText = "Add Patient";

        private Patient? _editingPatient;

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
            IsEditing = false;
            ModalTitle = "Add New Patient";
            ModalButtonText = "Add Patient";
            _editingPatient = null;

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
            if (string.IsNullOrWhiteSpace(NewPatientName))
            {
                System.Windows.MessageBox.Show("Patient name is required.", "Validation Error", System.Windows.MessageBoxButton.OK, System.Windows.MessageBoxImage.Error);
                return;
            }

            // NIC Validation
            if (!string.IsNullOrWhiteSpace(NewPatientId))
            {
                bool isValidNic = Regex.IsMatch(NewPatientId, @"^([0-9]{9}[xXvV]|[0-9]{12})$");
                if (!isValidNic)
                {
                    System.Windows.MessageBox.Show("Please enter a valid NIC (9 digits with V/X or 12 digits).", "Validation Error", System.Windows.MessageBoxButton.OK, System.Windows.MessageBoxImage.Error);
                    return;
                }
            }

            // Phone Validation (Sri Lanka: 0XXXXXXXXX or +94XXXXXXXXX)
            if (!string.IsNullOrWhiteSpace(NewPatientPhone))
            {
                bool isValidPhone = Regex.IsMatch(NewPatientPhone, @"^(?:0|(?:\+94))[0-9]{9}$");
                if (!isValidPhone)
                {
                    System.Windows.MessageBox.Show("Please enter a valid Sri Lankan phone number (e.g., 0712345678 or +94712345678).", "Validation Error", System.Windows.MessageBoxButton.OK, System.Windows.MessageBoxImage.Error);
                    return;
                }
            }

            int.TryParse(NewPatientAge, out int age);

            try
            {
                if (IsEditing && _editingPatient != null)
                {
                    _editingPatient.Nic = NewPatientId;
                    _editingPatient.Name = NewPatientName;
                    _editingPatient.Age = age;
                    _editingPatient.Gender = NewPatientGender;
                    _editingPatient.Phone = NewPatientPhone;
                    _editingPatient.Notes = NewPatientNotes;

                    _dbService.UpdatePatient(_editingPatient);

                    // Refresh the list to show changes (since Patient doesn't implement INotifyPropertyChanged)
                    var index = Patients.IndexOf(_editingPatient);
                    if (index != -1)
                    {
                        Patients[index] = _editingPatient;
                    }
                    System.Windows.MessageBox.Show("Patient updated successfully!", "Success", System.Windows.MessageBoxButton.OK, System.Windows.MessageBoxImage.Information);
                }
                else
                {
                    var patient = new Patient
                    {
                        Nic = NewPatientId,
                        Name = NewPatientName,
                        Age = age,
                        Gender = NewPatientGender,
                        Phone = NewPatientPhone,
                        Notes = NewPatientNotes,
                        CreatedAt = System.DateTime.Now
                    };

                    _dbService.AddPatient(patient);
                    Patients.Insert(0, patient);
                    System.Windows.MessageBox.Show("Patient added successfully!", "Success", System.Windows.MessageBoxButton.OK, System.Windows.MessageBoxImage.Information);
                }

                IsAddModalOpen = false;
            }
            catch (Exception ex)
            {
                System.Windows.MessageBox.Show($"Database Error: {ex.Message}", "Error", System.Windows.MessageBoxButton.OK, System.Windows.MessageBoxImage.Error);
            }

        }

        [RelayCommand]
        private void EditPatient(Patient patient)
        {
            if (patient == null) return;

            _editingPatient = patient;
            IsEditing = true;
            ModalTitle = "Edit Patient";
            ModalButtonText = "Update Patient";

            NewPatientId = patient.Nic ?? "";
            NewPatientName = patient.Name;
            NewPatientAge = patient.Age.ToString();
            NewPatientGender = patient.Gender;
            NewPatientPhone = patient.Phone;
            NewPatientNotes = patient.Notes ?? "";

            IsAddModalOpen = true;
        }

        [RelayCommand]
        private void DeletePatient(Patient patient)
        {
            if (patient == null) return;

            var result = System.Windows.MessageBox.Show(
                $"Are you sure you want to delete patient {patient.Name}?",
                "Confirm Delete",
                System.Windows.MessageBoxButton.YesNo,
                System.Windows.MessageBoxImage.Warning);

            if (result == System.Windows.MessageBoxResult.Yes)
            {
                _dbService.DeletePatient(patient.Id);
                Patients.Remove(patient);
            }
        }
    }
}
