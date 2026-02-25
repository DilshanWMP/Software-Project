using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using EndoscopyApp.Models;
using EndoscopyApp.Services;
using System.Collections.ObjectModel;
using System.IO;
using System;
using System.Collections.Generic;
using System.Linq;

namespace EndoscopyApp.ViewModels
{
    public partial class GalleryViewModel : ViewModelBase
    {
        private readonly DatabaseService _dbService;

        [ObservableProperty]
        private ObservableCollection<Patient> _patients = new();

        [ObservableProperty]
        private Patient? _selectedPatient;

        [ObservableProperty]
        private ObservableCollection<string> _mediaFiles = new();
        // Storing file paths for simplicity in Phase 1

        public GalleryViewModel()
        {
            _dbService = new DatabaseService();
            LoadPatients();
        }

        private void LoadPatients()
        {
            var list = _dbService.GetAllPatients();
            Patients = new ObservableCollection<Patient>(list);
        }

        partial void OnSelectedPatientChanged(Patient? value)
        {
            if (value != null)
            {
                LoadMedia(value);
            }
            else
            {
                MediaFiles.Clear();
            }
        }

        private void LoadMedia(Patient patient)
        {
            MediaFiles.Clear();
            var settingsService = new SettingsService();
            var settings = settingsService.LoadSettings();
            string patientDir = Path.Combine(settings.MediaPath, patient.Id.ToString());
            
            if (Directory.Exists(patientDir))
            {
                var files = Directory.GetFiles(patientDir);
                foreach (var file in files)
                {
                    MediaFiles.Add(file);
                }
            }
        }

        [RelayCommand]
        public void OpenFile(string filePath)
        {
            if (File.Exists(filePath))
            {
                try
                {
                    var psi = new System.Diagnostics.ProcessStartInfo() { FileName = filePath, UseShellExecute = true };
                    System.Diagnostics.Process.Start(psi);
                }
                catch (Exception ex)
                {
                    // Handle error
                }
            }
        }
    }
}
