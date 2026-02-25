using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using System.Windows.Media.Imaging;
using EndoscopyApp.Services;
using EndoscopyApp.Models;
using System.IO;
using System;
using System.Windows;
using System.Collections.ObjectModel;

namespace EndoscopyApp.ViewModels
{
    public partial class RecordViewModel : ViewModelBase
    {
        private readonly VideoCaptureService _videoService;
        private readonly DatabaseService _dbService;
        private readonly MainViewModel? _mainViewModel;
        private Patient? _currentPatient;

        [ObservableProperty]
        private string _patientName = "";
        [ObservableProperty]
        private string _patientAge = "";
        [ObservableProperty]
        private string _patientGender = "Male";
        [ObservableProperty]
        private string _patientPhone = "";

        [ObservableProperty]
        private WriteableBitmap? _currentFrame;

        [ObservableProperty]
        private bool _isCameraRunning;

        [ObservableProperty]
        private bool _isRecording;

        [ObservableProperty]
        private string _patientId = "000000";

        [ObservableProperty]
        private bool _isPatientRegistered;

        [ObservableProperty]
        private bool _isFullscreen;

        public ObservableCollection<string> GenderOptions { get; } = new ObservableCollection<string> { "Male", "Female", "Other" };

        public RecordViewModel()
        {
            _videoService = new VideoCaptureService();
            _videoService.FrameReady += OnFrameReady;
            _dbService = new DatabaseService();
        }

        public RecordViewModel(MainViewModel mainViewModel) : this()
        {
            _mainViewModel = mainViewModel;

            // Automatically start camera when navigating to this view
            StartCamera();
        }

        public void SetPatient(Patient patient)
        {
            _currentPatient = patient;
            PatientName = patient.Name;
            PatientAge = patient.Age.ToString();
            PatientGender = patient.Gender;
            PatientPhone = patient.Phone;
            PatientId = patient.Id.ToString("D6");
            IsPatientRegistered = true;
        }

        private void OnFrameReady(object? sender, WriteableBitmap bitmap)
        {
            System.Windows.Application.Current.Dispatcher.Invoke(() =>
           {
               CurrentFrame = bitmap;
           });
        }

        [RelayCommand]
        public void StartSession()
        {
            if (string.IsNullOrWhiteSpace(PatientName))
            {
                MessageBox.Show("Please enter patient name.");
                return;
            }

            if (!int.TryParse(PatientAge, out int age))
            {
                MessageBox.Show("Invalid Age.");
                return;
            }

            _currentPatient = new Patient
            {
                Name = PatientName,
                Age = age,
                Gender = PatientGender,
                Phone = PatientPhone,
                CreatedAt = DateTime.Now
            };

            // Save to DB
            try
            {
                _dbService.AddPatient(_currentPatient);
                IsPatientRegistered = true;

                // Start Camera automatically
                StartCamera();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error saving patient: {ex.Message}");
            }
        }

        [RelayCommand]
        public void StartCamera()
        {
            try
            {
                _videoService.Start();
                IsCameraRunning = true;
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Camera Error: {ex.Message}");
            }
        }

        [RelayCommand]
        public void StopCamera()
        {
            _videoService.Stop();
            IsCameraRunning = false;
        }

        [RelayCommand]
        public void ToggleRecording()
        {
            if (!IsCameraRunning) return;

            // Check if patient is available for recording/snapshots, but allow viewing without one
            if (_currentPatient == null)
            {
                MessageBox.Show("Please register or select a patient first.");
                return;
            }

            if (IsRecording)
            {
                _videoService.StopRecording();
                IsRecording = false;
                MessageBox.Show("Recording Saved.");
            }
            else
            {
                // Create directory for patient
                string patientDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Media", _currentPatient.Id.ToString());
                Directory.CreateDirectory(patientDir);
                string fileName = $"REC_{DateTime.Now:yyyyMMdd_HHmmss}.avi";
                string filePath = Path.Combine(patientDir, fileName);

                _videoService.StartRecording(filePath);
                IsRecording = true;

                // Save metadata to DB (Placeholder implementation as DB service AddMedia not shown in previous step fully)
                // media = new MediaFile { PatientId = _currentPatient.Id, FilePath = filePath, FileType = "Video" }
                // _dbService.AddMedia(media);
            }
        }

        [RelayCommand]
        public void CancelRecording()
        {
            if (IsRecording)
            {
                _videoService.StopRecording();
                IsRecording = false;
                // In a real app, we might delete the temporary file here
                MessageBox.Show("Recording Cancelled.");
            }
        }

        [RelayCommand]
        public void ToggleFullscreen()
        {
            IsFullscreen = !IsFullscreen;
        }

        [RelayCommand]
        public void NavigateBack()
        {
            _mainViewModel?.NavigateToHome();
        }

        [RelayCommand]
        public void TakeSnapshot()
        {
            if (!IsCameraRunning) return;

            if (_currentPatient == null)
            {
                MessageBox.Show("Please register or select a patient first.");
                return;
            }

            var frame = _videoService.CaptureSnapshot();
            if (!frame.Empty())
            {
                string patientDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Media", _currentPatient.Id.ToString());
                Directory.CreateDirectory(patientDir);
                string fileName = $"IMG_{DateTime.Now:yyyyMMdd_HHmmss}.jpg";
                string filePath = Path.Combine(patientDir, fileName);

                frame.SaveImage(filePath);
                MessageBox.Show("Snapshot Saved.");

                // Save metadata to DB
            }
        }

        public void Cleanup()
        {
            _videoService.Stop();
            _videoService.Dispose();
        }
    }
}
