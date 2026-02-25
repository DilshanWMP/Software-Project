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
        private bool _isPatientRegistered;

        public ObservableCollection<string> GenderOptions { get; } = new ObservableCollection<string> { "Male", "Female", "Other" };

        public RecordViewModel()
        {
            _videoService = new VideoCaptureService();
            _videoService.FrameReady += OnFrameReady;
            _dbService = new DatabaseService();
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
             if (!_isCameraRunning || _currentPatient == null) return;

             if (IsRecording)
             {
                 _videoService.StopRecording();
                 IsRecording = false;
                 MessageBox.Show("Recording Saved.");
             }
             else
             {
                 var settingsService = new SettingsService();
                 var settings = settingsService.LoadSettings();
                 string patientDir = Path.Combine(settings.MediaPath, _currentPatient.Id.ToString());
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
        public void TakeSnapshot()
        {
            if (!_isCameraRunning || _currentPatient == null) return;

             var frame = _videoService.CaptureSnapshot();
             if (!frame.Empty())
             {
                 var settingsService = new SettingsService();
                 var settings = settingsService.LoadSettings();
                 string patientDir = Path.Combine(settings.MediaPath, _currentPatient.Id.ToString());
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
