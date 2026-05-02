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
        public event Action<string>? NotificationRequested;
        public event Action? SnapshotFlashRequested;

        private readonly VideoCaptureService _videoService;
        private readonly DatabaseService _dbService;
        private readonly SettingsService _settingsService;
        private FootPedalService? _footPedalService;
        private readonly MainViewModel? _mainViewModel;
        private Patient? _currentPatient;
        private AppSettings _settings;

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

        private string _currentVideoFilePath = "";

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
            _settingsService = new SettingsService();
            _settings = _settingsService.LoadSettings();

            // Initialize Foot Pedal using settings
            if (!string.IsNullOrWhiteSpace(_settings.FootPedalPort) && _settings.FootPedalPort != "None")
            {
                _footPedalService = new FootPedalService(_settings.FootPedalPort);
                _footPedalService.PedalPressed += OnPedalPressed;
            }
        }

        private void OnPedalPressed()
        {
            Application.Current.Dispatcher.BeginInvoke(new Action(() =>
            {
                if (TakeSnapshotCommand.CanExecute(null))
                {
                    TakeSnapshotCommand.Execute(null);
                }
            }));
        }

        public RecordViewModel(MainViewModel mainViewModel) : this()
        {
            _mainViewModel = mainViewModel;

            // Automatically start camera when navigating to this view
            _ = StartCamera(); // Fire and forget safely
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

        private bool _isRendering = false;
        private void OnFrameReady(object? sender, WriteableBitmap bitmap)
        {
            if (_isRendering) return;

            _isRendering = true;
            System.Windows.Application.Current.Dispatcher.BeginInvoke(() =>
           {
               try
               {
                   CurrentFrame = bitmap;
               }
               finally
               {
                   _isRendering = false;
               }
           }, System.Windows.Threading.DispatcherPriority.Render);
        }

        [RelayCommand]
        public async Task StartSession()
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
                MessageBox.Show("Patient registered successfully!", "Success", MessageBoxButton.OK, MessageBoxImage.Information);

                // Start Camera automatically
                await StartCamera();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error saving patient: {ex.Message}");
            }
        }

        [RelayCommand]
        public async Task StartCamera()
        {
            if (IsCameraRunning) return;

            try
            {
                await _videoService.Start(_settings.CameraIndex);
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
                NotificationRequested?.Invoke("Recording Saved");
            }
            else
            {
                // Create directory for patient using settings path
                string baseDir = string.IsNullOrWhiteSpace(_settings.MediaPath)
                    ? Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Media")
                    : _settings.MediaPath;

                string patientDir = Path.Combine(baseDir, _currentPatient.Id.ToString());
                Directory.CreateDirectory(patientDir);
                string fileName = $"REC_{DateTime.Now:yyyyMMdd_HHmmss}.avi";
                _currentVideoFilePath = Path.Combine(patientDir, fileName);

                _videoService.StartRecording(_currentVideoFilePath);
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
                var result = MessageBox.Show("Are you sure you want to cancel? If you cancel, the recorded video will be deleted.", "Cancel Recording", MessageBoxButton.YesNo, MessageBoxImage.Warning);
                
                if (result == MessageBoxResult.Yes)
                {
                    _videoService.StopRecording();
                    IsRecording = false;

                    // Delete the video file
                    if (!string.IsNullOrEmpty(_currentVideoFilePath) && File.Exists(_currentVideoFilePath))
                    {
                        try
                        {
                            // A small delay ensures the video writer releases the file lock
                            System.Threading.Thread.Sleep(200);
                            File.Delete(_currentVideoFilePath);
                        }
                        catch (Exception ex)
                        {
                            System.Diagnostics.Debug.WriteLine($"Failed to delete cancelled video: {ex.Message}");
                        }
                    }

                    NotificationRequested?.Invoke("Recording Canceled");
                }
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
                // Create directory for patient using settings path
                string baseDir = string.IsNullOrWhiteSpace(_settings.MediaPath)
                    ? Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Media")
                    : _settings.MediaPath;

                string patientDir = Path.Combine(baseDir, _currentPatient.Id.ToString());
                Directory.CreateDirectory(patientDir);
                string fileName = $"IMG_{DateTime.Now:yyyyMMdd_HHmmss}.jpg";
                string filePath = Path.Combine(patientDir, fileName);

                frame.SaveImage(filePath);
                SnapshotFlashRequested?.Invoke();
                NotificationRequested?.Invoke("Snapshot Saved");

                // Save metadata to DB
            }
        }

        public void Cleanup()
        {
            _videoService.Stop();
            _footPedalService?.Dispose();
            _videoService.Dispose();
        }
    }

}
