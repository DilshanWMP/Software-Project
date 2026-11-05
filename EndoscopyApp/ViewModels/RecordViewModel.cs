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
        public void ToggleRecording() => ToggleRecordingCore(fromHardware: false);

        /// <summary>Hardware push button: long press -> start / stop recording.</summary>
        public void OnHardwareLongPress() => ToggleRecordingCore(fromHardware: true);

        /// <summary>Hardware push button: short press -> snapshot (works while recording or not).</summary>
        public void OnHardwareShortPress() => _ = TakeSnapshotCore(fromHardware: true);

        /// <summary>
        /// UI clicks keep the original MessageBox behaviour. Hardware presses must never
        /// open a modal dialog (the operator cannot click it while holding the scope),
        /// so they report problems with the on-screen toast instead.
        /// </summary>
        private void ReportProblem(string message, bool fromHardware)
        {
            if (fromHardware) NotificationRequested?.Invoke(message);
            else MessageBox.Show(message);
        }

        private string GetPatientDirectory(Patient patient)
        {
            string baseDir = string.IsNullOrWhiteSpace(_settings.MediaPath)
                ? Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Media")
                : _settings.MediaPath;

            string patientDir = Path.Combine(baseDir, patient.Id.ToString());
            Directory.CreateDirectory(patientDir);
            return patientDir;
        }

        private void ToggleRecordingCore(bool fromHardware)
        {
            if (!IsCameraRunning)
            {
                if (fromHardware) NotificationRequested?.Invoke("Camera is not running");
                return;
            }

            // Check if patient is available for recording/snapshots, but allow viewing without one
            if (_currentPatient == null)
            {
                ReportProblem("Please register or select a patient first.", fromHardware);
                return;
            }

            if (IsRecording)
            {
                _videoService.StopRecording();
                IsRecording = false;
                NotificationRequested?.Invoke("Recording Saved");
                return;
            }

            try
            {
                string patientDir = GetPatientDirectory(_currentPatient);
                string fileName = $"REC_{DateTime.Now:yyyyMMdd_HHmmss}.avi";
                string filePath = Path.Combine(patientDir, fileName);

                if (_videoService.StartRecording(filePath))
                {
                    IsRecording = true;
                    NotificationRequested?.Invoke("Recording Started");
                }
                else
                {
                    ReportProblem("Could not start recording (video file could not be created).", fromHardware);
                }
            }
            catch (Exception ex)
            {
                ReportProblem($"Could not start recording: {ex.Message}", fromHardware);
            }

            // Save metadata to DB (Placeholder implementation as DB service AddMedia not shown in previous step fully)
            // media = new MediaFile { PatientId = _currentPatient.Id, FilePath = filePath, FileType = "Video" }
            // _dbService.AddMedia(media);
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
            Cleanup();
            _mainViewModel?.NavigateToHome();
        }

        [RelayCommand]
        public Task TakeSnapshot() => TakeSnapshotCore(fromHardware: false);

        private bool _isTakingSnapshot;

        private async Task TakeSnapshotCore(bool fromHardware)
        {
            if (!IsCameraRunning)
            {
                if (fromHardware) NotificationRequested?.Invoke("Camera is not running");
                return;
            }

            if (_currentPatient == null)
            {
                ReportProblem("Please register or select a patient first.", fromHardware);
                return;
            }

            if (_isTakingSnapshot) return; // ignore presses while the previous one is saving
            _isTakingSnapshot = true;
            try
            {
                var patient = _currentPatient;
                using var frame = await _videoService.CaptureSnapshotAsync();
                if (frame.Empty())
                {
                    ReportProblem("Snapshot failed (no camera frame).", fromHardware);
                    return;
                }

                string patientDir = GetPatientDirectory(patient);
                // Milliseconds in the name so quick successive presses don't overwrite each other
                string fileName = $"IMG_{DateTime.Now:yyyyMMdd_HHmmss_fff}.jpg";
                string filePath = Path.Combine(patientDir, fileName);

                await Task.Run(() => frame.SaveImage(filePath));
                SnapshotFlashRequested?.Invoke();
                NotificationRequested?.Invoke("Snapshot Saved");

                // Save metadata to DB
            }
            catch (Exception ex)
            {
                ReportProblem($"Snapshot failed: {ex.Message}", fromHardware);
            }
            finally
            {
                _isTakingSnapshot = false;
            }
        }

        public void Cleanup()
        {
            if (IsRecording)
            {
                _videoService.StopRecording();
                IsRecording = false;
            }

            if (!IsCameraRunning) return;

            _videoService.Stop();
            IsCameraRunning = false;
            _videoService.Dispose();
        }
    }

}
