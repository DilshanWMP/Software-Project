using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using System.Windows.Media.Imaging;
using EndoscopyApp.Services;
using EndoscopyApp.Models;
using System.Windows.Threading;

namespace EndoscopyApp.ViewModels
{
    public partial class LiveViewModel : ViewModelBase
    {
        private readonly VideoCaptureService _videoService;
        private readonly MainViewModel? _mainViewModel;
        private readonly SettingsService _settingsService;
        private readonly ModelInferenceService _modelService;
        private AppSettings _settings;

        [ObservableProperty]
        private WriteableBitmap? _currentFrame;

        [ObservableProperty]
        private bool _isCameraRunning;

        [ObservableProperty]
        private bool _isModelEnabled;

        [ObservableProperty]
        private string _inferenceStatus = "Model not loaded";

        public LiveViewModel()
        {
            _videoService = new VideoCaptureService();
            _videoService.FrameReady += OnFrameReady;
            _settingsService = new SettingsService();
            _settings = _settingsService.LoadSettings();
            _modelService = new ModelInferenceService();
        }

        public LiveViewModel(MainViewModel mainViewModel) : this()
        {
            _mainViewModel = mainViewModel;
        }

        private bool _isRendering = false;
        private void OnFrameReady(object? sender, WriteableBitmap bitmap)
        {
            if (_isRendering) return; // Drop frame if UI is still busy

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
        public async Task EnableModel()
        {
            if (IsModelEnabled) return;

            try
            {
                InferenceStatus = "Loading model...";
                await _modelService.Initialize();
                IsModelEnabled = _modelService.IsInitialized;
                InferenceStatus = IsModelEnabled ? "Model ready" : "Failed to load model";
            }
            catch (Exception ex)
            {
                InferenceStatus = $"Error: {ex.Message}";
                System.Windows.MessageBox.Show($"Model Loading Error: {ex.Message}");
            }
        }

        [RelayCommand]
        public void DisableModel()
        {
            IsModelEnabled = false;
            InferenceStatus = "Model disabled";
        }

        [RelayCommand]
        public async Task StartCamera()
        {
            if (IsCameraRunning) return;

            try
            {
                // Run camera initialization in the background to prevent UI freezing
                await _videoService.Start(_settings.CameraIndex);
                IsCameraRunning = true;
            }
            catch (System.Exception ex)
            {
                System.Windows.MessageBox.Show($"Camera Error: {ex.Message}");
            }
        }

        [RelayCommand]
        public void StopCamera()
        {
            _videoService.Stop();
            IsCameraRunning = false;
        }

        [RelayCommand]
        public void NavigateBack()
        {
            Cleanup();
            _mainViewModel?.NavigateToHome();
        }

        // Cleanup when navigating away or closing
        public void Cleanup()
        {
            if (!IsCameraRunning) return;

            _videoService.StopRecording();
            _videoService.Stop();
            IsCameraRunning = false;
            _videoService.Dispose();
        }
    }
}
