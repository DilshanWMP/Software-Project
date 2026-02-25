using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using System.Windows.Media.Imaging;
using EndoscopyApp.Services;
using System.Windows.Threading;

namespace EndoscopyApp.ViewModels
{
    public partial class LiveViewModel : ViewModelBase
    {
        private readonly VideoCaptureService _videoService;
        private readonly MainViewModel? _mainViewModel;

        [ObservableProperty]
        private WriteableBitmap? _currentFrame;

        [ObservableProperty]
        private bool _isCameraRunning;

        public LiveViewModel()
        {
            _videoService = new VideoCaptureService();
            _videoService.FrameReady += OnFrameReady;
        }

        public LiveViewModel(MainViewModel mainViewModel) : this()
        {
            _mainViewModel = mainViewModel;
        }

        private void OnFrameReady(object? sender, WriteableBitmap bitmap)
        {
            // Update UI on UI thread
            System.Windows.Application.Current.Dispatcher.Invoke(() =>
            {
                CurrentFrame = bitmap;
            });
        }

        [RelayCommand]
        public void StartCamera()
        {
            try
            {
                _videoService.Start();
                IsCameraRunning = true;
            }
            catch (System.Exception ex)
            {
                // Handle error
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
            _mainViewModel?.NavigateToHome();
        }

        // Cleanup when navigating away or closing
        public void Cleanup()
        {
            _videoService.Stop();
            _videoService.Dispose();
        }
    }
}
