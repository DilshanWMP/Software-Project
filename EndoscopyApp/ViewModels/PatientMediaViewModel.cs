using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using EndoscopyApp.Models;
using EndoscopyApp.Services;
using EndoscopyApp.Services.Ai;
using System.Collections.ObjectModel;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Windows;

namespace EndoscopyApp.ViewModels
{
    public partial class PatientMediaViewModel : ViewModelBase
    {
        private readonly MainViewModel _mainViewModel;
        private readonly DatabaseService _dbService;

        [ObservableProperty]
        private Patient _patient;

        [ObservableProperty]
        private ObservableCollection<MediaFileViewModel> _videos = new();

        [ObservableProperty]
        private ObservableCollection<MediaFileViewModel> _snapshots = new();

        [ObservableProperty]
        private int _selectedTabIndex;

        public PatientMediaViewModel(MainViewModel mainViewModel, Patient patient)
        {
            _mainViewModel = mainViewModel;
            _patient = patient;
            _dbService = new DatabaseService();
            LoadMedia();
        }

        private void LoadMedia()
        {
            Videos.Clear();
            Snapshots.Clear();

            try
            {
                var settingsService = new SettingsService();
                var settings = settingsService.LoadSettings();
                string patientDir = Path.Combine(settings.MediaPath, Patient.Id.ToString());

                if (Directory.Exists(patientDir))
                {
                    var files = Directory.GetFiles(patientDir);
                    foreach (var file in files)
                    {
                        var fileName = Path.GetFileName(file);
                        var media = new MediaFileViewModel(file);

                        if (fileName.StartsWith("REC_") || file.EndsWith(".avi") || file.EndsWith(".mp4"))
                        {
                            Videos.Add(media);
                        }
                        else if (fileName.StartsWith("IMG_") || file.EndsWith(".jpg") || file.EndsWith(".png"))
                        {
                            Snapshots.Add(media);
                        }
                    }
                }
            }
            catch (System.Exception ex)
            {
                MessageBox.Show($"Failed to load media: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        [RelayCommand]
        private void NavigateBack()
        {
            _mainViewModel.NavigateToRecordedVideos();
        }

        [RelayCommand]
        private void DownloadMedia(MediaFileViewModel media)
        {
            if (File.Exists(media.FilePath))
            {
                var extension = Path.GetExtension(media.FilePath).ToLower();
                var filter = extension switch
                {
                    ".avi" => "AVI Video|*.avi",
                    ".mp4" => "MP4 Video|*.mp4",
                    ".jpg" or ".jpeg" => "JPEG Image|*.jpg",
                    ".png" => "PNG Image|*.png",
                    _ => "All Files|*.*"
                };

                var saveFileDialog = new Microsoft.Win32.SaveFileDialog
                {
                    FileName = media.FileName,
                    Filter = filter,
                    InitialDirectory = System.Environment.GetFolderPath(System.Environment.SpecialFolder.Desktop)
                };

                if (saveFileDialog.ShowDialog() == true)
                {
                    try
                    {
                        File.Copy(media.FilePath, saveFileDialog.FileName, true);
                        MessageBox.Show("File downloaded successfully.", "Success", MessageBoxButton.OK, MessageBoxImage.Information);
                    }
                    catch (System.Exception ex)
                    {
                        MessageBox.Show($"Failed to download file: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                    }
                }
            }
        }

        [RelayCommand]
        private void DeleteMedia(MediaFileViewModel media)
        {
            var result = MessageBox.Show($"Are you sure you want to delete {media.FileName}?", "Confirm Delete", MessageBoxButton.YesNo, MessageBoxImage.Warning);
            if (result == MessageBoxResult.Yes)
            {
                if (File.Exists(media.FilePath))
                {
                    try
                    {
                        File.Delete(media.FilePath);
                        if (!media.IsVideo) AnalysisStore.Delete(media.FilePath); // saved AI result + heatmap
                        Videos.Remove(media);
                        Snapshots.Remove(media);
                    }
                    catch (System.Exception ex)
                    {
                        MessageBox.Show($"Could not delete file. It might be in use by another program.\n\nError: {ex.Message}", "Delete Failed", MessageBoxButton.OK, MessageBoxImage.Error);
                    }
                }
            }
        }

        [RelayCommand]
        private void ViewMedia(MediaFileViewModel media)
        {
            if (File.Exists(media.FilePath))
            {
                _mainViewModel.NavigateToMediaViewer(media, Patient);
            }
        }

        /// <summary>Opens a snapshot in the Media Viewer and runs the AI straight away (still images only).</summary>
        [RelayCommand]
        private void AnalyseMedia(MediaFileViewModel media)
        {
            if (media == null || media.IsVideo) return;
            if (File.Exists(media.FilePath))
            {
                _mainViewModel.NavigateToMediaViewer(media, Patient, autoAnalyse: true);
            }
        }
    }

    public partial class MediaFileViewModel : ObservableObject
    {
        [ObservableProperty]
        private string _filePath;

        [ObservableProperty]
        private string _fileName;

        [ObservableProperty]
        private System.DateTime _timestamp;

        [ObservableProperty]
        private System.Windows.Media.ImageSource? _thumbnail;

        [ObservableProperty]
        private bool _isVideo;

        /// <summary>Badge text for a saved AI analysis, e.g. "AI: Polyp". Null when none (or for videos).</summary>
        [ObservableProperty]
        [NotifyPropertyChangedFor(nameof(HasAiSummary))]
        private string? _aiSummary;

        [ObservableProperty]
        private System.Windows.Media.Brush? _aiSummaryBrush;

        public bool HasAiSummary => !string.IsNullOrEmpty(AiSummary);

        public MediaFileViewModel(string filePath)
        {
            _filePath = filePath;
            _fileName = Path.GetFileName(filePath);
            _timestamp = File.GetCreationTime(filePath);

            var ext = Path.GetExtension(filePath).ToLower();
            IsVideo = _fileName.StartsWith("REC_") || ext == ".avi" || ext == ".mp4";

            _ = GenerateThumbnailAsync();
            if (!IsVideo) _ = LoadAiSummaryAsync();
        }

        private async System.Threading.Tasks.Task LoadAiSummaryAsync()
        {
            var path = FilePath;
            var saved = await System.Threading.Tasks.Task.Run(() =>
            {
                try
                {
                    return AnalysisStore.Exists(path) ? AnalysisStore.Load(path) : null;
                }
                catch
                {
                    return null;
                }
            });
            if (saved == null) return;

            var view = AiStatusPresenter.Describe(saved);
            AiSummaryBrush = view.Accent;
            AiSummary = view.Badge;
        }

        private async System.Threading.Tasks.Task GenerateThumbnailAsync()
        {
            if (IsVideo)
            {
                Thumbnail = await System.Threading.Tasks.Task.Run(() =>
                {
                    try
                    {
                        using var capture = new OpenCvSharp.VideoCapture(FilePath);
                        if (capture.IsOpened())
                        {
                            using var mat = new OpenCvSharp.Mat();
                            if (capture.Read(mat) && !mat.Empty())
                            {
                                var bmp = OpenCvSharp.WpfExtensions.BitmapSourceConverter.ToBitmapSource(mat);
                                bmp.Freeze();
                                return bmp;
                            }
                        }
                    }
                    catch { }
                    return null;
                });
            }
            else
            {
                Thumbnail = await System.Threading.Tasks.Task.Run(() =>
                {
                    try
                    {
                        var bmp = new System.Windows.Media.Imaging.BitmapImage();
                        bmp.BeginInit();
                        bmp.CacheOption = System.Windows.Media.Imaging.BitmapCacheOption.OnLoad;
                        bmp.UriSource = new System.Uri(FilePath);
                        bmp.DecodePixelWidth = 350;
                        bmp.EndInit();
                        bmp.Freeze();
                        return bmp;
                    }
                    catch { }
                    return null;
                });
            }
        }
    }
}
