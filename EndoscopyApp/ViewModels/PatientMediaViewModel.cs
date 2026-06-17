using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using EndoscopyApp.Models;
using EndoscopyApp.Services;
using System.Collections.ObjectModel;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Threading.Tasks;
using OpenCvSharp;

namespace EndoscopyApp.ViewModels
{
    public partial class PatientMediaViewModel : ViewModelBase
    {
        private readonly MainViewModel _mainViewModel;
        private readonly DatabaseService _dbService;
        private readonly ModelInferenceService _modelService;

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
            _modelService = new ModelInferenceService();
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
            var result = MessageBox.Show($"Are you sure you want to delete {media.FileName}?", "Confirm Delete", MessageBoxButton.YesNo);
            if (result == MessageBoxResult.Yes)
            {
                if (File.Exists(media.FilePath))
                {
                    File.Delete(media.FilePath);
                    Videos.Remove(media);
                    Snapshots.Remove(media);
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

        [RelayCommand]
        private async Task AnalyzeMedia(MediaFileViewModel media)
        {
            if (!File.Exists(media.FilePath))
            {
                MessageBox.Show("Image file not found.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            if (!media.IsVideo) // Only analyze snapshots (images)
            {
                string resultMessage = "";

                try
                {
                    // Run entire analysis on background thread - don't wait for it on UI thread
                    await Task.Run(async () =>
                    {
                        try
                        {
                            // Initialize model if not already done
                            if (!_modelService.IsInitialized)
                            {
                                await _modelService.Initialize();
                            }

                            // Load image
                            using var image = OpenCvSharp.Cv2.ImRead(media.FilePath);
                            if (image.Empty())
                            {
                                resultMessage = "ERROR: Failed to load image.";
                                return;
                            }

                            // Run inference
                            var detections = await _modelService.RunInference(image);

                            if (detections.Count > 0)
                            {
                                // Draw bounding boxes on the image
                                var resultImage = image.Clone();
                                foreach (var detection in detections)
                                {
                                    int x1 = (int)(detection.X * resultImage.Width);
                                    int y1 = (int)(detection.Y * resultImage.Height);
                                    int x2 = (int)((detection.X + detection.Width) * resultImage.Width);
                                    int y2 = (int)((detection.Y + detection.Height) * resultImage.Height);

                                    // Draw rectangle
                                    OpenCvSharp.Cv2.Rectangle(resultImage, new OpenCvSharp.Point(x1, y1), new OpenCvSharp.Point(x2, y2), new OpenCvSharp.Scalar(0, 255, 0), 2);

                                    // Draw confidence score
                                    string label = $"Conf: {detection.Confidence:F2}";
                                    OpenCvSharp.Cv2.PutText(resultImage, label, new OpenCvSharp.Point(x1, y1 - 5),
                                        0, 0.5, new OpenCvSharp.Scalar(0, 255, 0), 2);
                                }

                                // Save analyzed image
                                string analyzedPath = Path.Combine(Path.GetDirectoryName(media.FilePath) ?? AppDomain.CurrentDomain.BaseDirectory, 
                                    Path.GetFileNameWithoutExtension(media.FilePath) + "_analyzed.jpg");
                                resultImage.SaveImage(analyzedPath);

                                resultMessage = $"Analysis complete!\n\nDetections found: {detections.Count}\n\n";
                                for (int i = 0; i < detections.Count; i++)
                                {
                                    resultMessage += $"Detection {i + 1}:\n" +
                                        $"  Confidence: {detections[i].Confidence:F2}\n" +
                                        $"  Class ID: {detections[i].ClassId}\n" +
                                        $"  Position: ({detections[i].X:F2}, {detections[i].Y:F2})\n\n";
                                }

                                resultMessage += $"Analyzed image saved to:\n{analyzedPath}";
                                resultImage.Dispose();
                            }
                            else
                            {
                                resultMessage = "No detections found in the image.";
                            }
                        }
                        catch (Exception ex)
                        {
                            resultMessage = $"ERROR: {ex.Message}";
                        }
                    });

                    // Show result on UI thread after background work is complete
                    if (resultMessage.StartsWith("ERROR:"))
                    {
                        MessageBox.Show(resultMessage.Replace("ERROR: ", ""), "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                    }
                    else if (resultMessage.Length > 0)
                    {
                        MessageBox.Show(resultMessage, "Analysis Results", MessageBoxButton.OK, MessageBoxImage.Information);
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Analysis failed: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
            else
            {
                MessageBox.Show("Analysis is only available for images, not videos.", "Info", MessageBoxButton.OK, MessageBoxImage.Information);
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

        public MediaFileViewModel(string filePath)
        {
            _filePath = filePath;
            _fileName = Path.GetFileName(filePath);
            _timestamp = File.GetCreationTime(filePath);

            var ext = Path.GetExtension(filePath).ToLower();
            IsVideo = _fileName.StartsWith("REC_") || ext == ".avi" || ext == ".mp4";

            _ = GenerateThumbnailAsync();
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
