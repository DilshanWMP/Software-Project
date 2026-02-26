using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using EndoscopyApp.Models;
using EndoscopyApp.Services;
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
    }

    public partial class MediaFileViewModel : ObservableObject
    {
        [ObservableProperty]
        private string _filePath;

        [ObservableProperty]
        private string _fileName;

        [ObservableProperty]
        private System.DateTime _timestamp;

        public MediaFileViewModel(string filePath)
        {
            _filePath = filePath;
            _fileName = Path.GetFileName(filePath);
            _timestamp = File.GetCreationTime(filePath);
        }
    }
}
