using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using EndoscopyApp.Models;
using EndoscopyApp.Services;
using Microsoft.Win32;

namespace EndoscopyApp.ViewModels
{
    public partial class PredictionsViewModel : ViewModelBase
    {
        private readonly MainViewModel? _mainViewModel;
        private readonly PredictionService _predictionService;

        [ObservableProperty]
        [NotifyPropertyChangedFor(nameof(CanAnalyze))]
        private string? _selectedImagePath;

        [ObservableProperty]
        private string _selectedFileName = "No image selected";

        [ObservableProperty]
        private BitmapImage? _imagePreview;

        [ObservableProperty]
        [NotifyPropertyChangedFor(nameof(CanAnalyze))]
        private bool _isBusy;

        [ObservableProperty]
        private bool _hasResult;

        [ObservableProperty]
        private PredictionResult? _result;

        [ObservableProperty]
        private ObservableCollection<ProbabilityItem> _rankedProbabilities = new();

        [ObservableProperty]
        [NotifyPropertyChangedFor(nameof(HasError))]
        private string? _errorMessage;

        public bool HasError => !string.IsNullOrEmpty(ErrorMessage);

        public bool CanAnalyze => !IsBusy && !string.IsNullOrEmpty(SelectedImagePath) && File.Exists(SelectedImagePath);

        public string Disclaimer => "Decision support only. Requires clinician review.";

        public PredictionsViewModel() : this(null, null)
        {
        }

        public PredictionsViewModel(MainViewModel? mainViewModel, string? initialImagePath = null)
        {
            _mainViewModel = mainViewModel;
            _predictionService = new PredictionService();

            if (!string.IsNullOrEmpty(initialImagePath) && File.Exists(initialImagePath))
            {
                SetSelectedImage(initialImagePath);
            }
        }

        [RelayCommand]
        public void Browse()
        {
            var dialog = new OpenFileDialog
            {
                Title = "Select Laparoscopy Snapshot",
                Filter = "Image Files (*.jpg;*.jpeg;*.png)|*.jpg;*.jpeg;*.png|All Files (*.*)|*.*"
            };

            if (dialog.ShowDialog() == true)
            {
                SetSelectedImage(dialog.FileName);
            }
        }

        public void SetSelectedImage(string path)
        {
            try
            {
                SelectedImagePath = path;
                SelectedFileName = Path.GetFileName(path);
                ErrorMessage = null;
                HasResult = false;
                Result = null;
                RankedProbabilities.Clear();

                var bitmap = new BitmapImage();
                bitmap.BeginInit();
                bitmap.CacheOption = BitmapCacheOption.OnLoad;
                bitmap.UriSource = new Uri(path, UriKind.Absolute);
                bitmap.EndInit();
                bitmap.Freeze();
                ImagePreview = bitmap;
            }
            catch (Exception ex)
            {
                ErrorMessage = $"Failed to load image: {ex.Message}";
                ImagePreview = null;
            }
        }

        [RelayCommand]
        public async Task AnalyzeAsync()
        {
            if (string.IsNullOrEmpty(SelectedImagePath) || !File.Exists(SelectedImagePath))
            {
                ErrorMessage = "Please select a valid image file first.";
                return;
            }

            IsBusy = true;
            ErrorMessage = null;

            try
            {
                var isAvailable = await _predictionService.IsServiceAvailableAsync();
                if (!isAvailable)
                {
                    ErrorMessage = "Prediction service is offline at http://127.0.0.1:8000. Please start the service.";
                    return;
                }

                var prediction = await _predictionService.PredictAsync(SelectedImagePath);
                Result = prediction;

                RankedProbabilities.Clear();
                if (prediction.Probabilities != null)
                {
                    foreach (var pair in prediction.Probabilities.OrderByDescending(p => p.Value))
                    {
                        RankedProbabilities.Add(new ProbabilityItem
                        {
                            Label = pair.Key.Replace("_", " "),
                            Probability = pair.Value
                        });
                    }
                }

                HasResult = true;
            }
            catch (Exception ex)
            {
                ErrorMessage = $"Analysis failed: {ex.Message}";
                HasResult = false;
            }
            finally
            {
                IsBusy = false;
            }
        }

        [RelayCommand]
        public void Clear()
        {
            SelectedImagePath = null;
            SelectedFileName = "No image selected";
            ImagePreview = null;
            Result = null;
            HasResult = false;
            RankedProbabilities.Clear();
            ErrorMessage = null;
        }

        [RelayCommand]
        public void NavigateBack()
        {
            _mainViewModel?.NavigateToHome();
        }
    }
}

