using System.IO;
using System.Threading.Tasks;
using System.Windows.Media;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;

namespace EndoscopyApp.ViewModels
{
    /// <summary>
    /// "AI Analysis" screen: analyse any image file (e.g. a test image) with the in-process AI engine.
    /// Results are NOT saved (SaveResults = false), so nothing is written next to the chosen file.
    /// </summary>
    public partial class PredictionsViewModel : ViewModelBase
    {
        private readonly MainViewModel? _mainViewModel;

        public AiAnalysisViewModel Ai { get; } = new AiAnalysisViewModel(saveResults: false);

        [ObservableProperty]
        private string _selectedFileName = "No image selected";

        [ObservableProperty]
        [NotifyPropertyChangedFor(nameof(HasImagePreview))]
        private ImageSource? _imagePreview;

        [ObservableProperty]
        [NotifyPropertyChangedFor(nameof(HasError))]
        private string? _errorMessage;

        public bool HasImagePreview => ImagePreview != null;
        public bool HasError => !string.IsNullOrEmpty(ErrorMessage);

        public PredictionsViewModel() : this(null, null)
        {
        }

        public PredictionsViewModel(MainViewModel? mainViewModel, string? initialImagePath = null)
        {
            _mainViewModel = mainViewModel;

            if (!string.IsNullOrEmpty(initialImagePath) && File.Exists(initialImagePath))
            {
                _ = SetSelectedImageAsync(initialImagePath);
            }
        }

        [RelayCommand]
        public void Browse()
        {
            var dialog = new OpenFileDialog
            {
                Title = "Select endoscopy image",
                Filter = "Image Files (*.jpg;*.jpeg;*.png;*.bmp)|*.jpg;*.jpeg;*.png;*.bmp|All Files (*.*)|*.*"
            };

            if (dialog.ShowDialog() == true)
            {
                _ = SetSelectedImageAsync(dialog.FileName);
            }
        }

        public async Task SetSelectedImageAsync(string path)
        {
            SelectedFileName = Path.GetFileName(path);
            ErrorMessage = null;
            ImagePreview = null;
            var setTask = Ai.SetImageAsync(path);

            var bitmap = await Task.Run(() => AiAnalysisViewModel.LoadImageUnlocked(path));
            if (Ai.ImagePath != path) return; // another image was chosen meanwhile
            ImagePreview = bitmap;
            if (bitmap == null)
            {
                ErrorMessage = "Failed to load the image preview. The file may not be a supported image.";
            }
            await setTask;
        }

        [RelayCommand]
        public void Clear()
        {
            SelectedFileName = "No image selected";
            ImagePreview = null;
            ErrorMessage = null;
            _ = Ai.SetImageAsync(null);
        }

        [RelayCommand]
        public void NavigateBack()
        {
            _mainViewModel?.NavigateToHome();
        }
    }
}
