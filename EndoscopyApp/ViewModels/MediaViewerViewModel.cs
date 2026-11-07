using System.Threading.Tasks;
using System.Windows.Media;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using EndoscopyApp.Models;

namespace EndoscopyApp.ViewModels
{
    public partial class MediaViewerViewModel : ViewModelBase
    {
        private readonly MainViewModel _mainViewModel;
        private readonly Patient _patient;

        [ObservableProperty]
        private MediaFileViewModel _media;

        /// <summary>The still image, loaded into memory so the file is not locked (it may be deleted later).</summary>
        [ObservableProperty]
        private ImageSource? _imageSource;

        /// <summary>AI panel state. Null for videos (AI is for still images only).</summary>
        public AiAnalysisViewModel? Ai { get; }

        public bool ShowAiPanel => Ai != null;

        /// <summary>Source for the video player; null for images so the MediaElement never opens (and locks) a snapshot.</summary>
        public string? VideoPath => Media.IsVideo ? Media.FilePath : null;

        public MediaViewerViewModel(MainViewModel mainViewModel, MediaFileViewModel media, Patient patient, bool autoAnalyse = false)
        {
            _mainViewModel = mainViewModel;
            _media = media;
            _patient = patient;

            if (!media.IsVideo)
            {
                Ai = new AiAnalysisViewModel(saveResults: true);
                _ = LoadImageAsync(media.FilePath);
                _ = Ai.SetImageAsync(media.FilePath, autoAnalyse);
            }
        }

        private async Task LoadImageAsync(string path)
        {
            ImageSource = await Task.Run(() => AiAnalysisViewModel.LoadImageUnlocked(path));
        }

        [RelayCommand]
        private void NavigateBack()
        {
            _mainViewModel.NavigateToPatientMedia(_patient);
        }
    }
}
