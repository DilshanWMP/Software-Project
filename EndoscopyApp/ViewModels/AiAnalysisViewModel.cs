using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using EndoscopyApp.Services.Ai;
using MaterialDesignThemes.Wpf;

namespace EndoscopyApp.ViewModels
{
    /// <summary>
    /// Reusable state for the AI result panel (Views/Controls/AiResultPanel.xaml).
    /// Used by the Media Viewer (SaveResults = true) and the AI Analysis screen (SaveResults = false).
    /// Still images only; never pass a video.
    /// </summary>
    public partial class AiAnalysisViewModel : ObservableObject
    {
        private const int TopFindingCount = 3;

        // The engine loads its ONNX models in its constructor, so create it once, off the UI thread.
        private static Task<IFindingsEngine?>? _engineTask;
        private static string? _engineLoadError;

        private IFindingsEngine? _engine;
        private int _generation; // bumped whenever the image changes, so stale async work is discarded

        /// <summary>When true, saved results are loaded/written through AnalysisStore. When false, nothing touches disk.</summary>
        public bool SaveResults { get; }

        public string Disclaimer => AnalysisResult.DisclaimerText;

        [ObservableProperty]
        [NotifyPropertyChangedFor(nameof(HasImage), nameof(ShowEmptyState), nameof(EmptyStateText))]
        [NotifyCanExecuteChangedFor(nameof(AnalyseCommand))]
        private string? _imagePath;

        [ObservableProperty]
        [NotifyPropertyChangedFor(nameof(ShowEmptyState), nameof(AnalyseButtonText))]
        [NotifyCanExecuteChangedFor(nameof(AnalyseCommand))]
        private bool _isBusy;

        [ObservableProperty]
        [NotifyPropertyChangedFor(nameof(ShowEmptyState), nameof(ShowEngineUnavailable))]
        [NotifyCanExecuteChangedFor(nameof(AnalyseCommand))]
        private bool _isEngineLoading = true;

        [ObservableProperty]
        [NotifyPropertyChangedFor(nameof(ShowEmptyState), nameof(ShowEngineUnavailable))]
        [NotifyCanExecuteChangedFor(nameof(AnalyseCommand))]
        private bool _isEngineAvailable;

        [ObservableProperty]
        [NotifyPropertyChangedFor(nameof(UnavailableMessage))]
        private string? _unavailableReason;

        [ObservableProperty]
        [NotifyPropertyChangedFor(nameof(HasResult), nameof(ShowEmptyState), nameof(AnalyseButtonText))]
        private AnalysisResult? _result;

        [ObservableProperty] private string _statusTitle = "";
        [ObservableProperty] private string _statusMessage = "";
        [ObservableProperty] private bool _hasStatusMessage;
        [ObservableProperty] private Brush _statusAccentBrush = AiPalette.Neutral;
        [ObservableProperty] private Brush _statusBackgroundBrush = AiPalette.NeutralBg;
        [ObservableProperty] private PackIconKind _statusIcon = PackIconKind.InformationOutline;

        [ObservableProperty] private IReadOnlyList<FindingItem> _topFindings = Array.Empty<FindingItem>();
        [ObservableProperty] private IReadOnlyList<FindingItem> _otherFindings = Array.Empty<FindingItem>();
        [ObservableProperty] private bool _hasFindings;
        [ObservableProperty] private bool _hasOtherFindings;
        [ObservableProperty] private string _showAllText = "";

        [ObservableProperty]
        [NotifyPropertyChangedFor(nameof(ShowAllText))]
        private bool _showAllFindings;

        [ObservableProperty]
        [NotifyPropertyChangedFor(nameof(CanShowHeatmap))]
        [NotifyCanExecuteChangedFor(nameof(ToggleHeatmapCommand))]
        private BitmapSource? _heatmapImage;

        [ObservableProperty]
        [NotifyPropertyChangedFor(nameof(HeatmapButtonText))]
        private bool _showHeatmap;

        [ObservableProperty] private string _heatmapCaption = "";
        [ObservableProperty] private string _metadataText = "";

        public bool HasImage => !string.IsNullOrEmpty(ImagePath);
        public bool HasResult => Result != null;
        public bool CanShowHeatmap => HeatmapImage != null;
        public bool ShowEngineUnavailable => !IsEngineLoading && !IsEngineAvailable;
        public string UnavailableMessage => $"AI model not available: {UnavailableReason ?? "unknown reason"}";
        public bool ShowEmptyState => !HasResult && !IsBusy && !IsEngineLoading && IsEngineAvailable;
        public string EmptyStateText => HasImage
            ? "This image has not been analysed yet. Click Analyse to run the AI."
            : "Choose an image to analyse.";
        public string AnalyseButtonText => HasResult ? "Re-analyse" : "Analyse";
        public string HeatmapButtonText => ShowHeatmap ? "Hide heatmap" : "Show heatmap";

        public AiAnalysisViewModel(bool saveResults)
        {
            SaveResults = saveResults;
            _initTask = InitEngineAsync();
        }

        private readonly Task _initTask;

        private static Task<IFindingsEngine?> GetEngineAsync()
        {
            // Only ever called from the UI thread, so the null-coalescing assignment is safe.
            return _engineTask ??= Task.Run<IFindingsEngine?>(() =>
            {
                try
                {
                    return FindingsEngine.Shared;
                }
                catch (Exception ex)
                {
                    _engineLoadError = ex.Message;
                    return null;
                }
            });
        }

        private async Task InitEngineAsync()
        {
            var engine = await GetEngineAsync();
            _engine = engine;
            if (engine == null)
            {
                IsEngineAvailable = false;
                UnavailableReason = _engineLoadError ?? "the AI engine could not be created.";
            }
            else
            {
                IsEngineAvailable = engine.IsAvailable;
                UnavailableReason = engine.IsAvailable ? null : (engine.UnavailableReason ?? "model files are missing.");
            }
            IsEngineLoading = false;
        }

        /// <summary>
        /// Shows a new image (or none). Clears the old result; with SaveResults, loads any saved result.
        /// With autoAnalyse, runs the AI if there is no saved result.
        /// </summary>
        public async Task SetImageAsync(string? path, bool autoAnalyse = false)
        {
            int gen = ++_generation;
            ImagePath = path;
            IsBusy = false;
            ClearResult();

            if (string.IsNullOrEmpty(path)) return;

            if (SaveResults)
            {
                AnalysisResult? saved = null;
                try
                {
                    saved = await Task.Run(() => AnalysisStore.Load(path));
                }
                catch
                {
                    saved = null;
                }
                if (gen != _generation) return;
                if (saved != null)
                {
                    await ApplyResultAsync(saved, gen);
                    return;
                }
            }

            if (autoAnalyse)
            {
                await _initTask;
                if (gen != _generation) return;
                if (AnalyseCommand.CanExecute(null))
                {
                    await AnalyseCommand.ExecuteAsync(null);
                }
            }
        }

        private bool CanAnalyse() =>
            !IsBusy && !IsEngineLoading && IsEngineAvailable && _engine != null && HasImage;

        [RelayCommand(CanExecute = nameof(CanAnalyse))]
        private async Task AnalyseAsync()
        {
            var engine = _engine;
            var path = ImagePath;
            if (engine == null || string.IsNullOrEmpty(path)) return;

            int gen = ++_generation;
            IsBusy = true;
            try
            {
                AnalysisResult result;
                try
                {
                    result = await engine.AnalyseAsync(path);
                }
                catch (Exception ex)
                {
                    result = new AnalysisResult
                    {
                        Status = AnalysisStatus.Error,
                        Message = $"Analysis failed: {ex.Message}",
                        ImagePath = path,
                        ModelVersion = engine.ModelVersion
                    };
                }

                if (gen != _generation) return;

                if (SaveResults && result.Status != AnalysisStatus.Error)
                {
                    try { await Task.Run(() => AnalysisStore.Save(result)); } catch { /* Save never throws by contract */ }
                }

                await ApplyResultAsync(result, gen);
            }
            finally
            {
                if (gen == _generation) IsBusy = false;
            }
        }

        [RelayCommand(CanExecute = nameof(CanShowHeatmap))]
        private void ToggleHeatmap()
        {
            ShowHeatmap = !ShowHeatmap;
        }

        [RelayCommand]
        private void ToggleShowAll()
        {
            ShowAllFindings = !ShowAllFindings;
        }

        partial void OnShowAllFindingsChanged(bool value) => UpdateShowAllText();

        private void UpdateShowAllText()
        {
            ShowAllText = ShowAllFindings ? "Show fewer" : $"Show all ({OtherFindings.Count} more)";
        }

        private void ClearResult()
        {
            Result = null;
            ShowHeatmap = false;
            HeatmapImage = null;
            HeatmapCaption = "";
            TopFindings = Array.Empty<FindingItem>();
            OtherFindings = Array.Empty<FindingItem>();
            HasFindings = false;
            HasOtherFindings = false;
            ShowAllFindings = false;
            StatusTitle = "";
            StatusMessage = "";
            HasStatusMessage = false;
            MetadataText = "";
        }

        private async Task ApplyResultAsync(AnalysisResult result, int gen)
        {
            BitmapSource? heatmap = null;
            if (result.HeatmapPng is { Length: > 0 } png)
            {
                heatmap = await Task.Run(() => DecodePng(png));
            }
            if (gen != _generation) return;

            ClearResult();

            var view = AiStatusPresenter.Describe(result);
            StatusTitle = view.Title;
            StatusMessage = view.Message;
            HasStatusMessage = !string.IsNullOrWhiteSpace(view.Message);
            StatusAccentBrush = view.Accent;
            StatusBackgroundBrush = view.Background;
            StatusIcon = view.Icon;

            var items = result.Findings.Select(f => new FindingItem(f)).ToList();
            TopFindings = items.Take(TopFindingCount).ToList();
            OtherFindings = items.Skip(TopFindingCount).ToList();
            HasFindings = items.Count > 0;
            HasOtherFindings = OtherFindings.Count > 0;
            UpdateShowAllText();

            HeatmapImage = heatmap;
            if (heatmap != null)
            {
                var cls = result.Findings.FirstOrDefault(f => f.Key == result.HeatmapClassKey) ?? result.Top;
                HeatmapCaption = cls != null
                    ? $"Heatmap shows the image areas that most influenced \"{cls.DisplayName}\". It is not an outline of a lesion."
                    : "Heatmap shows the image areas that most influenced the top result. It is not an outline of a lesion.";
            }

            var model = string.IsNullOrWhiteSpace(result.ModelVersion) ? "unknown" : result.ModelVersion;
            MetadataText = $"Analysed {result.AnalysedAt:dd MMM yyyy HH:mm} · model {model} · {result.InferenceMs:F0} ms";

            Result = result;
        }

        private static BitmapSource? DecodePng(byte[] png)
        {
            try
            {
                using var ms = new MemoryStream(png);
                var bmp = new BitmapImage();
                bmp.BeginInit();
                bmp.CacheOption = BitmapCacheOption.OnLoad;
                bmp.StreamSource = ms;
                bmp.EndInit();
                bmp.Freeze();
                return bmp;
            }
            catch
            {
                return null;
            }
        }

        /// <summary>Loads an image file into memory (file is not kept locked). Returns null on failure. Thread-safe (frozen).</summary>
        public static BitmapSource? LoadImageUnlocked(string path)
        {
            try
            {
                var bmp = new BitmapImage();
                bmp.BeginInit();
                bmp.CacheOption = BitmapCacheOption.OnLoad;
                bmp.CreateOptions = BitmapCreateOptions.IgnoreImageCache;
                bmp.UriSource = new Uri(path, UriKind.Absolute);
                bmp.EndInit();
                bmp.Freeze();
                return bmp;
            }
            catch
            {
                return null;
            }
        }
    }

    /// <summary>One bar in the findings list.</summary>
    public sealed class FindingItem
    {
        public FindingItem(Finding finding)
        {
            Name = finding.DisplayName;
            Probability = finding.Probability;
            IsAbnormal = finding.IsAbnormal;
            AboveThreshold = finding.AboveThreshold;
            BarBrush = IsAbnormal && AboveThreshold ? AiPalette.Finding : AiPalette.Bar;
        }

        public string Name { get; }
        public double Probability { get; }
        public bool IsAbnormal { get; }
        public bool AboveThreshold { get; }
        public Brush BarBrush { get; }
        public string PercentText => $"{Probability * 100:F1}%";
    }

    internal static class AiPalette
    {
        private static SolidColorBrush Make(string hex)
        {
            var b = new SolidColorBrush((Color)ColorConverter.ConvertFromString(hex));
            b.Freeze();
            return b;
        }

        public static readonly Brush Finding = Make("#B7791F");     // amber
        public static readonly Brush FindingBg = Make("#FFFFF0");
        public static readonly Brush Calm = Make("#2B6CB0");        // neutral blue
        public static readonly Brush CalmBg = Make("#EBF8FF");
        public static readonly Brush Neutral = Make("#718096");     // grey
        public static readonly Brush NeutralBg = Make("#F7FAFC");
        public static readonly Brush Warning = Make("#DD6B20");     // orange
        public static readonly Brush WarningBg = Make("#FFFAF0");
        public static readonly Brush Error = Make("#C53030");       // red
        public static readonly Brush ErrorBg = Make("#FFF5F5");
        public static readonly Brush Bar = Make("#4C51BF");
    }

    internal sealed record AiStatusView(string Title, string Message, Brush Accent, Brush Background, PackIconKind Icon, string? Badge);

    /// <summary>Turns an AnalysisResult into display text/colours. Wording is decision-support only (no diagnostic claims).</summary>
    internal static class AiStatusPresenter
    {
        public static AiStatusView Describe(AnalysisResult r)
        {
            var top = r.Top;
            switch (r.Status)
            {
                case AnalysisStatus.Ok when top != null && top.IsAbnormal && top.AboveThreshold:
                    return new AiStatusView(
                        $"Possible finding: {top.DisplayName}",
                        $"Top result: {top.DisplayName} ({top.Probability:P0}).",
                        AiPalette.Finding, AiPalette.FindingBg, PackIconKind.AlertCircleOutline,
                        $"AI: {top.DisplayName}");

                case AnalysisStatus.Ok when top != null && !top.IsAbnormal:
                    return new AiStatusView(
                        "No abnormal finding detected",
                        $"Top result: {top.DisplayName} ({top.Probability:P0}).",
                        AiPalette.Calm, AiPalette.CalmBg, PackIconKind.CheckCircleOutline,
                        $"AI: {top.DisplayName}");

                case AnalysisStatus.Ok:
                    // Abnormal top below its own threshold (or no findings at all): treat as uncertain.
                    return new AiStatusView(
                        "Uncertain",
                        top != null
                            ? $"The top result ({top.DisplayName}, {top.Probability:P0}) is below its decision threshold."
                            : "The AI returned no findings for this image.",
                        AiPalette.Neutral, AiPalette.NeutralBg, PackIconKind.HelpCircleOutline,
                        "AI: Uncertain");

                case AnalysisStatus.Uncertain:
                    return new AiStatusView(
                        "Uncertain",
                        r.Message ?? "The AI is not confident about this image.",
                        AiPalette.Neutral, AiPalette.NeutralBg, PackIconKind.HelpCircleOutline,
                        "AI: Uncertain");

                case AnalysisStatus.NotSuitable:
                    var problems = r.Quality?.Problems is { Count: > 0 } p
                        ? string.Join(Environment.NewLine, p.Select(x => "• " + x))
                        : (r.Message ?? "The image failed the quality check.");
                    return new AiStatusView(
                        "Image not suitable for AI analysis",
                        problems,
                        AiPalette.Warning, AiPalette.WarningBg, PackIconKind.ImageOff,
                        "AI: Not suitable");

                case AnalysisStatus.NotEndoscopy:
                    return new AiStatusView(
                        "Not an endoscopy image",
                        r.Message ?? "This does not look like an endoscopy image.",
                        AiPalette.Warning, AiPalette.WarningBg, PackIconKind.ImageRemove,
                        "AI: Not endoscopy");

                default:
                    return new AiStatusView(
                        "Analysis failed",
                        r.Message ?? "Something went wrong while analysing this image.",
                        AiPalette.Error, AiPalette.ErrorBg, PackIconKind.AlertOctagonOutline,
                        null);
            }
        }
    }
}
