using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace EndoscopyApp.Services.Ai
{
    /// <summary>Overall outcome of analysing one image.</summary>
    public enum AnalysisStatus
    {
        /// <summary>Findings computed and the top finding is confident enough.</summary>
        Ok,
        /// <summary>Findings computed but the model is not confident (top probability below the uncertain threshold).</summary>
        Uncertain,
        /// <summary>Image failed the quality check (blurry, too dark, too bright, too small). No findings.</summary>
        NotSuitable,
        /// <summary>The gate model says this is not an endoscopy image. No findings.</summary>
        NotEndoscopy,
        /// <summary>Something went wrong (file unreadable, model missing). See Message.</summary>
        Error
    }

    /// <summary>One class from the findings model, with its calibrated probability.</summary>
    public sealed record Finding(
        string Key,            // stable id from the model json, e.g. "polyp"
        string DisplayName,    // text shown to the doctor, e.g. "Polyp"
        double Probability,    // calibrated softmax probability, 0..1
        bool IsAbnormal,       // false only for "no abnormal finding" / "other_procedure"-type classes
        bool AboveThreshold);  // Probability >= this class's threshold from the model json

    /// <summary>Result of the quality rules (run before any model).</summary>
    public sealed record QualityReport(
        bool IsOk,
        double Brightness,      // mean gray 0..255 of the cropped image
        double Sharpness,       // variance of Laplacian of the cropped image
        int Width,
        int Height,
        IReadOnlyList<string> Problems); // human-readable, e.g. "Image is too blurry"

    /// <summary>Everything the UI needs to show (and what gets saved next to the image).</summary>
    public sealed class AnalysisResult
    {
        public const string DisclaimerText = "Decision support only. Requires clinician review.";

        public AnalysisStatus Status { get; init; }

        /// <summary>Short explanation for NotSuitable / NotEndoscopy / Uncertain / Error. Null when Ok.</summary>
        public string? Message { get; init; }

        /// <summary>All classes, sorted by probability (highest first). Empty unless Status is Ok or Uncertain.</summary>
        public IReadOnlyList<Finding> Findings { get; init; } = Array.Empty<Finding>();

        public Finding? Top => Findings.Count > 0 ? Findings[0] : null;

        /// <summary>Probability that the image is an endoscopy image (gate model). Null if the gate did not run.</summary>
        public double? EndoscopyProbability { get; init; }

        public QualityReport? Quality { get; init; }

        /// <summary>PNG bytes, same pixel size as the original image, RGBA: coloured heat with alpha, transparent elsewhere.
        /// Overlay it on the original image with identical Stretch. Null when there are no findings.</summary>
        public byte[]? HeatmapPng { get; init; }

        /// <summary>Which class the heatmap explains (normally Top.Key).</summary>
        public string? HeatmapClassKey { get; init; }

        public string ModelVersion { get; init; } = "";
        public string GateModelVersion { get; init; } = "";
        public double InferenceMs { get; init; }
        public DateTime AnalysedAt { get; init; } = DateTime.Now;
        public string ImagePath { get; init; } = "";
        public string Disclaimer => DisclaimerText;
    }

    /// <summary>Runs the AI on a single still image. Implemented by FindingsEngine (in-process ONNX Runtime, CPU).</summary>
    public interface IFindingsEngine : IDisposable
    {
        /// <summary>False when the model files are missing or failed to load.</summary>
        bool IsAvailable { get; }

        /// <summary>Why IsAvailable is false (shown to the user). Null when available.</summary>
        string? UnavailableReason { get; }

        string ModelVersion { get; }

        /// <summary>Never throws for bad input: returns Status = Error with a Message instead.
        /// Runs off the UI thread; safe to call concurrently.</summary>
        Task<AnalysisResult> AnalyseAsync(string imagePath, CancellationToken cancellationToken = default);
    }
}
