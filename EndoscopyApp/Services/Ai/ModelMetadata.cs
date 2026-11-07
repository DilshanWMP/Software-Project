using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace EndoscopyApp.Services.Ai
{
    /// <summary>One class entry of gi_findings.json (index order = logit order).</summary>
    public sealed class ModelClassInfo
    {
        public int Index { get; set; }
        public string Key { get; set; } = "";
        public string Display { get; set; } = "";
        public bool Abnormal { get; set; }
        public double Threshold { get; set; } = 0.5;
    }

    /// <summary>Border crop settings (contract section 2).</summary>
    public sealed class PreprocessSettings
    {
        public bool BorderCrop { get; set; } = true;
        public int BorderGrayThreshold { get; set; } = 20;
        public double BorderMinFraction { get; set; } = 0.02;
        public double BorderMinKeep { get; set; } = 0.25;
        public string Resize { get; set; } = "bilinear_stretch";
    }

    /// <summary>Quality rule thresholds (contract section 3).</summary>
    public sealed class QualitySettings
    {
        public int MinSidePx { get; set; } = 128;
        public double DarkMeanBelow { get; set; } = 25;
        public double BrightMeanAbove { get; set; } = 235;
        public double BlurLaplacianVarBelow { get; set; } = 15.0;
        public int AnalysisLongSidePx { get; set; } = 512;
    }

    /// <summary>
    /// Metadata json shipped next to each ONNX model (gi_findings.json, endo_gate.json). snake_case on disk.
    /// The gate json only uses ModelVersion/Architecture/InputSize/Temperature/EndoscopyThreshold.
    /// </summary>
    public sealed class ModelMetadata
    {
        public string ModelVersion { get; set; } = "";
        public string Architecture { get; set; } = "";
        public int InputSize { get; set; }
        public double Temperature { get; set; } = 1.0;
        public double UncertainBelow { get; set; } = 0.55;
        public double EndoscopyThreshold { get; set; } = 0.5;
        public List<ModelClassInfo> Classes { get; set; } = new();
        public PreprocessSettings Preprocess { get; set; } = new();
        public QualitySettings Quality { get; set; } = new();
        public string? TrainedOn { get; set; }
        public string? Created { get; set; }

        private static readonly JsonSerializerOptions JsonOptions = new()
        {
            PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
            PropertyNameCaseInsensitive = true,
            ReadCommentHandling = JsonCommentHandling.Skip,
            AllowTrailingCommas = true,
            NumberHandling = JsonNumberHandling.AllowReadingFromString
        };

        public static ModelMetadata Parse(string json)
        {
            var meta = JsonSerializer.Deserialize<ModelMetadata>(json, JsonOptions)
                       ?? throw new InvalidDataException("Model metadata json is empty.");
            meta.Preprocess ??= new PreprocessSettings();
            meta.Quality ??= new QualitySettings();
            meta.Classes ??= new List<ModelClassInfo>();
            // Classes are matched to logits by index, so order them by index.
            meta.Classes.Sort((a, b) => a.Index.CompareTo(b.Index));
            if (meta.InputSize <= 0) throw new InvalidDataException("Model metadata has no valid input_size.");
            if (meta.Temperature <= 0) throw new InvalidDataException("Model metadata temperature must be > 0.");
            return meta;
        }

        public static ModelMetadata Load(string path) => Parse(File.ReadAllText(path));

        /// <summary>Checks that class indices are exactly 0..C-1 (throws otherwise).</summary>
        public void ValidateClasses()
        {
            if (Classes.Count == 0) throw new InvalidDataException("Model metadata has no classes.");
            for (int i = 0; i < Classes.Count; i++)
            {
                if (Classes[i].Index != i)
                    throw new InvalidDataException($"Model metadata class indices must be 0..{Classes.Count - 1}.");
                if (string.IsNullOrWhiteSpace(Classes[i].Key))
                    throw new InvalidDataException($"Model metadata class {i} has no key.");
            }
        }
    }
}
