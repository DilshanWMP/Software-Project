using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace EndoscopyApp.Models
{
    public class PredictionResult
    {
        [JsonPropertyName("model_version")]
        public string ModelVersion { get; set; } = string.Empty;

        [JsonPropertyName("top_label")]
        public string TopLabel { get; set; } = string.Empty;

        [JsonPropertyName("confidence")]
        public double Confidence { get; set; }

        [JsonPropertyName("probabilities")]
        public Dictionary<string, double> Probabilities { get; set; } = new();

        [JsonPropertyName("finding_detected")]
        public bool FindingDetected { get; set; }

        [JsonPropertyName("decision_threshold")]
        public double DecisionThreshold { get; set; }

        [JsonPropertyName("uncertain")]
        public bool Uncertain { get; set; }

        [JsonPropertyName("image_size")]
        public List<int> ImageSize { get; set; } = new();

        [JsonPropertyName("inference_time_ms")]
        public double InferenceTimeMs { get; set; }

        [JsonPropertyName("disclaimer")]
        public string Disclaimer { get; set; } = "Decision support only. Requires clinician review.";
    }

    public class ProbabilityItem
    {
        public string Label { get; set; } = string.Empty;
        public double Probability { get; set; }
        public string PercentageText => $"{Probability * 100:F1}%";
    }
}

