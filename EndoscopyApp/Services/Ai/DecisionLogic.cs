using System;
using System.Collections.Generic;
using System.Linq;

namespace EndoscopyApp.Services.Ai
{
    /// <summary>Pure decision functions (AI_CONTRACT.md section 4). No ONNX, so they are unit-testable.</summary>
    public static class DecisionLogic
    {
        public const string NotEndoscopyMessage = "This does not look like an endoscopy image";
        public const string UncertainMessage = "The AI is not confident about this image";
        public const string ReadErrorMessage = "Could not read the image file";

        /// <summary>Numerically stable softmax(logits / temperature).</summary>
        public static double[] Softmax(IReadOnlyList<float> logits, double temperature = 1.0)
        {
            if (logits.Count == 0) return Array.Empty<double>();
            if (!(temperature > 0)) temperature = 1.0;

            var z = new double[logits.Count];
            double max = double.NegativeInfinity;
            for (int i = 0; i < z.Length; i++)
            {
                z[i] = logits[i] / temperature;
                if (z[i] > max) max = z[i];
            }
            double sum = 0;
            for (int i = 0; i < z.Length; i++)
            {
                z[i] = Math.Exp(z[i] - max);
                sum += z[i];
            }
            for (int i = 0; i < z.Length; i++) z[i] /= sum;
            return z;
        }

        /// <summary>Gate: probability of class 1 ("endoscopy").</summary>
        public static double EndoscopyProbability(IReadOnlyList<float> gateLogits, double temperature)
        {
            var p = Softmax(gateLogits, temperature);
            return p.Length > 1 ? p[1] : 0.0;
        }

        public static bool IsEndoscopy(double pEndo, double threshold) => pEndo >= threshold;

        /// <summary>One Finding per class, sorted by probability (highest first, ties by class index).</summary>
        public static IReadOnlyList<Finding> BuildFindings(IReadOnlyList<double> probs, IReadOnlyList<ModelClassInfo> classes)
        {
            if (probs.Count != classes.Count)
                throw new ArgumentException($"Got {probs.Count} probabilities for {classes.Count} classes.");

            return Enumerable.Range(0, probs.Count)
                .OrderByDescending(i => probs[i])
                .ThenBy(i => i)
                .Select(i =>
                {
                    var c = classes[i];
                    string display = string.IsNullOrWhiteSpace(c.Display) ? c.Key : c.Display;
                    return new Finding(c.Key, display, probs[i], c.Abnormal, probs[i] >= c.Threshold);
                })
                .ToList();
        }

        /// <summary>Ok when the top probability reaches uncertainBelow, else Uncertain.</summary>
        public static AnalysisStatus DecideStatus(double topProbability, double uncertainBelow)
            => topProbability < uncertainBelow ? AnalysisStatus.Uncertain : AnalysisStatus.Ok;

        public static string? MessageFor(AnalysisStatus status) => status switch
        {
            AnalysisStatus.Uncertain => UncertainMessage,
            AnalysisStatus.NotEndoscopy => NotEndoscopyMessage,
            _ => null
        };

        /// <summary>Joins quality problems into the NotSuitable message.</summary>
        public static string JoinProblems(IEnumerable<string> problems) => string.Join(". ", problems);
    }
}
