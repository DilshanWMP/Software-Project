using System;
using System.Collections.Generic;
using System.Linq;
using EndoscopyApp.Services.Ai;

namespace EndoscopyApp.Tests
{
    public class DecisionLogicTests
    {
        private static List<ModelClassInfo> Classes() => new()
        {
            new() { Index = 0, Key = "no_abnormal_finding", Display = "No abnormal finding", Abnormal = false, Threshold = 0.5 },
            new() { Index = 1, Key = "polyp", Display = "Polyp", Abnormal = true, Threshold = 0.3 },
            new() { Index = 2, Key = "ulcer", Display = "", Abnormal = true, Threshold = 0.5 },
        };

        [Fact]
        public void Softmax_SumsToOne_AndMatchesReference()
        {
            var p = DecisionLogic.Softmax(new float[] { 1f, 2f, 3f });
            Assert.Equal(1.0, p.Sum(), 12);
            Assert.Equal(0.09003057, p[0], 6);
            Assert.Equal(0.24472847, p[1], 6);
            Assert.Equal(0.66524096, p[2], 6);
        }

        [Fact]
        public void Softmax_Temperature_FlattensDistribution()
        {
            var logits = new float[] { 1f, 2f, 3f };
            var p1 = DecisionLogic.Softmax(logits, 1.0);
            var p2 = DecisionLogic.Softmax(logits, 2.0);
            Assert.True(p2[2] < p1[2]);
            // softmax(x/2) == softmax(x * 0.5)
            var manual = DecisionLogic.Softmax(new float[] { 0.5f, 1f, 1.5f });
            for (int i = 0; i < 3; i++) Assert.Equal(manual[i], p2[i], 12);
        }

        [Fact]
        public void Softmax_IsStable_ForLargeLogits()
        {
            var p = DecisionLogic.Softmax(new float[] { 1000f, 1000f });
            Assert.Equal(0.5, p[0], 12);
            Assert.False(double.IsNaN(p[1]));
        }

        [Fact]
        public void Gate_UsesIndexOne_AndThreshold()
        {
            double pEndo = DecisionLogic.EndoscopyProbability(new float[] { 0f, 2f }, 1.0);
            Assert.Equal(1 / (1 + Math.Exp(-2)), pEndo, 9);
            Assert.True(DecisionLogic.IsEndoscopy(pEndo, 0.5));
            Assert.False(DecisionLogic.IsEndoscopy(0.49, 0.5));
            Assert.True(DecisionLogic.IsEndoscopy(0.5, 0.5));
        }

        [Fact]
        public void BuildFindings_SortsDescending_AndAppliesPerClassThreshold()
        {
            var f = DecisionLogic.BuildFindings(new[] { 0.25, 0.35, 0.40 }, Classes());
            Assert.Equal(new[] { "ulcer", "polyp", "no_abnormal_finding" }, f.Select(x => x.Key).ToArray());
            Assert.Equal("ulcer", f[0].DisplayName);       // empty display falls back to key
            Assert.False(f[0].AboveThreshold);             // 0.40 < 0.5
            Assert.True(f[1].AboveThreshold);              // 0.35 >= 0.3
            Assert.False(f[2].IsAbnormal);
            Assert.True(f[1].IsAbnormal);
        }

        [Fact]
        public void BuildFindings_ThresholdIsInclusive()
        {
            var f = DecisionLogic.BuildFindings(new[] { 0.5, 0.3, 0.2 }, Classes());
            Assert.True(f.Single(x => x.Key == "no_abnormal_finding").AboveThreshold);
            Assert.True(f.Single(x => x.Key == "polyp").AboveThreshold);
        }

        [Fact]
        public void BuildFindings_Throws_OnCountMismatch()
        {
            Assert.Throws<ArgumentException>(() => DecisionLogic.BuildFindings(new[] { 1.0 }, Classes()));
        }

        [Theory]
        [InlineData(0.54, 0.55, AnalysisStatus.Uncertain)]
        [InlineData(0.55, 0.55, AnalysisStatus.Ok)]
        [InlineData(0.90, 0.55, AnalysisStatus.Ok)]
        public void DecideStatus_UsesUncertainBelow(double top, double uncertainBelow, AnalysisStatus expected)
        {
            Assert.Equal(expected, DecisionLogic.DecideStatus(top, uncertainBelow));
        }

        [Fact]
        public void Messages_MatchContract()
        {
            Assert.Equal("The AI is not confident about this image", DecisionLogic.MessageFor(AnalysisStatus.Uncertain));
            Assert.Null(DecisionLogic.MessageFor(AnalysisStatus.Ok));
            Assert.Equal("Image is too dark. Image is too blurry",
                DecisionLogic.JoinProblems(new[] { QualityChecker.TooDark, QualityChecker.TooBlurry }));
        }

        [Fact]
        public void Metadata_ParsesContractExample()
        {
            const string json = """
            {
              "model_version": "gi_findings_v1", "architecture": "efficientnet_b0", "input_size": 384,
              "temperature": 1.23, "uncertain_below": 0.55,
              "classes": [
                {"index": 1, "key": "polyp", "display": "Polyp", "abnormal": true, "threshold": 0.4},
                {"index": 0, "key": "no_abnormal_finding", "display": "No abnormal finding", "abnormal": false, "threshold": 0.5}
              ],
              "preprocess": {"border_crop": true, "border_gray_threshold": 20, "border_min_fraction": 0.02, "border_min_keep": 0.25, "resize": "bilinear_stretch"},
              "quality": {"min_side_px": 128, "dark_mean_below": 25, "bright_mean_above": 235, "blur_laplacian_var_below": 15.0, "analysis_long_side_px": 512},
              "trained_on": "x", "created": "2026-11-07T12:00:00"
            }
            """;
            var m = ModelMetadata.Parse(json);
            m.ValidateClasses();
            Assert.Equal("gi_findings_v1", m.ModelVersion);
            Assert.Equal(384, m.InputSize);
            Assert.Equal(1.23, m.Temperature);
            Assert.Equal("no_abnormal_finding", m.Classes[0].Key); // sorted by index
            Assert.Equal(0.4, m.Classes[1].Threshold);
            Assert.Equal(20, m.Preprocess.BorderGrayThreshold);
            Assert.Equal(0.25, m.Preprocess.BorderMinKeep);
            Assert.Equal(512, m.Quality.AnalysisLongSidePx);
            Assert.Equal(15.0, m.Quality.BlurLaplacianVarBelow);

            var gate = ModelMetadata.Parse("""{"model_version":"endo_gate_v1","architecture":"mobilenetv3_small_100","input_size":224,"temperature":1.0,"endoscopy_threshold":0.6,"created":"x"}""");
            Assert.Equal(0.6, gate.EndoscopyThreshold);
            Assert.Equal(224, gate.InputSize);
        }
    }
}
