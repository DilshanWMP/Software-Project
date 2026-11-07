using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using EndoscopyApp.Services.Ai;
using OpenCvSharp;
using Xunit.Abstractions;

namespace EndoscopyApp.Tests
{
    public class EngineTests
    {
        private readonly ITestOutputHelper _out;
        public EngineTests(ITestOutputHelper output) => _out = output;

        private const string ModelsMissing = "Dev models not present in EndoscopyApp/AiModels (Agent A writes them).";

        [Fact]
        public async Task MissingModels_IsUnavailable_AndReturnsError()
        {
            using var engine = new FindingsEngine(TestHelpers.NewTempDir());
            Assert.False(engine.IsAvailable);
            Assert.Equal("AI model files not found (AiModels folder).", engine.UnavailableReason);

            var r = await engine.AnalyseAsync("C:\\nowhere\\x.jpg");
            Assert.Equal(AnalysisStatus.Error, r.Status);
            Assert.Equal(engine.UnavailableReason, r.Message);
            Assert.Equal("C:\\nowhere\\x.jpg", r.ImagePath);
        }

        [Fact]
        public void CorruptModels_IsUnavailable_WithReason()
        {
            string dir = TestHelpers.NewTempDir();
            File.WriteAllText(Path.Combine(dir, "gi_findings.onnx"), "garbage");
            File.WriteAllText(Path.Combine(dir, "endo_gate.onnx"), "garbage");
            File.WriteAllText(Path.Combine(dir, "gi_findings.json"),
                """{"model_version":"x","input_size":8,"temperature":1,"classes":[{"index":0,"key":"a","display":"A","abnormal":false,"threshold":0.5}]}""");
            File.WriteAllText(Path.Combine(dir, "endo_gate.json"), """{"model_version":"g","input_size":8,"temperature":1}""");
            using var engine = new FindingsEngine(dir);
            Assert.False(engine.IsAvailable);
            Assert.StartsWith("AI model could not be loaded", engine.UnavailableReason);
        }

        [SkippableFact]
        public async Task MissingOrBadFile_ReturnsReadError()
        {
            Skip.IfNot(TestHelpers.ModelsPresent, ModelsMissing);
            using var engine = new FindingsEngine(TestHelpers.ModelsDir);
            Assert.True(engine.IsAvailable, engine.UnavailableReason);

            var r = await engine.AnalyseAsync(Path.Combine(TestHelpers.NewTempDir(), "missing.jpg"));
            Assert.Equal(AnalysisStatus.Error, r.Status);
            Assert.Equal("Could not read the image file", r.Message);

            var r2 = await engine.AnalyseAsync(null!);
            Assert.Equal(AnalysisStatus.Error, r2.Status);
        }

        [SkippableFact]
        public async Task SyntheticImages_GiveExpectedStatuses_AndHeatmapSize()
        {
            Skip.IfNot(TestHelpers.ModelsPresent, ModelsMissing);
            using var engine = new FindingsEngine(TestHelpers.ModelsDir);
            Assert.True(engine.IsAvailable, engine.UnavailableReason);
            string dir = TestHelpers.NewTempDir();

            // Too dark -> NotSuitable, no findings.
            string dark = Path.Combine(dir, "dark.png");
            using (var m = new Mat(300, 400, MatType.CV_8UC3, Scalar.All(5))) Cv2.ImWrite(dark, m);
            var rd = await engine.AnalyseAsync(dark);
            Assert.Equal(AnalysisStatus.NotSuitable, rd.Status);
            Assert.Contains("Image is too dark", rd.Message);
            Assert.Empty(rd.Findings);
            Assert.NotNull(rd.Quality);

            // Bordered textured image: whatever the dev gate says, the result must be consistent.
            string tex = Path.Combine(dir, "tex.png");
            using (var m = new Mat(400, 500, MatType.CV_8UC3, Scalar.All(0)))
            using (var t = TestHelpers.Textured(400, 360))
            using (var roi = new Mat(m, new Rect(50, 20, 400, 360)))
            {
                t.CopyTo(roi);
                Cv2.ImWrite(tex, m);
            }
            var rt = await engine.AnalyseAsync(tex);
            Assert.NotNull(rt.EndoscopyProbability);
            Assert.Equal(engine.ModelVersion, rt.ModelVersion);
            if (rt.Status is AnalysisStatus.Ok or AnalysisStatus.Uncertain)
            {
                Assert.Equal(engine.FindingsMetadata!.Classes.Count, rt.Findings.Count);
                Assert.Equal(1.0, rt.Findings.Sum(f => f.Probability), 6);
                Assert.NotNull(rt.HeatmapPng);
                using var hm = Cv2.ImDecode(rt.HeatmapPng!, ImreadModes.Unchanged);
                Assert.Equal(500, hm.Width);
                Assert.Equal(400, hm.Height);
                Assert.Equal(0, hm.At<Vec4b>(5, 5)[3]); // inside the black border: transparent
                Assert.Equal(rt.Top!.Key, rt.HeatmapClassKey);
            }
            else
            {
                Assert.Equal(AnalysisStatus.NotEndoscopy, rt.Status);
                Assert.Equal("This does not look like an endoscopy image", rt.Message);
            }
        }

        [SkippableFact]
        public async Task Cancellation_ReturnsError_NotException()
        {
            Skip.IfNot(TestHelpers.ModelsPresent, ModelsMissing);
            using var engine = new FindingsEngine(TestHelpers.ModelsDir);
            using var cts = new CancellationTokenSource();
            cts.Cancel();
            var r = await engine.AnalyseAsync("whatever.jpg", cts.Token);
            Assert.Equal(AnalysisStatus.Error, r.Status);
            Assert.Equal(FindingsEngine.CancelledMessage, r.Message);
        }

        // ---------- golden parity (contract section 6) ----------

        private sealed class GoldenCase
        {
            public string Image { get; set; } = "";
            public int[] Crop { get; set; } = Array.Empty<int>();
            public double[] FindingsProbs { get; set; } = Array.Empty<double>();
            public double GatePEndo { get; set; }
            public string TopKey { get; set; } = "";
        }

        [SkippableFact]
        public void GoldenParity_MatchesPython()
        {
            // ENDO_GOLDEN_DIR lets a developer point the test at an ad-hoc golden folder.
            string golden = Environment.GetEnvironmentVariable("ENDO_GOLDEN_DIR") is { Length: > 0 } g ? g : Path.Combine(TestHelpers.RepoRoot ?? ".", "ml", "golden");
            string casesPath = Path.Combine(golden, "cases.json");
            Skip.IfNot(File.Exists(casesPath), "ml/golden/cases.json not present yet (Agent A generates it with make_golden.py).");
            Skip.IfNot(TestHelpers.ModelsPresent, ModelsMissing);

            var cases = JsonSerializer.Deserialize<List<GoldenCase>>(File.ReadAllText(casesPath),
                new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower, PropertyNameCaseInsensitive = true })!;
            Assert.NotEmpty(cases);

            using var engine = new FindingsEngine(TestHelpers.ModelsDir);
            Assert.True(engine.IsAvailable, engine.UnavailableReason);
            var classes = engine.FindingsMetadata!.Classes;

            // Golden numbers are only valid for the model files they were made from.
            string modelsJson = Path.Combine(golden, "models.json");
            if (File.Exists(modelsJson))
            {
                using var doc = JsonDocument.Parse(File.ReadAllText(modelsJson));
                if (doc.RootElement.TryGetProperty("findings_model_version", out var fv))
                    Assert.True(fv.GetString() == engine.ModelVersion,
                        $"ml/golden was made for {fv.GetString()} but AiModels has {engine.ModelVersion}: rerun ml/src/make_golden.py.");
                if (doc.RootElement.TryGetProperty("gate_model_version", out var gv))
                    Assert.True(gv.GetString() == engine.GateModelVersion,
                        $"ml/golden was made for {gv.GetString()} but AiModels has {engine.GateModelVersion}: rerun ml/src/make_golden.py.");
            }

            foreach (var c in cases)
            {
                string img = Path.Combine(golden, "images", c.Image);
                var raw = engine.RunRaw(img);
                Assert.True(raw != null, $"{c.Image}: could not be read");
                var r = raw!.Crop;
                int[] crop = { r.X, r.Y, r.X + r.Width - 1, r.Y + r.Height - 1 };
                Assert.True(c.Crop.SequenceEqual(crop), $"{c.Image}: crop [{string.Join(",", crop)}] != python [{string.Join(",", c.Crop)}]");

                Assert.Equal(c.FindingsProbs.Length, raw.FindingsProbs.Length);
                double maxDiff = c.FindingsProbs.Zip(raw.FindingsProbs, (a, b) => Math.Abs(a - b)).Max();
                double gateDiff = Math.Abs(c.GatePEndo - raw.GatePEndo);
                _out.WriteLine($"{c.Image}: crop ok, max |dp| = {maxDiff:E2}, gate |dp| = {gateDiff:E2}");
                Assert.True(maxDiff <= 1e-3, $"{c.Image}: findings probs differ by {maxDiff}");
                Assert.True(gateDiff <= 1e-3, $"{c.Image}: gate p_endo differs by {gateDiff}");

                int top = Array.IndexOf(raw.FindingsProbs, raw.FindingsProbs.Max());
                if (!string.IsNullOrEmpty(c.TopKey) && maxDiff < 1e-4)
                    Assert.Equal(c.TopKey, classes[top].Key);
            }
        }

        // ---------- end-to-end timings on real captured snapshots ----------

        [SkippableFact]
        public async Task EndToEnd_OnCapturedSnapshots_ReportsTimings()
        {
            Skip.IfNot(TestHelpers.ModelsPresent, ModelsMissing);
            string media = Path.Combine(TestHelpers.RepoRoot ?? ".", "EndoscopyApp", "bin", "Debug", "net9.0-windows", "Media");
            var images = Directory.Exists(media)
                ? Directory.GetFiles(media, "IMG_*.jpg", SearchOption.AllDirectories).Where(p => !p.Contains(Path.DirectorySeparatorChar + ".ai" + Path.DirectorySeparatorChar)).Take(20).ToList()
                : new List<string>();
            Skip.If(images.Count == 0, "No IMG_*.jpg snapshots under EndoscopyApp/bin/Debug/net9.0-windows/Media.");

            var sw = Stopwatch.StartNew();
            using var engine = new FindingsEngine(TestHelpers.ModelsDir);
            _out.WriteLine($"Cold model load: {engine.LoadMs:F0} ms (ctor wall {sw.ElapsedMilliseconds} ms), available={engine.IsAvailable}, threads={Math.Min(4, Environment.ProcessorCount)}");
            Assert.True(engine.IsAvailable, engine.UnavailableReason);

            var times = new List<double>();
            for (int i = 0; i < images.Count; i++)
            {
                var r = await engine.AnalyseAsync(images[i]);
                Assert.NotEqual(AnalysisStatus.Error, r.Status);
                times.Add(r.InferenceMs);
                using var probe = Cv2.ImRead(images[i], ImageDecode);
                _out.WriteLine($"{Path.GetFileName(images[i])} {probe.Width}x{probe.Height}: {r.Status} {r.InferenceMs:F0} ms top={r.Top?.Key}:{r.Top?.Probability:F3} pEndo={r.EndoscopyProbability:F3} q=[{string.Join("; ", r.Quality?.Problems ?? Array.Empty<string>())}]");
            }
            var warm = times.Skip(1).DefaultIfEmpty(times[0]).ToList();
            _out.WriteLine($"First analyse: {times[0]:F0} ms; warm mean {warm.Average():F0} ms, max {warm.Max():F0} ms over {warm.Count} images");
        }

        private const ImreadModes ImageDecode = ImagePreprocessor.DecodeFlags;
    }
}
