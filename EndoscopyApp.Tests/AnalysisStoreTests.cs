using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using EndoscopyApp.Services.Ai;

namespace EndoscopyApp.Tests
{
    public class AnalysisStoreTests
    {
        private static AnalysisResult Sample(string imagePath, byte[]? png) => new()
        {
            Status = AnalysisStatus.Uncertain,
            Message = "The AI is not confident about this image",
            Findings = new[]
            {
                new Finding("polyp", "Polyp", 0.45, true, false),
                new Finding("no_abnormal_finding", "No abnormal finding", 0.40, false, false),
            },
            EndoscopyProbability = 0.97,
            Quality = new QualityReport(true, 101.5, 88.25, 640, 480, new[] { "a", "b" }),
            HeatmapPng = png,
            HeatmapClassKey = "polyp",
            ModelVersion = "gi_findings_dev",
            GateModelVersion = "endo_gate_dev",
            InferenceMs = 123.4,
            AnalysedAt = new DateTime(2026, 11, 7, 10, 30, 15, 123),
            ImagePath = imagePath
        };

        [Fact]
        public void SaveLoad_RoundTrips_AndDeleteRemovesFolder()
        {
            string dir = TestHelpers.NewTempDir();
            string img = Path.Combine(dir, "IMG_20261107_103015_123.jpg");
            File.WriteAllBytes(img, new byte[] { 1 });
            var png = new byte[] { 0x89, 0x50, 0x4E, 0x47, 1, 2, 3 };

            Assert.False(AnalysisStore.Exists(img));
            Assert.True(AnalysisStore.Save(Sample(img, png)));
            Assert.True(AnalysisStore.Exists(img));
            Assert.True(File.Exists(Path.Combine(dir, ".ai", "IMG_20261107_103015_123.jpg.json")));
            Assert.True(File.Exists(Path.Combine(dir, ".ai", "IMG_20261107_103015_123.jpg.heatmap.png")));

            // camelCase, schema 1, enum as string, no heatmap bytes in json.
            using (var doc = JsonDocument.Parse(File.ReadAllText(AnalysisStore.GetJsonPath(img))))
            {
                var root = doc.RootElement;
                Assert.Equal(1, root.GetProperty("schema").GetInt32());
                Assert.Equal("Uncertain", root.GetProperty("status").GetString());
                Assert.Equal("polyp", root.GetProperty("findings")[0].GetProperty("key").GetString());
                Assert.False(root.TryGetProperty("heatmapPng", out _));
            }

            var r = AnalysisStore.Load(img);
            Assert.NotNull(r);
            Assert.Equal(AnalysisStatus.Uncertain, r!.Status);
            Assert.Equal("The AI is not confident about this image", r.Message);
            Assert.Equal(png, r.HeatmapPng);
            Assert.Equal("polyp", r.HeatmapClassKey);
            Assert.Equal("gi_findings_dev", r.ModelVersion);
            Assert.Equal("endo_gate_dev", r.GateModelVersion);
            Assert.Equal(123.4, r.InferenceMs);
            Assert.Equal(new DateTime(2026, 11, 7, 10, 30, 15, 123), r.AnalysedAt);
            Assert.Equal(0.97, r.EndoscopyProbability);
            Assert.Equal(new[] { "a", "b" }, r.Quality!.Problems.ToArray());
            Assert.Equal(640, r.Quality.Width);
            Assert.Equal(2, r.Findings.Count);
            Assert.Equal(new Finding("polyp", "Polyp", 0.45, true, false), r.Findings[0]);
            Assert.Equal("polyp", r.Top!.Key);
            Assert.Equal(img, r.ImagePath);

            AnalysisStore.Delete(img);
            Assert.False(AnalysisStore.Exists(img));
            Assert.False(Directory.Exists(Path.Combine(dir, ".ai")));
            Assert.True(File.Exists(img)); // the image itself is not touched
        }

        [Fact]
        public void Save_WithoutHeatmap_RemovesStaleHeatmap()
        {
            string dir = TestHelpers.NewTempDir();
            string img = Path.Combine(dir, "a b.jpg");
            Assert.True(AnalysisStore.Save(Sample(img, new byte[] { 1, 2 })));
            Assert.True(AnalysisStore.Save(Sample(img, null)));
            Assert.False(File.Exists(AnalysisStore.GetHeatmapPath(img)));
            Assert.Null(AnalysisStore.Load(img)!.HeatmapPng);
        }

        [Fact]
        public void Robust_NeverThrows()
        {
            string dir = TestHelpers.NewTempDir();
            string img = Path.Combine(dir, "x.jpg");
            Assert.Null(AnalysisStore.Load(img));
            Assert.Null(AnalysisStore.Load(""));
            Assert.False(AnalysisStore.Exists(""));
            AnalysisStore.Delete(img);
            AnalysisStore.Delete("");
            Assert.False(AnalysisStore.Save(new AnalysisResult { ImagePath = "" }));

            Directory.CreateDirectory(Path.Combine(dir, ".ai"));
            File.WriteAllText(AnalysisStore.GetJsonPath(img), "{ not json");
            Assert.Null(AnalysisStore.Load(img));
        }

        [Fact]
        public void ConcurrentSaves_DoNotFail()
        {
            string dir = TestHelpers.NewTempDir();
            string img = Path.Combine(dir, "c.jpg");
            var results = new bool[16];
            Parallel.For(0, results.Length, i => results[i] = AnalysisStore.Save(Sample(img, new byte[] { (byte)i })));
            Assert.All(results, Assert.True);
            Assert.NotNull(AnalysisStore.Load(img));
            Assert.Single(Directory.GetFiles(Path.Combine(dir, ".ai"), "*.json"));
        }
    }
}
