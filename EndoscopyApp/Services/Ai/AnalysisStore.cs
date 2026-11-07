using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace EndoscopyApp.Services.Ai
{
    /// <summary>
    /// Saves an analysis next to the image so it can be shown again:
    ///   {imageDir}/.ai/{imageFileName}.json          (AnalysisResult without the heatmap bytes, plus "schema": 1)
    ///   {imageDir}/.ai/{imageFileName}.heatmap.png   (AnalysisResult.HeatmapPng)
    /// The ".ai" folder is not listed by the galleries (they only read the patient folder's top level).
    /// </summary>
    public static class AnalysisStore
    {
        public const int SchemaVersion = 1;
        public const string FolderName = ".ai";

        // Serialises file access within the app (callers use background threads; files are small, so one lock is enough).
        private static readonly object FileLock = new();

        private static readonly JsonSerializerOptions JsonOptions = new()
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            PropertyNameCaseInsensitive = true,
            WriteIndented = true,
            Converters = { new JsonStringEnumConverter() }
        };

        /// <summary>On-disk shape (camelCase). Mirrors AnalysisResult minus the heatmap bytes.</summary>
        private sealed class StoredAnalysis
        {
            public int Schema { get; set; } = SchemaVersion;
            public AnalysisStatus Status { get; set; }
            public string? Message { get; set; }
            public List<Finding> Findings { get; set; } = new();
            public double? EndoscopyProbability { get; set; }
            public QualityReport? Quality { get; set; }
            public string? HeatmapFile { get; set; }
            public string? HeatmapClassKey { get; set; }
            public string ModelVersion { get; set; } = "";
            public string GateModelVersion { get; set; } = "";
            public double InferenceMs { get; set; }
            public DateTime AnalysedAt { get; set; }
            public string ImagePath { get; set; } = "";
            public string Disclaimer { get; set; } = AnalysisResult.DisclaimerText;
        }

        public static string GetJsonPath(string imagePath) => GetPaths(imagePath).Json;
        public static string GetHeatmapPath(string imagePath) => GetPaths(imagePath).Heatmap;

        private static (string Folder, string Json, string Heatmap) GetPaths(string imagePath)
        {
            string full = Path.GetFullPath(imagePath);
            string dir = Path.GetDirectoryName(full) ?? "";
            string name = Path.GetFileName(full);
            string folder = Path.Combine(dir, FolderName);
            return (folder, Path.Combine(folder, name + ".json"), Path.Combine(folder, name + ".heatmap.png"));
        }

        /// <summary>Writes the result (and heatmap if present). Never throws; returns false on failure.</summary>
        public static bool Save(AnalysisResult result)
        {
            try
            {
                if (result == null || string.IsNullOrWhiteSpace(result.ImagePath)) return false;
                var (folder, json, heatmap) = GetPaths(result.ImagePath);
                lock (FileLock) { SaveCore(result, folder, json, heatmap); }
                return true;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[AnalysisStore] Save failed: {ex.Message}");
                return false;
            }
        }

        private static void SaveCore(AnalysisResult result, string folder, string json, string heatmap)
        {
            var di = Directory.CreateDirectory(folder);
            try { di.Attributes |= FileAttributes.Hidden; } catch { /* cosmetic only */ }

            bool hasHeatmap = result.HeatmapPng is { Length: > 0 };
            if (hasHeatmap) WriteAtomic(heatmap, result.HeatmapPng!);
            else TryDeleteFile(heatmap);

            var stored = new StoredAnalysis
            {
                Status = result.Status,
                Message = result.Message,
                Findings = result.Findings.ToList(),
                EndoscopyProbability = result.EndoscopyProbability,
                Quality = result.Quality,
                HeatmapFile = hasHeatmap ? Path.GetFileName(heatmap) : null,
                HeatmapClassKey = result.HeatmapClassKey,
                ModelVersion = result.ModelVersion,
                GateModelVersion = result.GateModelVersion,
                InferenceMs = result.InferenceMs,
                AnalysedAt = result.AnalysedAt,
                ImagePath = result.ImagePath
            };
            WriteAtomic(json, JsonSerializer.SerializeToUtf8Bytes(stored, JsonOptions));
        }

        /// <summary>Returns the saved result for this image (heatmap included), or null if none / unreadable.</summary>
        public static AnalysisResult? Load(string imagePath)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(imagePath)) return null;
                var (_, json, heatmap) = GetPaths(imagePath);
                StoredAnalysis? stored;
                byte[]? png = null;
                lock (FileLock)
                {
                    if (!File.Exists(json)) return null;
                    stored = JsonSerializer.Deserialize<StoredAnalysis>(File.ReadAllBytes(json), JsonOptions);
                    if (stored == null || stored.Schema > SchemaVersion) return null;
                    if (stored.HeatmapFile != null && File.Exists(heatmap))
                    {
                        try { png = File.ReadAllBytes(heatmap); } catch { png = null; }
                    }
                }

                return new AnalysisResult
                {
                    Status = stored.Status,
                    Message = stored.Message,
                    Findings = stored.Findings ?? new List<Finding>(),
                    EndoscopyProbability = stored.EndoscopyProbability,
                    Quality = stored.Quality,
                    HeatmapPng = png,
                    HeatmapClassKey = stored.HeatmapClassKey,
                    ModelVersion = stored.ModelVersion ?? "",
                    GateModelVersion = stored.GateModelVersion ?? "",
                    InferenceMs = stored.InferenceMs,
                    AnalysedAt = stored.AnalysedAt,
                    ImagePath = imagePath // the folder may have moved since it was saved
                };
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[AnalysisStore] Load failed: {ex.Message}");
                return null;
            }
        }

        /// <summary>True if a saved result exists for this image.</summary>
        public static bool Exists(string imagePath)
        {
            try
            {
                return !string.IsNullOrWhiteSpace(imagePath) && File.Exists(GetPaths(imagePath).Json);
            }
            catch
            {
                return false;
            }
        }

        /// <summary>Deletes the saved result + heatmap for this image (call when the image is deleted). Never throws.</summary>
        public static void Delete(string imagePath)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(imagePath)) return;
                var (folder, json, heatmap) = GetPaths(imagePath);
                lock (FileLock)
                {
                    TryDeleteFile(json);
                    TryDeleteFile(heatmap);
                    if (Directory.Exists(folder) && !Directory.EnumerateFileSystemEntries(folder).Any())
                        Directory.Delete(folder);
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[AnalysisStore] Delete failed: {ex.Message}");
            }
        }

        private static void WriteAtomic(string path, byte[] bytes)
        {
            string tmp = path + "." + Guid.NewGuid().ToString("N") + ".tmp"; // unique: Save may run concurrently
            File.WriteAllBytes(tmp, bytes);
            File.Move(tmp, path, overwrite: true);
        }

        private static void TryDeleteFile(string path)
        {
            try
            {
                if (File.Exists(path)) File.Delete(path);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[AnalysisStore] Could not delete {path}: {ex.Message}");
            }
        }
    }
}
