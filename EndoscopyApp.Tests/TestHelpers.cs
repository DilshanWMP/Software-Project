global using Xunit;
using System;
using System.IO;
using OpenCvSharp;

namespace EndoscopyApp.Tests
{
    internal static class TestHelpers
    {
        /// <summary>Repo root (the folder containing EndoscopyApp.sln), or null.</summary>
        public static string? RepoRoot
        {
            get
            {
                var dir = new DirectoryInfo(AppContext.BaseDirectory);
                while (dir != null)
                {
                    if (File.Exists(Path.Combine(dir.FullName, "EndoscopyApp.sln"))) return dir.FullName;
                    dir = dir.Parent;
                }
                return null;
            }
        }

        /// <summary>The source AiModels folder (EndoscopyApp/AiModels).</summary>
        public static string ModelsDir => Path.Combine(RepoRoot ?? ".", "EndoscopyApp", "AiModels");

        public static bool ModelsPresent =>
            File.Exists(Path.Combine(ModelsDir, "gi_findings.onnx")) && File.Exists(Path.Combine(ModelsDir, "gi_findings.json")) &&
            File.Exists(Path.Combine(ModelsDir, "endo_gate.onnx")) && File.Exists(Path.Combine(ModelsDir, "endo_gate.json"));

        public static string NewTempDir()
        {
            string dir = Path.Combine(Path.GetTempPath(), "EndoAiTests", Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            return dir;
        }

        /// <summary>Textured mid-grey image (passes the quality rules).</summary>
        public static Mat Textured(int w, int h, int seed = 1)
        {
            var rnd = new Random(seed);
            var data = new byte[w * h * 3];
            for (int i = 0; i < data.Length; i++) data[i] = (byte)rnd.Next(60, 200);
            var m = new Mat(h, w, MatType.CV_8UC3);
            System.Runtime.InteropServices.Marshal.Copy(data, 0, m.Data, data.Length);
            return m;
        }
    }
}
