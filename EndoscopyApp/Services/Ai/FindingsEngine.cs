using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace EndoscopyApp.Services.Ai
{
    /// <summary>Crop and calibrated probabilities straight from the models (used by the golden parity test).</summary>
    public sealed record RawModelOutputs(OpenCvSharp.Rect Crop, double[] FindingsProbs, double GatePEndo);

    /// <summary>
    /// In-process ONNX Runtime (CPU) engine. Loads AiModels/gi_findings.onnx(+json) and AiModels/endo_gate.onnx(+json)
    /// once, then runs: decode + border crop -> quality check -> gate -> findings -> heatmap (AI_CONTRACT.md).
    /// </summary>
    public sealed class FindingsEngine : IFindingsEngine
    {
        public const string FindingsModelName = "gi_findings";
        public const string GateModelName = "endo_gate";
        public const string CancelledMessage = "Analysis was cancelled";

        private const string InputName = "input";
        private const string LogitsName = "logits";
        private const string CamName = "cam";

        private static readonly Lazy<FindingsEngine> _shared =
            new(() => new FindingsEngine(DefaultModelsDirectory), LazyThreadSafetyMode.ExecutionAndPublication);

        /// <summary>The one engine for the whole app (models are loaded once, on first access; this can take a second or two).</summary>
        public static FindingsEngine Shared => _shared.Value;

        /// <summary>Creates/loads Shared on a background thread so the UI thread never blocks on model loading.</summary>
        public static Task<FindingsEngine> GetSharedAsync() => Task.Run(() => Shared);

        /// <summary>{exe dir}/AiModels</summary>
        public static string DefaultModelsDirectory => Path.Combine(AppContext.BaseDirectory, "AiModels");

        private InferenceSession? _findings;
        private InferenceSession? _gate;
        private ModelMetadata? _findingsMeta;
        private ModelMetadata? _gateMeta;
        private int _camClassAxisLength;
        private volatile bool _disposed;
        // Analyses hold the read side (they run concurrently); Dispose takes the write side so sessions are never freed mid-run.
        private readonly ReaderWriterLockSlim _lifetime = new(LockRecursionPolicy.SupportsRecursion);

        public bool IsAvailable => !_disposed && _findings != null && _gate != null;
        public string? UnavailableReason { get; private set; }
        public string ModelVersion => _findingsMeta?.ModelVersion ?? "";
        public string GateModelVersion => _gateMeta?.ModelVersion ?? "";

        /// <summary>Metadata of the findings model (null when unavailable).</summary>
        public ModelMetadata? FindingsMetadata => _findingsMeta;

        /// <summary>Metadata of the gate model (null when unavailable).</summary>
        public ModelMetadata? GateMetadata => _gateMeta;

        /// <summary>Wall time spent loading both models (ms).</summary>
        public double LoadMs { get; }

        /// <summary>Loads the models from a folder (tests use their own folder; the app uses Shared).</summary>
        public FindingsEngine(string modelsDirectory)
        {
            var sw = Stopwatch.StartNew();
            try
            {
                Load(modelsDirectory);
            }
            catch (Exception ex)
            {
                DisposeSessions();
                _findingsMeta = null;
                _gateMeta = null;
                UnavailableReason = ex is FileNotFoundException
                    ? "AI model files not found (AiModels folder)."
                    : "AI model could not be loaded: " + ex.Message;
                Debug.WriteLine($"[FindingsEngine] {UnavailableReason}");
            }
            LoadMs = sw.Elapsed.TotalMilliseconds;
        }

        private void Load(string dir)
        {
            string fOnnx = Path.Combine(dir, FindingsModelName + ".onnx");
            string fJson = Path.Combine(dir, FindingsModelName + ".json");
            string gOnnx = Path.Combine(dir, GateModelName + ".onnx");
            string gJson = Path.Combine(dir, GateModelName + ".json");
            foreach (var f in new[] { fOnnx, fJson, gOnnx, gJson })
            {
                if (!File.Exists(f)) throw new FileNotFoundException("Missing model file.", f);
            }

            var findingsMeta = ModelMetadata.Load(fJson);
            findingsMeta.ValidateClasses();
            var gateMeta = ModelMetadata.Load(gJson);

            using var options = new SessionOptions
            {
                IntraOpNumThreads = Math.Min(4, Environment.ProcessorCount),
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
            };

            _findings = new InferenceSession(fOnnx, options);
            _gate = new InferenceSession(gOnnx, options);

            int c = findingsMeta.Classes.Count;
            ValidateInput(_findings, findingsMeta.InputSize, FindingsModelName);
            ValidateOutput(_findings, LogitsName, new[] { 1, c }, FindingsModelName);
            var camDims = GetOutputDims(_findings, CamName, FindingsModelName);
            if (camDims.Length != 4 || (camDims[1] > 0 && camDims[1] != c))
                throw new InvalidDataException($"{FindingsModelName}: output 'cam' must be [1, {c}, h, w].");
            _camClassAxisLength = c;

            ValidateInput(_gate, gateMeta.InputSize, GateModelName);
            ValidateOutput(_gate, LogitsName, new[] { 1, 2 }, GateModelName);

            _findingsMeta = findingsMeta;
            _gateMeta = gateMeta;
            UnavailableReason = null;
        }

        private static void ValidateInput(InferenceSession s, int size, string name)
        {
            if (!s.InputMetadata.TryGetValue(InputName, out var meta))
                throw new InvalidDataException($"{name}: model has no input named '{InputName}'.");
            int[] expected = { 1, 3, size, size };
            CheckDims(meta.Dimensions, expected, $"{name}: input '{InputName}'");
        }

        private static void ValidateOutput(InferenceSession s, string output, int[] expected, string name)
            => CheckDims(GetOutputDims(s, output, name), expected, $"{name}: output '{output}'");

        private static int[] GetOutputDims(InferenceSession s, string output, string name)
        {
            if (!s.OutputMetadata.TryGetValue(output, out var meta))
                throw new InvalidDataException($"{name}: model has no output named '{output}'.");
            return meta.Dimensions;
        }

        // Dynamic dims (<= 0) are accepted.
        private static void CheckDims(int[] actual, int[] expected, string what)
        {
            bool ok = actual.Length == expected.Length;
            for (int i = 0; ok && i < actual.Length; i++)
                ok = actual[i] <= 0 || actual[i] == expected[i];
            if (!ok)
                throw new InvalidDataException(
                    $"{what} has shape [{string.Join(", ", actual)}], expected [{string.Join(", ", expected)}] (json and onnx disagree).");
        }

        public Task<AnalysisResult> AnalyseAsync(string imagePath, CancellationToken cancellationToken = default)
        {
            return Task.Run(() =>
            {
                try
                {
                    _lifetime.EnterReadLock();
                    try
                    {
                        return Analyse(imagePath, cancellationToken);
                    }
                    finally
                    {
                        _lifetime.ExitReadLock();
                    }
                }
                catch (OperationCanceledException)
                {
                    return ErrorResult(imagePath, CancelledMessage);
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"[FindingsEngine] Analyse failed: {ex}");
                    return ErrorResult(imagePath, "AI analysis failed: " + ex.Message);
                }
            }, CancellationToken.None); // not the caller token: a pre-cancelled token must still give a result, not an exception
        }

        private AnalysisResult ErrorResult(string imagePath, string? message) => new()
        {
            Status = AnalysisStatus.Error,
            Message = message,
            ImagePath = imagePath ?? "",
            ModelVersion = ModelVersion,
            GateModelVersion = GateModelVersion,
            AnalysedAt = DateTime.Now
        };

        /// <summary>Synchronous analysis (runs on the calling thread). Throws OperationCanceledException on cancel.</summary>
        public AnalysisResult Analyse(string imagePath, CancellationToken ct = default)
        {
            var findingsSession = _findings;
            var gateSession = _gate;
            var fm = _findingsMeta;
            var gm = _gateMeta;
            if (_disposed || findingsSession == null || gateSession == null || fm == null || gm == null)
                return ErrorResult(imagePath, UnavailableReason ?? "AI engine is not available.");

            ct.ThrowIfCancellationRequested();
            var sw = Stopwatch.StartNew();

            using var image = ImagePreprocessor.Load(imagePath, fm.Preprocess);
            if (image == null) return ErrorResult(imagePath, DecisionLogic.ReadErrorMessage);
            ct.ThrowIfCancellationRequested();

            // Quality.
            var quality = QualityChecker.Check(image.CroppedBgr, fm.Quality);
            if (!quality.IsOk)
            {
                return new AnalysisResult
                {
                    Status = AnalysisStatus.NotSuitable,
                    Message = DecisionLogic.JoinProblems(quality.Problems),
                    Quality = quality,
                    ImagePath = imagePath,
                    ModelVersion = fm.ModelVersion,
                    GateModelVersion = gm.ModelVersion,
                    InferenceMs = sw.Elapsed.TotalMilliseconds,
                    AnalysedAt = DateTime.Now
                };
            }
            ct.ThrowIfCancellationRequested();

            // Gate.
            double pEndo;
            var gateInput = ImagePreprocessor.ToTensor(image.CroppedBgr, gm.InputSize);
            using (var gateOut = gateSession.Run(new[] { NamedOnnxValue.CreateFromTensor(InputName, gateInput) }, new[] { LogitsName }))
            {
                var logits = gateOut.First().AsEnumerable<float>().ToArray();
                pEndo = DecisionLogic.EndoscopyProbability(logits, gm.Temperature);
            }
            if (!DecisionLogic.IsEndoscopy(pEndo, gm.EndoscopyThreshold))
            {
                return new AnalysisResult
                {
                    Status = AnalysisStatus.NotEndoscopy,
                    Message = DecisionLogic.NotEndoscopyMessage,
                    EndoscopyProbability = pEndo,
                    Quality = quality,
                    ImagePath = imagePath,
                    ModelVersion = fm.ModelVersion,
                    GateModelVersion = gm.ModelVersion,
                    InferenceMs = sw.Elapsed.TotalMilliseconds,
                    AnalysedAt = DateTime.Now
                };
            }
            ct.ThrowIfCancellationRequested();

            // Findings.
            float[] findingLogits;
            float[] cam;
            int camH, camW;
            var input = ImagePreprocessor.ToTensor(image.CroppedBgr, fm.InputSize);
            using (var outs = findingsSession.Run(new[] { NamedOnnxValue.CreateFromTensor(InputName, input) }, new[] { LogitsName, CamName }))
            {
                findingLogits = outs.First(o => o.Name == LogitsName).AsEnumerable<float>().ToArray();
                var camTensor = outs.First(o => o.Name == CamName).AsTensor<float>();
                var dims = camTensor.Dimensions;
                if (dims.Length != 4 || dims[1] != _camClassAxisLength)
                    throw new InvalidDataException("Unexpected 'cam' output shape.");
                camH = dims[2];
                camW = dims[3];
                cam = camTensor.ToArray();
            }
            if (findingLogits.Length != fm.Classes.Count)
                throw new InvalidDataException("Model output does not match the number of classes in its json.");

            double[] probs = DecisionLogic.Softmax(findingLogits, fm.Temperature);
            var findings = DecisionLogic.BuildFindings(probs, fm.Classes);
            var top = findings[0];
            var status = DecisionLogic.DecideStatus(top.Probability, fm.UncertainBelow);
            ct.ThrowIfCancellationRequested();

            // Heatmap for the top class.
            int topIndex = fm.Classes.FindIndex(c => c.Key == top.Key);
            int plane = camH * camW;
            var topCam = new float[plane];
            Array.Copy(cam, topIndex * plane, topCam, 0, plane);
            byte[] png = HeatmapRenderer.Render(topCam, camH, camW, image.Crop, image.Original.Width, image.Original.Height);

            return new AnalysisResult
            {
                Status = status,
                Message = DecisionLogic.MessageFor(status),
                Findings = findings,
                EndoscopyProbability = pEndo,
                Quality = quality,
                HeatmapPng = png,
                HeatmapClassKey = top.Key,
                ImagePath = imagePath,
                ModelVersion = fm.ModelVersion,
                GateModelVersion = gm.ModelVersion,
                InferenceMs = sw.Elapsed.TotalMilliseconds,
                AnalysedAt = DateTime.Now
            };
        }

        /// <summary>Raw model outputs for one image, for the golden parity test. Null when unavailable/unreadable.</summary>
        public RawModelOutputs? RunRaw(string imagePath)
        {
            _lifetime.EnterReadLock();
            try
            {
                return RunRawCore(imagePath);
            }
            finally
            {
                _lifetime.ExitReadLock();
            }
        }

        private RawModelOutputs? RunRawCore(string imagePath)
        {
            var fm = _findingsMeta; var gm = _gateMeta;
            if (!IsAvailable || fm == null || gm == null) return null;
            using var image = ImagePreprocessor.Load(imagePath, fm.Preprocess);
            if (image == null) return null;

            double pEndo;
            using (var gateOut = _gate!.Run(new[] { NamedOnnxValue.CreateFromTensor(InputName, ImagePreprocessor.ToTensor(image.CroppedBgr, gm.InputSize)) }, new[] { LogitsName }))
                pEndo = DecisionLogic.EndoscopyProbability(gateOut.First().AsEnumerable<float>().ToArray(), gm.Temperature);

            double[] probs;
            using (var outs = _findings!.Run(new[] { NamedOnnxValue.CreateFromTensor(InputName, ImagePreprocessor.ToTensor(image.CroppedBgr, fm.InputSize)) }, new[] { LogitsName }))
                probs = DecisionLogic.Softmax(outs.First().AsEnumerable<float>().ToArray(), fm.Temperature);

            return new RawModelOutputs(image.Crop, probs, pEndo);
        }

        private void DisposeSessions()
        {
            _findings?.Dispose();
            _gate?.Dispose();
            _findings = null;
            _gate = null;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            UnavailableReason = "AI engine has been shut down.";
            _lifetime.EnterWriteLock();
            try
            {
                DisposeSessions();
            }
            finally
            {
                _lifetime.ExitWriteLock();
            }
        }
    }
}
