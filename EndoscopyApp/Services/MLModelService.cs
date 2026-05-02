using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace EndoscopyApp.Services
{
    public class MLModelService : IDisposable
    {
        private InferenceSession? _session;
        private int _inputHeight = 640;
        private int _inputWidth = 640;
        private string _inputName = "";

        // Default classes based on standard endoscopy findings
        private readonly Dictionary<int, string> _classes = new()
        {
            { 0, "Healthy" },
            { 1, "Polyp" },
            { 2, "Ulcer" },
            { 3, "Bleeding" },
            { 4, "Inflammation" }
        };

        public bool IsLoaded => _session != null;

        public MLModelService()
        {
            string absoluteModelPath = @"d:\5th sem\Software group project\Software-Project\CNN model\best.onnx";

            if (File.Exists(absoluteModelPath))
            {
                try
                {
                    _session = new InferenceSession(absoluteModelPath);
                    var inputMeta = _session.InputMetadata.First();
                    _inputName = inputMeta.Key;
                    
                    // Attempt to dynamically get shape if specified
                    if (inputMeta.Value.Dimensions.Length == 4)
                    {
                        // typically [batch, channels, height, width]
                        var h = inputMeta.Value.Dimensions[2];
                        var w = inputMeta.Value.Dimensions[3];
                        if (h > 0) _inputHeight = h;
                        if (w > 0) _inputWidth = w;
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine("Failed to load model: " + ex.Message);
                }
            }
        }

        public (string prediction, float confidence)? AnalyseImage(string imagePath)
        {
            if (_session == null || !File.Exists(imagePath)) return null;

            try
            {
                using var image = new Mat(imagePath, ImreadModes.Color);
                if (image.Empty()) return null;

                using var resized = new Mat();
                Cv2.Resize(image, resized, new Size(_inputWidth, _inputHeight));

                // Convert to RGB (OpenCV uses BGR by default)
                using var rgb = new Mat();
                Cv2.CvtColor(resized, rgb, ColorConversionCodes.BGR2RGB);

                // Convert to Float Tensor [1, 3, H, W]
                var tensor = new DenseTensor<float>(new[] { 1, 3, _inputHeight, _inputWidth });

                // YOLO models usually require pixels normalized to [0, 1]
                for (int y = 0; y < _inputHeight; y++)
                {
                    for (int x = 0; x < _inputWidth; x++)
                    {
                        var vec = rgb.At<Vec3b>(y, x);
                        tensor[0, 0, y, x] = vec.Item0 / 255f; // R
                        tensor[0, 1, y, x] = vec.Item1 / 255f; // G
                        tensor[0, 2, y, x] = vec.Item2 / 255f; // B
                    }
                }

                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor(_inputName, tensor)
                };

                using var results = _session.Run(inputs);
                var outputTensor = results.First().AsTensor<float>();
                
                // YOLO Classification [1, num_classes]
                if (outputTensor.Dimensions.Length == 2)
                {
                    int bestClass = -1;
                    float bestScore = -1f;

                    for (int i = 0; i < outputTensor.Dimensions[1]; i++)
                    {
                        if (outputTensor[0, i] > bestScore)
                        {
                            bestScore = outputTensor[0, i];
                            bestClass = i;
                        }
                    }

                    string className = _classes.ContainsKey(bestClass) ? _classes[bestClass] : $"Class {bestClass}";
                    return (className, bestScore);
                }
                // YOLO Detection [1, 4 + classes, 8400]
                else if (outputTensor.Dimensions.Length == 3)
                {
                    int numClasses = outputTensor.Dimensions[1] - 4;
                    int numAnchors = outputTensor.Dimensions[2];
                    
                    // Keep track of the highest score for EACH class
                    var classScores = new float[numClasses];

                    for (int c = 0; c < numClasses; c++)
                    {
                        float maxScoreForClass = 0f;
                        for (int a = 0; a < numAnchors; a++)
                        {
                            float score = outputTensor[0, 4 + c, a];
                            if (score > maxScoreForClass)
                            {
                                maxScoreForClass = score;
                            }
                        }
                        classScores[c] = maxScoreForClass;
                    }

                    // Collect all classes that cross a reasonable threshold (e.g. 0.3)
                    float threshold = 0.3f;
                    var detectedIllnesses = new List<string>();
                    
                    for (int c = 0; c < numClasses; c++)
                    {
                        if (classScores[c] > threshold)
                        {
                            string cName = _classes.ContainsKey(c) ? _classes[c] : $"Class {c}";
                            detectedIllnesses.Add($"{cName} ({(classScores[c] * 100).ToString("0.0")}%)");
                        }
                    }

                    if (detectedIllnesses.Count > 0)
                    {
                        // Return the combined string of all detected illnesses
                        return (string.Join(", ", detectedIllnesses), classScores.Max());
                    }
                    else
                    {
                        return ("Healthy (No abnormalities detected)", 1.0f);
                    }
                }
                
                return ("Unknown Model Format", 0f);
            }
            catch (Exception ex)
            {
                Console.WriteLine("Inference Error: " + ex.Message);
                return null;
            }
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}
