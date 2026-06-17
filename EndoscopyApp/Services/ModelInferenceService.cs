using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using OpenCvSharp;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace EndoscopyApp.Services
{
    public class ModelInferenceService : IDisposable
    {
        private InferenceSession? _session;
        private readonly string _modelPath;
        private const int INPUT_SIZE = 640; // YOLOv8 standard input size
        private const float CONFIDENCE_THRESHOLD = 0.5f;

        public ModelInferenceService(string modelPath = "")
        {
            _modelPath = modelPath;
            if (string.IsNullOrEmpty(_modelPath))
            {
                // Navigate from bin\Debug\net9.0-windows\ to solution root\CNN model\best.onnx
                string baseDir = AppDomain.CurrentDomain.BaseDirectory;
                string relativePath = System.IO.Path.Combine(baseDir, "..", "..", "..", "..", "CNN model", "best.onnx");
                _modelPath = System.IO.Path.GetFullPath(relativePath);
            }
        }

        public async Task Initialize()
        {
            await Task.Run(() =>
            {
                try
                {
                    if (!System.IO.File.Exists(_modelPath))
                    {
                        throw new FileNotFoundException($"Model file not found: {_modelPath}");
                    }

                    var sessionOptions = new SessionOptions();
                    sessionOptions.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;
                    _session = new InferenceSession(_modelPath, sessionOptions);

                    Console.WriteLine($"Model loaded successfully: {_modelPath}");
                }
                catch (Exception ex)
                {
                    throw new Exception($"Failed to initialize model: {ex.Message}", ex);
                }
            });
        }

        public bool IsInitialized => _session != null;

        public async Task<List<DetectionResult>> RunInference(Mat frame)
        {
            if (_session == null)
                throw new InvalidOperationException("Model not initialized. Call Initialize() first.");

            return await Task.Run(() =>
            {
                try
                {
                    // Preprocess frame
                    var (processedFrame, scale) = PreprocessFrame(frame);

                    // Prepare input tensor
                    var inputMeta = _session.InputMetadata.Values.First();
                    var inputName = _session.InputNames[0];
                    var inputTensor = ProcessInputImage(processedFrame);

                    // Run inference
                    var inputs = new List<NamedOnnxValue>
                    {
                        NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
                    };

                    using var results = _session.Run(inputs);
                    var output = results.First().AsEnumerable<float>().ToArray();

                    // Parse results and postprocess
                    var detections = ParseOutput(output, scale, frame.Size());

                    return detections;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Inference error: {ex.Message}");
                    return new List<DetectionResult>();
                }
            });
        }

        private (Mat, float) PreprocessFrame(Mat frame)
        {
            Mat resized = new Mat();
            Mat padded = new Mat();

            // Calculate scale to maintain aspect ratio
            float scale = Math.Min((float)INPUT_SIZE / frame.Width, (float)INPUT_SIZE / frame.Height);
            int newWidth = (int)(frame.Width * scale);
            int newHeight = (int)(frame.Height * scale);

            // Resize with aspect ratio
            Cv2.Resize(frame, resized, new Size(newWidth, newHeight), interpolation: InterpolationFlags.Linear);

            // Create padded image
            int padX = (INPUT_SIZE - newWidth) / 2;
            int padY = (INPUT_SIZE - newHeight) / 2;

            Cv2.CopyMakeBorder(resized, padded, padY, INPUT_SIZE - newHeight - padY,
                padX, INPUT_SIZE - newWidth - padX, BorderTypes.Constant, new Scalar(114, 114, 114));

            resized.Dispose();
            return (padded, scale);
        }

        private DenseTensor<float> ProcessInputImage(Mat image)
        {
            // Convert BGR to RGB and normalize
            var tensor = new DenseTensor<float>(new[] { 1, 3, INPUT_SIZE, INPUT_SIZE });
            
            image.ConvertTo(image, MatType.CV_32F, 1.0 / 255.0);

            // Split channels
            var channels = image.Split();

            // Use unsafe pointers for fast pixel access
            unsafe
            {
                for (int c = 0; c < 3; c++)
                {
                    Mat channel = channels[2 - c]; // Reverse BGR to RGB
                    float* ptr = (float*)channel.DataPointer;
                    int stepInFloats = (int)(channel.Step() >> 2); // Step in bytes / 4 = step in floats

                    for (int h = 0; h < INPUT_SIZE; h++)
                    {
                        for (int w = 0; w < INPUT_SIZE; w++)
                        {
                            tensor[0, c, h, w] = ptr[h * stepInFloats + w];
                        }
                    }
                }
            }

            foreach (var channel in channels)
                channel.Dispose();

            return tensor;
        }

        private List<DetectionResult> ParseOutput(float[] output, float scale, Size originalSize)
        {
            var detections = new List<DetectionResult>();

            // YOLOv8 output format: [N, 84, 8400] or similar
            // Each detection: [x, y, w, h, confidence, class_scores...]
            
            int stride = 84; // 4 (bbox) + 1 (confidence) + 79 (classes) or 4+1+80 for COCO
            int numDetections = output.Length / stride;

            for (int i = 0; i < numDetections; i++)
            {
                int idx = i * stride;
                
                if (idx + 4 >= output.Length) break;

                float confidence = output[idx + 4];

                if (confidence < CONFIDENCE_THRESHOLD) continue;

                // Get class with highest score
                int classId = 0;
                float maxScore = output[idx + 5];

                for (int j = 5; j < Math.Min(idx + stride, output.Length); j++)
                {
                    if (output[j] > maxScore)
                    {
                        maxScore = output[j];
                        classId = j - 5;
                    }
                }

                // Parse bbox
                float x = output[idx];
                float y = output[idx + 1];
                float w = output[idx + 2];
                float h = output[idx + 3];

                // Inverse the padding and scaling
                int padX = (INPUT_SIZE - (int)(originalSize.Width * scale)) / 2;
                int padY = (INPUT_SIZE - (int)(originalSize.Height * scale)) / 2;

                float bboxX = ((x - padX) / scale) / originalSize.Width;
                float bboxY = ((y - padY) / scale) / originalSize.Height;
                float bboxW = (w / scale) / originalSize.Width;
                float bboxH = (h / scale) / originalSize.Height;

                detections.Add(new DetectionResult
                {
                    X = Math.Max(0, bboxX),
                    Y = Math.Max(0, bboxY),
                    Width = Math.Min(1 - bboxX, bboxW),
                    Height = Math.Min(1 - bboxY, bboxH),
                    Confidence = confidence,
                    ClassId = classId,
                    ClassScore = maxScore * confidence
                });
            }

            return detections.OrderByDescending(d => d.Confidence).ToList();
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }

    public class DetectionResult
    {
        public float X { get; set; }           // Normalized X (0-1)
        public float Y { get; set; }           // Normalized Y (0-1)
        public float Width { get; set; }       // Normalized width
        public float Height { get; set; }      // Normalized height
        public float Confidence { get; set; }  // Detection confidence
        public int ClassId { get; set; }       // Class ID
        public float ClassScore { get; set; }  // Class score
    }
}
