using System;
using System.Threading;
using System.Threading.Tasks;
using OpenCvSharp;
using OpenCvSharp.WpfExtensions;
using System.Windows.Media.Imaging;

namespace EndoscopyApp.Services
{
    public class VideoCaptureService : IDisposable
    {
        private VideoCapture? _capture;
        private CancellationTokenSource? _cts;
        private Task? _captureTask;
        
        public event EventHandler<WriteableBitmap>? FrameReady;

        public bool IsRunning { get; private set; }

        public void Start(int deviceIndex = 0)
        {
            if (IsRunning) return;

            _capture = new VideoCapture(deviceIndex);
            if (!_capture.IsOpened())
            {
                throw new Exception("Could not open video device.");
            }

            IsRunning = true;
            _cts = new CancellationTokenSource();
            _captureTask = Task.Run(() => CaptureLoop(_cts.Token));
        }

        public void Stop()
        {
            if (!IsRunning) return;

            IsRunning = false;
            _cts?.Cancel();
            try { _captureTask?.Wait(1000); } catch { /* Ignore cancellation */ }
            
            _capture?.Release();
            _capture?.Dispose();
            _capture = null;
        }

        private VideoWriter? _writer;
        public bool IsRecording { get; private set; }

        public void StartRecording(string filePath)
        {
            if (IsRecording || _capture == null) return;

            // Define codec (MJPG is common/safe, or XVID)
            // fourcc: 'M', 'J', 'P', 'G' -> .avi
            // fourcc: 'H', '2', '6', '4' -> .mp4 (requires openh264 on windows sometimes)
            // Let's use MJPG for simplicity and .avi for now to ensure compatibility without extra dlls
            var fourcc = VideoWriter.FourCC('M', 'J', 'P', 'G');
            var fps = 30.0;
            var size = new Size(_capture.FrameWidth, _capture.FrameHeight);
            
            _writer = new VideoWriter(filePath, fourcc, fps, size);
            
            if (_writer.IsOpened())
            {
                IsRecording = true;
            }
        }

        public void StopRecording()
        {
            if (!IsRecording) return;
            
            IsRecording = false;
            _writer?.Release();
            _writer?.Dispose();
            _writer = null;
        }

        public Mat CaptureSnapshot()
        {
             if (_capture != null && _capture.IsOpened())
             {
                 var frame = new Mat();
                 _capture.Read(frame);
                 return frame;
             }
             return new Mat();
        }
        
        // Revised CaptureLoop to handle recording
        private async Task CaptureLoop(CancellationToken token)
        {
            using var frame = new Mat();
            while (!token.IsCancellationRequested && IsRunning)
            {
                if (_capture != null && _capture.Read(frame) && !frame.Empty())
                {
                    // Recording
                    if (IsRecording && _writer != null && _writer.IsOpened())
                    {
                        _writer.Write(frame);
                    }

                    // UI Update
                    try 
                    {
                        var bitmap = frame.ToWriteableBitmap();
                        bitmap.Freeze();
                        FrameReady?.Invoke(this, bitmap);
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Debug.WriteLine($"Error converting frame: {ex.Message}");
                    }
                }
                await Task.Delay(33, token);
            }
        }

        public void Dispose()
        {
            Stop();
        }
    }
}
