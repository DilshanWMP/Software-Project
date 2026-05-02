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
        private DateTime _lastWriteTime;

        public event EventHandler<WriteableBitmap>? FrameReady;

        public bool IsRunning { get; private set; }

        public async Task Start(int deviceIndex = 0)
        {
            if (IsRunning) return;

            try
            {
                // Try DSHOW first as it handles high res better on Windows
                _capture = new VideoCapture(deviceIndex, VideoCaptureAPIs.DSHOW);

                if (!_capture.IsOpened())
                {
                    _capture = new VideoCapture(deviceIndex);
                }

                if (!_capture.IsOpened())
                {
                    throw new Exception("Camera could not be opened.");
                }

                // Try to set high resolution if supported
                // We don't throw if these fail, as some cameras are picky
                try
                {
                    _capture.Set(VideoCaptureProperties.FourCC, VideoWriter.FourCC('M', 'J', 'P', 'G'));
                    _capture.Set(VideoCaptureProperties.FrameWidth, 1920);
                    _capture.Set(VideoCaptureProperties.FrameHeight, 1080);
                }
                catch { /* Ignore setting failures */ }

                // Test if we can actually read a frame
                // Wait a moment for the camera buffer to warm up
                await Task.Delay(500);
                using var testFrame = new Mat();
                bool canRead = _capture.Read(testFrame);

                if (!canRead || testFrame.Empty())
                {
                    // If DSHOW + HighRes failed to produce a frame, try one more time with default settings
                    _capture.Release();
                    _capture = new VideoCapture(deviceIndex);
                    // One more small delay for the fallback
                    await Task.Delay(500);
                }

                IsRunning = true;
                _cts = new CancellationTokenSource();
                _captureTask = Task.Run(() => CaptureLoop(_cts.Token));
            }
            catch (Exception ex)
            {
                IsRunning = false;
                _capture?.Release();
                _capture = null;
                throw new Exception($"Failed to start camera: {ex.Message}");
            }
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
        private readonly object _writerLock = new object();
        public bool IsRecording { get; private set; }

        private double _recordedFps = 20.0;

        public void StartRecording(string filePath)
        {
            if (IsRecording || _capture == null) return;

            // Use a stable 20 FPS for recording to balance quality and file size
            _recordedFps = 20.0;
            var fourcc = VideoWriter.FourCC('M', 'J', 'P', 'G');
            var size = new Size(_capture.FrameWidth, _capture.FrameHeight);

            var writer = new VideoWriter(filePath, fourcc, _recordedFps, size);

            if (writer.IsOpened())
            {
                lock (_writerLock)
                {
                    _writer = writer;
                    IsRecording = true;
                    _lastWriteTime = DateTime.Now;
                }
            }
        }

        public void StopRecording()
        {
            if (!IsRecording) return;

            IsRecording = false;
            
            lock (_writerLock)
            {
                _writer?.Release();
                _writer?.Dispose();
                _writer = null;
            }
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

        private bool _isProcessingFrame = false;

        private async Task CaptureLoop(CancellationToken token)
        {
            using var frame = new Mat();
            while (!token.IsCancellationRequested && IsRunning)
            {
                if (_capture != null && _capture.Read(frame) && !frame.Empty())
                {
                    // Recording - ensure we write at the correct intervals to maintain real-time speed
                    if (IsRecording)
                    {
                        lock (_writerLock)
                        {
                            if (IsRecording && _writer != null && _writer.IsOpened())
                            {
                                var now = DateTime.Now;
                                var elapsed = (now - _lastWriteTime).TotalMilliseconds;
                                var frameInterval = 1000.0 / _recordedFps;

                                // If the camera is slow, we "stuff" or duplicate frames to fill the time gap.
                                // This prevents the video from playing back fast.
                                while (elapsed >= frameInterval)
                                {
                                    _writer.Write(frame);
                                    elapsed -= frameInterval;
                                    _lastWriteTime = _lastWriteTime.AddMilliseconds(frameInterval);
                                }
                            }
                        }
                    }

                    // UI Update - skip if the last frame is still being processed by the UI
                    if (!_isProcessingFrame)
                    {
                        try
                        {
                            _isProcessingFrame = true;
                            var bitmap = frame.ToWriteableBitmap();
                            bitmap.Freeze();
                            FrameReady?.Invoke(this, bitmap);
                            _isProcessingFrame = false;
                        }
                        catch (Exception ex)
                        {
                            System.Diagnostics.Debug.WriteLine($"Error converting frame: {ex.Message}");
                            _isProcessingFrame = false;
                        }
                    }
                }

                // Small delay to prevent 100% CPU usage but fast enough for 30-60fps
                await Task.Delay(5, token);
            }
        }

        public void Dispose()
        {
            Stop();
        }
    }
}
