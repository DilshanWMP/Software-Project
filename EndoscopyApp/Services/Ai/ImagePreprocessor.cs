using System;
using System.IO;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace EndoscopyApp.Services.Ai
{
    /// <summary>A decoded image plus its border-crop rectangle. Dispose to free the native buffers.</summary>
    public sealed class PreparedImage : IDisposable
    {
        public PreparedImage(Mat original, Rect crop)
        {
            Original = original;
            Crop = crop;
            CroppedBgr = new Mat(original, crop); // view, no copy
        }

        /// <summary>The full decoded image (BGR, 8-bit).</summary>
        public Mat Original { get; }

        /// <summary>Crop rectangle inside Original (x0 = X, y0 = Y, x1 = X + Width - 1, y1 = Y + Height - 1).</summary>
        public Rect Crop { get; }

        /// <summary>View of Original restricted to Crop (BGR).</summary>
        public Mat CroppedBgr { get; }

        public void Dispose()
        {
            CroppedBgr.Dispose();
            Original.Dispose();
        }
    }

    /// <summary>Decode, border crop and tensor conversion exactly as in AI_CONTRACT.md section 2 (ml/src/preprocess.py).</summary>
    public static class ImagePreprocessor
    {
        // EXIF orientation is ignored: WPF shows the stored pixels, so the heatmap must align with them (contract section 2).
        public const ImreadModes DecodeFlags = ImreadModes.Color | ImreadModes.IgnoreOrientation;

        /// <summary>Decodes a file as BGR 8-bit. Returns null if missing or undecodable.</summary>
        public static Mat? Decode(string path)
        {
            if (string.IsNullOrWhiteSpace(path) || !File.Exists(path)) return null;

            Mat? mat = null;
            try
            {
                mat = Cv2.ImRead(path, DecodeFlags);
            }
            catch
            {
                mat?.Dispose();
                mat = null;
            }

            if (mat == null || mat.Empty())
            {
                // ImRead cannot open some unicode paths on Windows: decode from bytes instead.
                mat?.Dispose();
                mat = null;
                try
                {
                    byte[] bytes = File.ReadAllBytes(path);
                    if (bytes.Length == 0) return null;
                    mat = Cv2.ImDecode(bytes, DecodeFlags);
                }
                catch
                {
                    mat?.Dispose();
                    return null;
                }
            }

            if (mat == null || mat.Empty() || mat.Type() != MatType.CV_8UC3)
            {
                mat?.Dispose();
                return null;
            }
            return mat;
        }

        /// <summary>Decode + border crop. Null when the file cannot be read.</summary>
        public static PreparedImage? Load(string path, PreprocessSettings settings)
        {
            Mat? bgr = Decode(path);
            if (bgr == null) return null;
            Rect crop = settings.BorderCrop
                ? FindBorderCrop(bgr, settings.BorderGrayThreshold, settings.BorderMinFraction, settings.BorderMinKeep)
                : new Rect(0, 0, bgr.Width, bgr.Height);
            return new PreparedImage(bgr, crop);
        }

        /// <summary>
        /// Border crop: rows/cols whose fraction of pixels with gray > threshold is > minFraction are content.
        /// Crop to the first..last content row/col (inclusive). Falls back to the full image when nothing
        /// qualifies or the crop would be smaller than minKeep of the width or height.
        /// </summary>
        public static Rect FindBorderCrop(Mat bgr, int grayThreshold, double minFraction, double minKeep)
        {
            int w = bgr.Width, h = bgr.Height;
            var full = new Rect(0, 0, w, h);
            if (w == 0 || h == 0) return full;

            using var gray = new Mat();
            Cv2.CvtColor(bgr, gray, ColorConversionCodes.BGR2GRAY);
            using var mask = new Mat();
            // 1 where gray > threshold, else 0.
            Cv2.Threshold(gray, mask, grayThreshold, 1, ThresholdTypes.Binary);

            using var rowCounts = new Mat();
            using var colCounts = new Mat();
            Cv2.Reduce(mask, rowCounts, ReduceDimension.Column, ReduceTypes.Sum, MatType.CV_32S); // h x 1
            Cv2.Reduce(mask, colCounts, ReduceDimension.Row, ReduceTypes.Sum, MatType.CV_32S);    // 1 x w

            var rows = new int[h];
            var cols = new int[w];
            Marshal.Copy(rowCounts.Data, rows, 0, h);
            Marshal.Copy(colCounts.Data, cols, 0, w);

            int y0 = -1, y1 = -1, x0 = -1, x1 = -1;
            for (int y = 0; y < h; y++)
            {
                if ((double)rows[y] / w > minFraction)
                {
                    if (y0 < 0) y0 = y;
                    y1 = y;
                }
            }
            for (int x = 0; x < w; x++)
            {
                if ((double)cols[x] / h > minFraction)
                {
                    if (x0 < 0) x0 = x;
                    x1 = x;
                }
            }

            if (y0 < 0 || x0 < 0) return full;
            int cw = x1 - x0 + 1, ch = y1 - y0 + 1;
            if (cw < minKeep * w || ch < minKeep * h) return full;
            return new Rect(x0, y0, cw, ch);
        }

        /// <summary>BGR crop -> RGB -> bilinear resize to size x size -> float32 /255 CHW with batch dim [1,3,S,S].</summary>
        public static DenseTensor<float> ToTensor(Mat bgrCrop, int size)
        {
            using var rgb = new Mat();
            Cv2.CvtColor(bgrCrop, rgb, ColorConversionCodes.BGR2RGB);
            using var resized = new Mat();
            Cv2.Resize(rgb, resized, new Size(size, size), 0, 0, InterpolationFlags.Linear);
            using var f32 = new Mat();
            resized.ConvertTo(f32, MatType.CV_32FC3, 1.0 / 255.0);

            int plane = size * size;
            var buffer = new float[3 * plane];
            Mat[] channels = Cv2.Split(f32);
            try
            {
                for (int c = 0; c < 3; c++)
                {
                    Mat ch = channels[c].IsContinuous() ? channels[c] : channels[c].Clone();
                    Marshal.Copy(ch.Data, buffer, c * plane, plane);
                    if (!ReferenceEquals(ch, channels[c])) ch.Dispose();
                }
            }
            finally
            {
                foreach (var m in channels) m.Dispose();
            }
            return new DenseTensor<float>(buffer, new[] { 1, 3, size, size });
        }
    }
}
