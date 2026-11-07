using System;
using System.Collections.Generic;
using OpenCvSharp;

namespace EndoscopyApp.Services.Ai
{
    /// <summary>Image quality rules run on the cropped image before any model (AI_CONTRACT.md section 3).</summary>
    public static class QualityChecker
    {
        public const string TooSmall = "Image resolution is too low";
        public const string TooDark = "Image is too dark";
        public const string TooBright = "Image is too bright";
        public const string TooBlurry = "Image is too blurry";

        public static QualityReport Check(Mat bgrCrop, QualitySettings q)
        {
            int w = bgrCrop.Width, h = bgrCrop.Height;
            var problems = new List<string>();

            if (Math.Min(w, h) < q.MinSidePx) problems.Add(TooSmall);

            using var gray = new Mat();
            Cv2.CvtColor(bgrCrop, gray, ColorConversionCodes.BGR2GRAY);

            // Normalise the analysis scale so blur scores do not depend on the capture resolution.
            using var small = new Mat();
            int longSide = Math.Max(w, h);
            if (q.AnalysisLongSidePx > 0 && longSide > 0 && longSide != q.AnalysisLongSidePx)
            {
                double scale = (double)q.AnalysisLongSidePx / longSide;
                var size = new Size(Math.Max(1, (int)Math.Round(w * scale)), Math.Max(1, (int)Math.Round(h * scale)));
                Cv2.Resize(gray, small, size, 0, 0, InterpolationFlags.Area);
            }
            else
            {
                gray.CopyTo(small);
            }

            double brightness = Cv2.Mean(small).Val0;
            if (brightness < q.DarkMeanBelow) problems.Add(TooDark);
            else if (brightness > q.BrightMeanAbove) problems.Add(TooBright);

            using var lap = new Mat();
            Cv2.Laplacian(small, lap, MatType.CV_64F);
            Cv2.MeanStdDev(lap, out _, out Scalar std);
            double sharpness = std.Val0 * std.Val0;
            if (sharpness < q.BlurLaplacianVarBelow) problems.Add(TooBlurry);

            return new QualityReport(problems.Count == 0, brightness, sharpness, w, h, problems);
        }
    }
}
