using System;
using System.Runtime.InteropServices;
using OpenCvSharp;

namespace EndoscopyApp.Services.Ai
{
    /// <summary>Class activation map -> transparent RGBA PNG the size of the original image (AI_CONTRACT.md section 4 step 7).</summary>
    public static class HeatmapRenderer
    {
        public const double MinHeat = 0.2;
        public const double MaxAlpha = 0.55;

        /// <summary>
        /// cam: h*w values (row-major) for one class. ReLU, replace the top-left cell with the max of its 3 neighbours,
        /// divide by max, resize to the crop, JET colours, alpha = heat * 0.55 * 255 (0 where heat &lt; 0.2),
        /// placed at the crop origin on a transparent canvas.
        /// </summary>
        public static byte[] Render(float[] cam, int camHeight, int camWidth, Rect crop, int originalWidth, int originalHeight)
        {
            if (cam.Length != camHeight * camWidth) throw new ArgumentException("cam length does not match its size.");

            // ReLU.
            var relu = new float[cam.Length];
            for (int i = 0; i < cam.Length; i++) relu[i] = cam[i] > 0f ? cam[i] : 0f;

            // gi_findings_v1 puts a large padding artefact in the top-left cell on ~90 % of images, which would
            // otherwise take the whole colour range. Its neighbours still carry any real heat near that corner.
            if (camHeight >= 2 && camWidth >= 2)
                relu[0] = Math.Max(relu[1], Math.Max(relu[camWidth], relu[camWidth + 1]));

            float max = 0f;
            foreach (float v in relu) if (v > max) max = v;
            if (max > 0f)
            {
                for (int i = 0; i < relu.Length; i++) relu[i] /= max;
            }
            else
            {
                Array.Clear(relu);
            }

            using var small = new Mat(camHeight, camWidth, MatType.CV_32FC1);
            Marshal.Copy(relu, 0, small.Data, relu.Length);

            using var heat = new Mat();
            Cv2.Resize(small, heat, new Size(crop.Width, crop.Height), 0, 0, InterpolationFlags.Linear);

            // Colours.
            using var heat8 = new Mat();
            heat.ConvertTo(heat8, MatType.CV_8UC1, 255.0);
            using var colour = new Mat();
            Cv2.ApplyColorMap(heat8, colour, ColormapTypes.Jet); // BGR

            // Alpha.
            using var alpha = new Mat();
            heat.ConvertTo(alpha, MatType.CV_8UC1, MaxAlpha * 255.0);
            using (var low = new Mat())
            {
                Cv2.Compare(heat, new Scalar(MinHeat), low, CmpType.LT);
                alpha.SetTo(new Scalar(0), low);
            }

            Mat[] bgr = Cv2.Split(colour);
            using var bgra = new Mat();
            try
            {
                Cv2.Merge(new[] { bgr[0], bgr[1], bgr[2], alpha }, bgra);
            }
            finally
            {
                foreach (var m in bgr) m.Dispose();
            }

            using var canvas = new Mat(originalHeight, originalWidth, MatType.CV_8UC4, Scalar.All(0));
            using (var roi = new Mat(canvas, crop))
            {
                bgra.CopyTo(roi);
            }

            // OpenCV stores BGRA; the PNG encoder writes it as a normal RGBA PNG.
            Cv2.ImEncode(".png", canvas, out byte[] png);
            return png;
        }
    }
}
