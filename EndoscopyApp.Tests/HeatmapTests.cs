using EndoscopyApp.Services.Ai;
using OpenCvSharp;

namespace EndoscopyApp.Tests
{
    public class HeatmapTests
    {
        private static Mat Decode(byte[] png) => Cv2.ImDecode(png, ImreadModes.Unchanged);

        [Fact]
        public void Png_IsOriginalSize_Rgba_AndTransparentOutsideCropAndBelowCutoff()
        {
            // 4x4 cam: one hot cell top-right, rest negative (ReLU -> 0).
            var cam = new float[16];
            for (int i = 0; i < cam.Length; i++) cam[i] = -1f;
            cam[3] = 5f;
            var crop = new Rect(50, 20, 200, 160);

            byte[] png = HeatmapRenderer.Render(cam, 4, 4, crop, 300, 200);
            Assert.Equal(new byte[] { 0x89, 0x50, 0x4E, 0x47 }, png[..4]);

            using var m = Decode(png);
            Assert.Equal(300, m.Width);
            Assert.Equal(200, m.Height);
            Assert.Equal(4, m.Channels());

            // Outside the crop: fully transparent.
            Assert.Equal(0, m.At<Vec4b>(5, 5)[3]);
            Assert.Equal(0, m.At<Vec4b>(199, 299)[3]);
            // Crop top-right = hottest: alpha ~ 0.55 * 255, colour red-ish in JET (BGR: R high).
            var hot = m.At<Vec4b>(20, 249);
            Assert.InRange(hot[3], 135, 141);
            Assert.True(hot[2] > 100);
            // Crop bottom-left: heat 0 (< 0.2) -> transparent.
            Assert.Equal(0, m.At<Vec4b>(179, 50)[3]);
        }

        [Fact]
        public void TopLeftSpike_IsReplacedByItsNeighbours()
        {
            // 3x3 cam: a big artefact in the top-left cell, the real heat in the centre.
            var cam = new float[] { 100f, 0f, 0f, 0f, 4f, 0f, 0f, 0f, 0f };
            byte[] png = HeatmapRenderer.Render(cam, 3, 3, new Rect(0, 0, 3, 3), 3, 3);
            using var m = Decode(png);
            Assert.Equal(140, m.At<Vec4b>(1, 1)[3]); // centre is now the max -> full alpha
            Assert.Equal(140, m.At<Vec4b>(0, 0)[3]); // corner takes the centre's value, not the spike
            Assert.Equal(0, m.At<Vec4b>(0, 2)[3]);
        }

        [Fact]
        public void AlphaIsZero_WhereHeatBelowCutoff_AndScaledAbove()
        {
            // 1x2 cam: 0.1 and 1.0 after normalisation; resized to 1x2 so values stay exact.
            byte[] png = HeatmapRenderer.Render(new[] { 0.1f, 1f }, 1, 2, new Rect(0, 0, 2, 1), 2, 1);
            using var m = Decode(png);
            Assert.Equal(0, m.At<Vec4b>(0, 0)[3]);
            Assert.Equal(140, m.At<Vec4b>(0, 1)[3]); // round(1.0 * 0.55 * 255) = 140
        }

        [Fact]
        public void AllNonPositiveCam_GivesFullyTransparentImage()
        {
            byte[] png = HeatmapRenderer.Render(new float[] { -1f, 0f, -3f, 0f }, 2, 2, new Rect(0, 0, 10, 10), 10, 10);
            using var m = Decode(png);
            Cv2.MinMaxLoc(m.ExtractChannel(3), out _, out double maxAlpha);
            Assert.Equal(0, maxAlpha);
        }
    }
}
