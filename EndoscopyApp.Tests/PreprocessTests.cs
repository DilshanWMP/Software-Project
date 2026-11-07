using System.IO;
using EndoscopyApp.Services.Ai;
using OpenCvSharp;

namespace EndoscopyApp.Tests
{
    public class PreprocessTests
    {
        private static Rect Crop(Mat m) => ImagePreprocessor.FindBorderCrop(m, 20, 0.02, 0.25);

        [Fact]
        public void BlackBorders_AreCropped_InclusiveBounds()
        {
            using var img = new Mat(200, 300, MatType.CV_8UC3, Scalar.All(0));
            // Content at x 40..259, y 10..189 (inclusive).
            using (var roi = new Mat(img, new Rect(40, 10, 220, 180))) roi.SetTo(new Scalar(120, 120, 120));

            var r = Crop(img);
            Assert.Equal(new Rect(40, 10, 220, 180), r);
        }

        [Fact]
        public void NearBlackBorderNoise_BelowThreshold_IsIgnored()
        {
            using var img = new Mat(200, 300, MatType.CV_8UC3, new Scalar(20, 20, 20)); // gray == 20 is NOT > 20
            using (var roi = new Mat(img, new Rect(50, 20, 200, 160))) roi.SetTo(new Scalar(100, 150, 200));
            Assert.Equal(new Rect(50, 20, 200, 160), Crop(img));
        }

        [Fact]
        public void SparseBrightPixels_BelowMinFraction_DoNotCountAsContent()
        {
            using var img = new Mat(100, 100, MatType.CV_8UC3, Scalar.All(0));
            using (var roi = new Mat(img, new Rect(10, 10, 80, 80))) roi.SetTo(Scalar.All(200));
            // Row 2 has exactly 2 bright pixels: 2/100 = 0.02 is not > 0.02, so it is not content.
            img.Set(2, 50, new Vec3b(255, 255, 255));
            img.Set(2, 51, new Vec3b(255, 255, 255));
            Assert.Equal(new Rect(10, 10, 80, 80), Crop(img));
        }

        [Fact]
        public void NoBorder_KeepsFullImage()
        {
            using var img = TestHelpers.Textured(320, 240);
            Assert.Equal(new Rect(0, 0, 320, 240), Crop(img));
        }

        [Fact]
        public void AllBlack_DoesNotCrop()
        {
            using var img = new Mat(120, 160, MatType.CV_8UC3, Scalar.All(0));
            Assert.Equal(new Rect(0, 0, 160, 120), Crop(img));
        }

        [Fact]
        public void TinyContent_BelowMinKeep_DoesNotCrop()
        {
            using var img = new Mat(200, 200, MatType.CV_8UC3, Scalar.All(0));
            using (var roi = new Mat(img, new Rect(90, 90, 20, 20))) roi.SetTo(Scalar.All(200)); // 10% of each side
            Assert.Equal(new Rect(0, 0, 200, 200), Crop(img));
        }

        [Fact]
        public void ToTensor_IsRgbChw_ScaledTo01()
        {
            using var img = new Mat(10, 10, MatType.CV_8UC3, new Scalar(255, 0, 51)); // B=255 G=0 R=51
            var t = ImagePreprocessor.ToTensor(img, 4);
            Assert.Equal(new[] { 1, 3, 4, 4 }, t.Dimensions.ToArray());
            Assert.Equal(0.2f, t[0, 0, 0, 0], 5); // R
            Assert.Equal(0f, t[0, 1, 3, 3], 5);   // G
            Assert.Equal(1f, t[0, 2, 2, 1], 5);   // B
        }

        [Fact]
        public void ToTensor_MatchesManualResizeAndPerPixelRead()
        {
            using var img = TestHelpers.Textured(37, 23, seed: 7);
            var t = ImagePreprocessor.ToTensor(img, 16);

            using var rgb = new Mat();
            Cv2.CvtColor(img, rgb, ColorConversionCodes.BGR2RGB);
            using var resized = new Mat();
            Cv2.Resize(rgb, resized, new Size(16, 16), 0, 0, InterpolationFlags.Linear);
            for (int y = 0; y < 16; y++)
                for (int x = 0; x < 16; x++)
                {
                    var px = resized.At<Vec3b>(y, x);
                    for (int c = 0; c < 3; c++)
                        Assert.Equal(px[c] / 255f, t[0, c, y, x], 6);
                }
        }

        [Fact]
        public void Decode_HandlesUnicodeAndSpacesInPath()
        {
            string dir = Path.Combine(TestHelpers.NewTempDir(), "pätient ශ්‍රී 1");
            Directory.CreateDirectory(dir);
            string path = Path.Combine(dir, "IMG test é.jpg");
            using (var img = TestHelpers.Textured(64, 48))
            {
                Cv2.ImEncode(".jpg", img, out byte[] bytes);
                File.WriteAllBytes(path, bytes);
            }
            using var decoded = ImagePreprocessor.Decode(path);
            Assert.NotNull(decoded);
            Assert.Equal(64, decoded!.Width);
            Assert.Equal(48, decoded.Height);
        }

        [Fact]
        public void Decode_ReturnsNull_ForMissingOrGarbage()
        {
            string dir = TestHelpers.NewTempDir();
            Assert.Null(ImagePreprocessor.Decode(Path.Combine(dir, "nope.jpg")));
            string junk = Path.Combine(dir, "junk.jpg");
            File.WriteAllText(junk, "not an image");
            Assert.Null(ImagePreprocessor.Decode(junk));
        }
    }
}
