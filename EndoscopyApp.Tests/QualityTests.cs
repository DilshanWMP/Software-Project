using EndoscopyApp.Services.Ai;
using OpenCvSharp;

namespace EndoscopyApp.Tests
{
    public class QualityTests
    {
        private static readonly QualitySettings Q = new()
        {
            MinSidePx = 128, DarkMeanBelow = 25, BrightMeanAbove = 235, BlurLaplacianVarBelow = 15.0, AnalysisLongSidePx = 512
        };

        [Fact]
        public void TexturedImage_Passes()
        {
            using var img = TestHelpers.Textured(640, 480);
            var r = QualityChecker.Check(img, Q);
            Assert.True(r.IsOk, string.Join(", ", r.Problems));
            Assert.Equal(640, r.Width);
            Assert.Equal(480, r.Height);
            Assert.InRange(r.Brightness, 100, 160);
        }

        [Fact]
        public void Dark_IsFlagged()
        {
            using var img = new Mat(480, 640, MatType.CV_8UC3);
            Cv2.Randu(img, Scalar.All(0), Scalar.All(30)); // mean ~15, textured
            var r = QualityChecker.Check(img, Q);
            Assert.False(r.IsOk);
            Assert.Contains(QualityChecker.TooDark, r.Problems);
        }

        [Fact]
        public void Bright_IsFlagged()
        {
            using var img = new Mat(480, 640, MatType.CV_8UC3);
            Cv2.Randu(img, Scalar.All(230), Scalar.All(256));
            var r = QualityChecker.Check(img, Q);
            Assert.Contains(QualityChecker.TooBright, r.Problems);
            Assert.DoesNotContain(QualityChecker.TooDark, r.Problems);
        }

        [Fact]
        public void Blurry_IsFlagged()
        {
            using var img = TestHelpers.Textured(640, 480);
            using var blurred = new Mat();
            Cv2.GaussianBlur(img, blurred, new Size(0, 0), 15);
            var r = QualityChecker.Check(blurred, Q);
            Assert.Contains(QualityChecker.TooBlurry, r.Problems);
        }

        [Fact]
        public void Small_IsFlagged()
        {
            using var img = TestHelpers.Textured(200, 100);
            var r = QualityChecker.Check(img, Q);
            Assert.Contains(QualityChecker.TooSmall, r.Problems);
            Assert.False(r.IsOk);
        }
    }
}
