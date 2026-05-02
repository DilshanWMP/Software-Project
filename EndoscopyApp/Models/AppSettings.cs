using System;
using System.IO;

namespace EndoscopyApp.Models
{
    public class AppSettings
    {
        public string AdminPassword { get; set; } = "admin123";
        public string MediaPath { get; set; } = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Media");
        public int CameraIndex { get; set; } = 0;
        public string FootPedalPort { get; set; } = "None";
    }
}
