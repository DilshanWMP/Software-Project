using System;

namespace EndoscopyApp.Models
{
    public class MediaFile
    {
        public int Id { get; set; }
        public int PatientId { get; set; }
        public string FilePath { get; set; } = string.Empty;
        public string FileType { get; set; } = string.Empty; // Image, Video
        public DateTime CreatedAt { get; set; } = DateTime.Now;
    }
}
