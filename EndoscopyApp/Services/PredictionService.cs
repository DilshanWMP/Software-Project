using System;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using EndoscopyApp.Models;

namespace EndoscopyApp.Services
{
    public class PredictionService
    {
        private readonly HttpClient _httpClient;
        private const string DefaultBaseUrl = "http://127.0.0.1:8000";

        public PredictionService(string baseUrl = DefaultBaseUrl)
        {
            _httpClient = new HttpClient
            {
                BaseAddress = new Uri(baseUrl),
                Timeout = TimeSpan.FromSeconds(10)
            };
        }

        public async Task<bool> IsServiceAvailableAsync(CancellationToken cancellationToken = default)
        {
            try
            {
                var response = await _httpClient.GetAsync("/health", cancellationToken);
                return response.IsSuccessStatusCode;
            }
            catch
            {
                return false;
            }
        }

        public async Task<PredictionResult> PredictAsync(string imagePath, CancellationToken cancellationToken = default)
        {
            if (!File.Exists(imagePath))
            {
                throw new FileNotFoundException($"Image file not found: {imagePath}");
            }

            using var content = new MultipartFormDataContent();
            var fileBytes = await File.ReadAllBytesAsync(imagePath, cancellationToken);
            var byteArrayContent = new ByteArrayContent(fileBytes);

            string ext = Path.GetExtension(imagePath).ToLowerInvariant();
            string mimeType = ext == ".png" ? "image/png" : "image/jpeg";
            byteArrayContent.Headers.ContentType = new MediaTypeHeaderValue(mimeType);

            content.Add(byteArrayContent, "file", Path.GetFileName(imagePath));

            var response = await _httpClient.PostAsync("/predict", content, cancellationToken);
            var responseString = await response.Content.ReadAsStringAsync(cancellationToken);

            if (!response.IsSuccessStatusCode)
            {
                throw new HttpRequestException($"Service error ({response.StatusCode}): {responseString}");
            }

            var result = JsonSerializer.Deserialize<PredictionResult>(responseString);
            return result ?? throw new InvalidOperationException("Failed to deserialize prediction response.");
        }
    }
}

