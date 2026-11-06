using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Threading;
using EndoscopyApp.ViewModels;

namespace EndoscopyApp;

/// <summary>
/// Interaction logic for App.xaml
/// </summary>
public partial class App : Application
{
    private const string AppTitle = "Endoscopy App";

    // If errors keep happening this often, the app is in a bad state: save and close.
    private const int MaxErrorsInWindow = 3;
    private static readonly TimeSpan ErrorWindow = TimeSpan.FromSeconds(10);

    private readonly Queue<DateTime> _recentErrors = new();
    private bool _isShowingError;
    private bool _isShuttingDown;

    public App()
    {
        // Safety net: an unexpected error must never lose a recording mid-procedure.
        DispatcherUnhandledException += OnUiThreadException;
        AppDomain.CurrentDomain.UnhandledException += OnFatalException;
        // Errors in fire-and-forget background tasks: ignore instead of crashing.
        TaskScheduler.UnobservedTaskException += (_, e) => e.SetObserved();
    }

    /// <summary>Errors on the UI thread: save the recording, tell the user, keep running.</summary>
    private void OnUiThreadException(object sender, DispatcherUnhandledExceptionEventArgs e)
    {
        e.Handled = true;
        bool recordingSaved = SaveRecording();

        if (_isShuttingDown) return;

        // The app failed while starting (main window never appeared): nothing to keep running.
        bool appNotStarted = MainWindow == null || !MainWindow.IsVisible;

        var now = DateTime.Now;
        _recentErrors.Enqueue(now);
        while (_recentErrors.Count > 0 && now - _recentErrors.Peek() > ErrorWindow)
            _recentErrors.Dequeue();

        if (appNotStarted || _recentErrors.Count >= MaxErrorsInWindow)
        {
            _isShuttingDown = true;
            string reason = appNotStarted
                ? "The app could not start."
                : "The app keeps running into errors and has to close.";
            MessageBox.Show($"{reason}\n\n{RecordingNote(recordingSaved)}Error: {e.Exception.Message}",
                AppTitle, MessageBoxButton.OK, MessageBoxImage.Error);
            Shutdown();
            return;
        }

        if (_isShowingError) return; // one message at a time
        _isShowingError = true;
        try
        {
            MessageBox.Show($"Something went wrong: {e.Exception.Message}\n\n{RecordingNote(recordingSaved)}You can continue using the app.",
                AppTitle, MessageBoxButton.OK, MessageBoxImage.Warning);
        }
        finally
        {
            _isShowingError = false;
        }
    }

    /// <summary>Errors on a background thread: Windows will close the app, so save the recording first.</summary>
    private void OnFatalException(object sender, UnhandledExceptionEventArgs e)
    {
        bool recordingSaved = SaveRecording();
        string error = (e.ExceptionObject as Exception)?.Message ?? "Unknown error";
        try
        {
            MessageBox.Show($"A serious error happened and the app has to close.\n\n{RecordingNote(recordingSaved)}Error: {error}",
                AppTitle, MessageBoxButton.OK, MessageBoxImage.Error);
        }
        catch { /* the process is ending anyway */ }
    }

    private static bool SaveRecording()
    {
        try
        {
            return MainViewModel.Current?.SaveRecordingAfterError() ?? false;
        }
        catch
        {
            return false;
        }
    }

    private static string RecordingNote(bool recordingSaved) =>
        recordingSaved ? "The recording was stopped and saved.\n\n" : "";
}
