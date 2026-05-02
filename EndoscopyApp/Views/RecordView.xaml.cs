using EndoscopyApp.ViewModels;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Animation;
using System;

namespace EndoscopyApp.Views
{
    public partial class RecordView : UserControl
    {
        private RecordViewModel? _viewModel;

        public RecordView()
        {
            InitializeComponent();
            this.DataContextChanged += OnDataContextChanged;
        }

        private void OnDataContextChanged(object sender, DependencyPropertyChangedEventArgs e)
        {
            // Unsubscribe from old view model
            if (_viewModel != null)
            {
                _viewModel.SnapshotFlashRequested -= OnSnapshotFlashRequested;
                _viewModel.NotificationRequested -= OnNotificationRequested;
            }

            // Subscribe to new view model
            if (this.DataContext is RecordViewModel vm)
            {
                _viewModel = vm;
                _viewModel.SnapshotFlashRequested += OnSnapshotFlashRequested;
                _viewModel.NotificationRequested += OnNotificationRequested;
            }
        }

        private void OnSnapshotFlashRequested()
        {
            Dispatcher.Invoke(() =>
            {
                if (this.Resources["FlashStoryboard"] is Storyboard sb)
                {
                    sb.Begin();
                }
            });
        }

        private void OnNotificationRequested(string message)
        {
            Dispatcher.Invoke(() =>
            {
                NotificationText.Text = message;
                if (this.Resources["ToastStoryboard"] is Storyboard sb)
                {
                    // Stop any existing animation to reset the state
                    sb.Stop();
                    sb.Begin();
                }
            });
        }
    }
}
