using System.Windows;
using EndoscopyApp.ViewModels;

namespace EndoscopyApp
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            var viewModel = new MainViewModel();
            DataContext = viewModel;
            Closed += (_, _) => viewModel.Shutdown();
        }
    }
}