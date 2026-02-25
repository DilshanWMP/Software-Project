namespace EndoscopyApp.Services
{
    public interface INavigationService
    {
        void NavigateTo<TViewModel>() where TViewModel : ViewModels.ViewModelBase;
    }
}
