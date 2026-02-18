using System;
using System.Globalization;
using System.Windows.Data;

namespace EndoscopyApp.Converters
{
    public class BooleanToStringConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            var boolValue = (bool)value;
            var param = parameter as string;
            
            if (!string.IsNullOrEmpty(param) && param.Contains("|"))
            {
                var parts = param.Split('|');
                return boolValue ? parts[0] : parts[1];
            }
            
            return boolValue.ToString();
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}
