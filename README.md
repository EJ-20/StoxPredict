# StoxPredict - Stock Prediction with Enhanced Visualizations

A comprehensive stock prediction project using LSTM neural networks with advanced visualizations for data analysis and model performance evaluation.

## Features

### 📊 Enhanced Visualizations
- **Modern Styling**: Professional color schemes and formatting
- **Comprehensive Dashboards**: Multi-panel analysis views
- **Interactive Plots**: Zoom, pan, and hover capabilities with Plotly
- **Model Performance Analysis**: Training history, predictions vs actual, residuals
- **Multi-stock Comparison**: Side-by-side analysis of NVDA, AMD, and TSLA

### 🤖 Machine Learning
- LSTM neural network for time series prediction
- Feature scaling and preprocessing
- Model performance metrics (RMSE, MAPE)
- Training history visualization

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Analysis (Matplotlib)
Run the enhanced main script:
```bash
python stoxData.py
```

This will generate:
- Stock price trends dashboard
- Model performance analysis
- Multi-stock comparison charts

### Interactive Analysis (Plotly)
For interactive visualizations:
```bash
python interactive_plots.py
```

This will create:
- Interactive candlestick charts
- Multi-stock comparison with hover details
- Correlation heatmaps
- Moving averages analysis

## Visual Improvements Implemented

### 1. **Modern Plotting Style**
- Professional color palette
- Consistent formatting and fonts
- Grid lines for better readability
- Proper date formatting

### 2. **Comprehensive Dashboards**
- **Stock Analysis Dashboard**: Price trends, volume, distribution, returns
- **Model Performance Dashboard**: Training loss, predictions, residuals, error distribution
- **Multi-stock Comparison**: Side-by-side analysis of all stocks

### 3. **Interactive Features**
- Zoom and pan capabilities
- Hover tooltips with detailed information
- Unified hover mode for better data exploration
- Responsive layouts

### 4. **Advanced Analytics**
- Moving averages (20-day and 50-day)
- Correlation analysis between stocks
- Daily returns analysis
- Price distribution histograms

### 5. **Model Evaluation**
- Training loss visualization
- Actual vs predicted comparison
- Residual analysis
- Error distribution analysis

## File Structure

```
StoxPredict/
├── stoxData.py          # Main analysis with enhanced matplotlib visualizations
├── interactive_plots.py # Interactive Plotly visualizations
├── requirements.txt     # Dependencies
├── README.md           # This file
├── NVDA.csv           # NVIDIA stock data
├── AMD.csv            # AMD stock data
└── TSLA.csv           # Tesla stock data
```

## Key Visual Enhancements

1. **Color Scheme**: Professional color palette with consistent theming
2. **Layout**: Well-organized subplots with proper spacing
3. **Interactivity**: Hover details, zoom, and pan capabilities
4. **Completeness**: Multiple analysis perspectives in single views
5. **Readability**: Clear titles, labels, and legends

## Performance Metrics

The model provides:
- **RMSE**: Root Mean Square Error
- **MAPE**: Mean Absolute Percentage Error
- **Training History**: Loss over epochs
- **Prediction Accuracy**: Visual comparison of actual vs predicted values

## Future Enhancements

- Real-time data integration
- Additional technical indicators
- Portfolio optimization visualizations
- Risk analysis charts
- Export capabilities for reports
    
