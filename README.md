# House Price Prediction

This project implements machine learning models to predict house prices using the California Housing dataset.

## Project Structure
- `data/housing.csv`: The dataset used for training and prediction.
- `house_price_prediction.ipynb`: Jupyter Notebook containing the code for EDA, training, and prediction.
- `model_training/Test_Case.py`: Script to manually input house features and get a price prediction.
- `requirements.txt`: List of Python dependencies.

## Models Used
1. **Simple Linear Regression**: Used for initial training and establishing a baseline.
2. **Random Forest Regression**: A powerful ensemble method for better prediction accuracy.
3. **XGBoost Regression**: Gradient boosting algorithm for high-performance prediction.

## Visualizations
- **Correlation Heatmap**: To view relationships between features.
- **Scatter Plot**: Lat vs. Long with median_house_value.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook house_price_prediction.ipynb
   ```
3. Run all cells to see the analysis and model results.

## Interactive Prediction Tool
To manually predict the price of a house using the trained model:
1. Run the script:
   ```bash
   python Test_Case.py
   ```
## What I learned

Income is by far the strongest predictor. Location matters a lot too. The simple linear model underfit pretty badly, but XGBoost nailed it once I tuned the hyperparameters.

Feel free to mess around with the feature engineering or try different models.   