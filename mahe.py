
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import yfinance as yf

 
ticker_symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2021-01-01'
data = yf.download(ticker_symbol, start=start_date, end=end_date)

 
data['Daily_Return'] = data['Adj Close'].pct_change()
data = data.dropna()

 
X = np.array(data['Daily_Return']).reshape(-1, 1)
y = np.array(data['Adj Close'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


plt.figure(figsize=(12, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Daily Returns')
plt.ylabel('Price')
plt.legend()
plt.show()


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
