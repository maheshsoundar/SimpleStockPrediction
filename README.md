Stock prediction using simplpe linear regression(Ridge)

Run the stock_prediction.py file after installing all the requirements given in requirements.txt. In the stock_prediction.py, change the stock ticker
if needed (default 'AAPL') and provide the date on which your prediction is needed.

Example.
if __name__ == "__main__":
	stock_predict = StockPredictModel()
	stock_predict.build_model(stock='AAPL')
	stock_predict.test_run('2020-12-30')

It provides the prediction of closing price for that particular stock for the given date after taking into cnsideration last 7 business days of the stock
and s&p500 closing priceduring same time period.

Note: THis is just to showcase the possiblities and methodologies to predict stock prices using machine learning. Advanced machine learning algorithms like LSTMs,
ensemble algorithms, etc. can be used to further improve the prediction apart from more feature engineering. THere is definitely a lot of room for improvement.