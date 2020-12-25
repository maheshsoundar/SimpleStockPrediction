import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.linear_model import RidgeCV
from datetime import datetime, timedelta
import pandas_market_calendars as mcal

seed = 5
np.random.seed(seed)

class StockPredictModel():
	'''
	Class used to predict closing price of a given stock on a given date.
	Change the stock ticker if needed (default 'AAPL') and 
	provide the date on which your prediction is needed.
	'''
	def __init__(self):
		self.ref_stock = '^GSPC'
		self.number_lags = 7
		self.target_col = 'Close'
		self.snp_col = 's&p_close'
		self.train_size = 0.66
		self.model = None
		self.scaler = None
		self.stock = ""

	def __prepare_merge_refstock_data(self,df,refdf):
		refdf.columns = [self.snp_col]
		df = df.merge(refdf,how='inner', left_index=True, right_index=True)
		return df

	def __download_stock_data_range(self,stock,start,end):
		df = yf.download(stock,start,end)
		df = df.filter([self.target_col])
		return df

	def __download_stock_data_train(self,stock):
		tick = yf.Ticker(stock)
		df = tick.history(period='max')
		df = df.filter([self.target_col])
		refdf = yf.Ticker(self.ref_stock).history(period='max')
		refdf = refdf.filter([self.target_col])
		df = self.__prepare_merge_refstock_data(df,refdf)
		return df

	def __get_start_date(self,date):
		business_days_to_sub = 0
		current_date = date
		nyse = list(mcal.get_calendar('NYSE').holidays().holidays)
		holidays = list(map(lambda x:pd.Timestamp(x).to_pydatetime(), nyse))
		while business_days_to_sub < self.number_lags:
			current_date -= timedelta(days=1)
			weekday = current_date.weekday()
			if weekday >= 5: # sunday = 6
				continue
			if current_date in holidays:
				continue
			business_days_to_sub += 1
		return current_date

	def __create_feats(self,df):
		for lag in range(1, self.number_lags + 1):
			df[self.target_col+'(t-' + str(lag)+')'] = df.Close.shift(lag)
			df[self.snp_col+'(t-' + str(lag)+')'] = df[self.snp_col].shift(lag)
		df.dropna(inplace=True)
		return df

	def __create_feats_test(self,df):
		for lag in range(1, self.number_lags):
			df[self.target_col+'(t-' + str(lag)+')'] = df.Close.shift(lag)
			df[self.snp_col+'(t-' + str(lag)+')'] = df[self.snp_col].shift(lag)
		df.dropna(inplace=True)
		return df

	def __split_target(self,df):
		target = df.Close
		df.drop(columns=[self.target_col,self.snp_col],inplace=True)
		return df, target

	def __scale_data(self,df):
		scaler = MinMaxScaler()
		scaled_df = pd.DataFrame(scaler.fit_transform(df),columns=df.columns.values)
		return scaled_df,scaler

	def __train_test_split(self,df,target):
		train_len = math.ceil(len(df.values) * self.train_size)
		x_train = df.iloc[0:train_len]
		y_train = target.iloc[0:train_len]
		x_test = df.iloc[train_len:]
		y_test = target.iloc[train_len:]
		return x_train,y_train,x_test,y_test

	def __train_model(self,x_train,y_train,x_test,y_test):
		rid = RidgeCV(alphas=[0.001, 0.01, 0.1, 1],cv=3).fit(x_train, y_train)
		return rid , rid.score(x_test, y_test)

	def __validate(self,date_text):
		try:
			datetime.strptime(date_text, '%Y-%m-%d')
		except ValueError:
			raise ValueError("Incorrect data format, should be YYYY-MM-DD")

	def __run_model(self,df):
		if(df is None or df.shape[0]==0):
			return None
		return self.model.predict(df)

	def build_model(self,stock='AAPL'):
		'''
		Attributes
    	----------
    	stock : str
        	stock symbol
        Builds a model for the given ticker symbol. Must be called before test_run.
        '''
		try:
			self.stock = stock
			data = self.__download_stock_data_train(self.stock)
			print("Stock data downloaded\n")
			data = self.__create_feats(data)
			data, target = self.__split_target(data)
			scaled_data,self.scaler = self.__scale_data(data)
			x_train,y_train,x_test,y_test = self.__train_test_split(scaled_data,target)
			print("Training model started\n")
			self.model, score = self.__train_model(x_train,y_train,x_test,y_test)
			print("*******Score of trained model is {} **********".format(score))
		except:
			print('Error occured in building model. Please try again with correct details\n')
			exit()

	def test_run(self,date):
		'''
		Attributes
    	----------
    	date : str
        	date for prediction
        Returns the predicted closing price. Must be called after building a model for the given stock.
		'''
		try:
			print("Test Run\n")
			self.__validate(date)
			date = datetime.strptime(date, '%Y-%m-%d').date()
			start = self.__get_start_date(date)

			if(self.stock == "" or self.model is None or self.scaler is None):
				print("**********Please build model first*********")
				return None
			df = self.__download_stock_data_range(self.stock,start,date)
			refdf = self.__download_stock_data_range(self.ref_stock,start,date)
			df = self.__prepare_merge_refstock_data(df,refdf)
			if(df.shape[0]>self.number_lags):
				df = df.iloc[1:]
			df = self.__create_feats_test(df)
			df = df.rename(columns={self.target_col: "close(t-0)", self.snp_col: "s&p_close(t-0)"})
			df = self.scaler.transform(df)
			print("Data ready for prediction\n")

			print("*****The prediction for date: {} is {}".format(date,self.__run_model(df)))
		except:
			print('Error occured. Please try again with correct details\n')
			exit()

if __name__ == "__main__":
	#Create an object for the class. Then build a model for the stock followed by the test_run to predict the price 
	#for given date. Change the stock symbol and the prediction date as needed.
	stock_predict = StockPredictModel()
	stock_predict.build_model(stock='AAPL')
	stock_predict.test_run('2020-12-28')




