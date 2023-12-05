import yfinance as yf

startDate = '2020-01-01'
endDate = '2020-04-02'

# stock ticker
ticker = 'LYFT'

resultData = yf.download(ticker, startDate, endDate)
resultData["Date"] = resultData.index
resultData = resultData[["Date", "Open", "High","Low", "Close", "Adj Close", "Volume"]]
resultData.reset_index(drop=True, inplace=True)
resultData.to_csv("{}.csv".format(ticker))