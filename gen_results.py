import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import six
from SVM import *
import json

# change this fo other stocks
stock_name = 'Lyft'

data = pd.read_csv('{0}_processed.csv'.format(stock_name), usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
data['Date'] = pd.to_datetime(data.Date, format='%Y-%m-%d')
data['OBV'] = (data['OBV'] - data['OBV'].min()) / (data['OBV'].max() - data['OBV'].min())  
data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')

with open('sentiment.json') as f:
    sentiment = json.load(f)

d = sentiment[stock_name]
for index, row in data.iterrows():
    if row['Date'] in d:
        data.loc[index, "Sentiment"] = d[row['Date']]

def gen_heatmap():
    corr = data.corr()
    corr.style.background_gradient(cmap='coolwarm')

    # Generate a mask for the upper triangle; True = do NOT show
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    # More details at https://seaborn.pydata.org/generated/seaborn.heatmap.html
    sns.heatmap(
        corr,          # The data to plot
        mask=mask,     # Mask some cells
        cmap=cmap,     # What colors to plot the heatmap as
        annot=True,    # Should the values be plotted in the cells?
        vmax=1.0,       # The maximum value of the legend. All higher vals will be same color
        vmin=-1.0,      # The minimum value of the legend. All lower vals will be same color
        center=0,      # The center value of the legend. With divergent cmap, where white is
        square=True,   # Force cells to be square
        linewidths=.5, # Width of lines that divide cells
        cbar_kws={"shrink": .5}  # Extra kwargs for the legend; in this case, shrink by 50%
    )

    # You can save this as a png with
    f.savefig('img/heatmap_{0}.png'.format(stock_name).format(stock_name))

def render_mpl_table(data, fname, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)
    mpl_table.auto_set_column_width(col=[0,1,2,3])

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])

    return ax, fig.savefig(fname)

render_mpl_table(data.round(2)[0:10], 'img/ds_{0}.png'.format(stock_name), header_columns=0, col_width=2.0)
gen_heatmap()

change = data["Close"].diff()
change[change<=0] = -1.0
change[change>0] = 1.0
data['CHG'] = change

def get_err(A):

    A_train = A[0:40,:]
    A_test = A[40:,:]
    w = np.linalg.lstsq(A_train[:, 0:A_train.shape[1] - 2], A_train[:,-1], rcond=None)[0]
    mse = np.mean(np.square(np.matmul(A_test[:, 0:A_test.shape[1] - 2], w) - A_test[:,-1]))
    return mse

def get_A(col_name=None):
    res = []
    for index, row in data.iterrows():
        tmp = [row['Open'], row['High'], row['Low'], row['Close'], row['Adj Close'], row['Volume']]
        if col_name is not None and not pd.isnull(row[col_name]):
            tmp.append(row[col_name])

        if col_name is None:
            res.append(tmp)
        else:
            if not pd.isnull(row[col_name]):
                res.append(tmp)
    res = np.array(res)
    num_cols = 8 if col_name == None else 9
    A = np.ones((res.shape[0], num_cols))
    A[:,0:A.shape[1] - 2] = res[:,0:A.shape[1] - 2]
    A[0:A.shape[0]-1, -1] = res[1:,-3]
    return np.delete(A, (-1), axis=0)

def get_A_SVM(col_name=None):
    res = []
    cols = ['Open', 'High', 'Low', 'Adj Close', 'Volume', 'Close']
    if col_name is not None:
        cols.append(col_name)
    cols.append('CHG')
    for index, row in data.iterrows():
        bools = [pd.isnull(row[name]) for name in cols]
        if not any(bools):
            tmp = []
            for name in cols:
                tmp.append(row[name])
            res.append(tmp)
    res = np.array(res)
    return (np.transpose(res[:,0:res.shape[1] - 1]), res[:,[-1]])

def err_svm(data):
    X = data[0]
    y = data[1]

    X_train = X[:,0:40]
    X_test = X[:,40:]
    y_train = y[0:40]
    y_test = y[40:]
    b, b0 = SoftMarg(X_train, y_train, 0.5)
    test_err = classification_acc(classify(X_test, b, b0), y_test)
    return test_err

def gen_results():
    # Create the pandas DataFrame 
    str1 = "Open, High, Low, Adj Close, Volume, Close(Today)"
    str2 = "Open, High, Low, Adj Close, Volume, Close(Today),  Sentiment"
    str3 = "Open, High, Low, Adj Close, Volume, Close(Today), RSI"
    str4 = "Open, High, Low, Adj Close, Volume, Close(Today), SMA"
    str5 = "Open, High, Low, Adj Close, Volume, Close(Today), OBV"

    data['CHG'] = data['CHG'].shift(-1)
    A = err_svm(get_A_SVM())
    B = err_svm(get_A_SVM('Sentiment'))
    C = err_svm(get_A_SVM('RSI'))
    D = err_svm(get_A_SVM('SMA'))
    E = err_svm(get_A_SVM('OBV'))
    data['CHG'] = data['CHG'].shift(1)

    res = [[str1, get_err(get_A()), err_svm(get_A_SVM()), A], 
            [str2, get_err(get_A('Sentiment')), err_svm(get_A_SVM('Sentiment')), B], 
            [str3, get_err(get_A('RSI')), err_svm(get_A_SVM('RSI')), C],
            [str4, get_err(get_A('SMA')), err_svm(get_A_SVM('SMA')), D],
            [str5, get_err(get_A('OBV')), err_svm(get_A_SVM('OBV')), E]] 

    df = pd.DataFrame(res, columns=['Features Used', 'MSE', 'Clas. Error (Tomrrrow\'s Trend)', 'Clas. Error (Today\'s Trend)']) 

    render_mpl_table(df.round(4), 'img/all_{0}.png'.format(stock_name),  header_columns=0, col_width=5.0)

gen_results()