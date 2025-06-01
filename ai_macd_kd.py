
 
import yfinance as yf
import pandas as pd
import numpy as np
import ta # Technical Analysis library (pip install ta)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import mplfinance as mpf # For plotting candlesticks (pip install mplfinance)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # Import mdates for date handling
# --- 1. 設定參數 ---


# --- 1. 設定參數 ---
# STOCK_TICKER = '6789.TW'  # 創意
# STOCK_TICKER = 'TSM'     # 台積電 ADR
# STOCK_TICKER = '^TWII'   # 台灣加權指數
# STOCK_TICKER = 'NVDA'    # NVIDIA
# STOCK_TICKER = '^SPX'    # S&P 500
# STOCK_TICKER = '5234.TW'  # 達興材料
# STOCK_TICKER = '3715.TW'  # 定穎投控
# STOCK_TICKER = '4760.TWO' # 勤凱科技 (上櫃)
# STOCK_TICKER = '8069.TWO' # 元太科技 (上櫃)
# STOCK_TICKER = '6863.TW'  # 永道-KY
# STOCK_TICKER = '6596.TWO' # 寬宏藝術 (上櫃)
STOCK_TICKER = '6223.TWO' # 檢測 (上櫃) - 此為腳本中最後生效的股票代碼
STOCK_TICKER = 'FORM' # 檢測 (上櫃) - 此為腳本中最後生效的股票代碼
STOCK_TICKER = 'GOOG'   #

START_DATE = '1990-01-01'
END_DATE = '2026-05-31' # 假設我們用一部分數據回測，一部分預測

DataFreq='1d' #  '1wk' '1d'


 
STD=1.3 #BUY-SELL thresshold equals X STD standard deviation 2 1.2

PREDICTION_PERIODS = 4 # 預測漲跌幅之目標日4 週約 20 個交易日後 20 5 

#最後? 4 筆不訓練
last_n_days = 30


# --- 2. 獲取數據 ---
print(f"正在下載 {STOCK_TICKER} 的股價數據...")
data = yf.download(STOCK_TICKER, start=START_DATE, end=END_DATE,interval=DataFreq)
 
#dimention adjust
data=data.xs(STOCK_TICKER,axis=1,level='Ticker')


if data.empty:
    print(f"無法獲取 {STOCK_TICKER} 的數據，請檢查股票代碼或日期範圍。")
    exit()

print("數據下載完成。")

# --- 3. 計算技術指標 (MACD) ---
print("正在計算 MACD 指標...")
macd = ta.trend.MACD(data['Close'])
data['MACD'] = macd.macd()
data['MACD_signal'] = macd.macd_signal()
data['MACD_hist'] = macd.macd_diff() # MACD - MACD_signal

# MACD in different periods
macd_short = ta.trend.MACD(data['Close'], window_slow=26*4, window_fast=12*4,window_sign= 9) #(window_slow: int = 26, window_fast: int = 12, window_sign: int = 9, fillna: bool = False)
data['MACD_long'] = macd_short.macd()
data['MACD_signal_long'] = macd_short.macd_signal()
data['MACD_hist_long'] = macd_short.macd_diff() # MACD - MACD_signal


print("MACD 指標計算完成。")

# Add KD calculation after MACD calculation
print("正在計算 KD 指標...")
stoch = ta.momentum.StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'])
data['STOCH_K'] = stoch.stoch()
data['STOCH_D'] = stoch.stoch_signal()
# Add a new column to indicate if K > D
data['K_gt_D'] = data['STOCH_K'] - data['STOCH_D']


# KD in different periods
stoch_short = ta.momentum.StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'], window=9*4, smooth_window=3*4, )#window: int = 14, smooth1: int = 3, smooth2: int = 3 
data['STOCH_K_long'] = stoch_short.stoch()
data['STOCH_D_long'] = stoch_short.stoch_signal()


print("KD 指標計算完成。")




# --- 4. 特徵工程 ---
# X: 使用 MACD 相關指標作為特徵
# y: 未來 N 日的價格變動百分比
data['Future_Price'] = data['Close'].shift(-PREDICTION_PERIODS)
data['Price_Change_Ratio'] = (data['Future_Price'] - data['Close'] ) / data['Close'] 

#計算N 日的歷史價格變動百分比的120日標準差
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats # For normal distribution and Q-Q plot

# Ensure 'Price_Change_Ratio' exists and is clean
if 'Price_Change_Ratio' in data and not data['Price_Change_Ratio'].isnull().all():
    price_change_ratio = data['Price_Change_Ratio'].dropna() # Drop NaNs just in case
 
    # Fit a normal distribution to the data
    mu, std = stats.norm.fit(price_change_ratio)


#data['STD_PCR']=data['Price_Change_Ratio'].rolling(window=300).std()
data['BUY_THRESHOLD']  =  STD*std/2
data['SELL_THRESHOLD'] = -1*STD*std/2

 
if data.empty:
    print("數據預處理後為空，可能是數據量不足或 PREDICTION_PERIODS 過大。")
    exit()

#變數因子
features = ['MACD', 'MACD_signal', 'MACD_hist', 'STOCH_K', 'STOCH_D', 'K_gt_D','MACD_hist_long','STOCH_D_long']

# 移除因為 shift 操作和指標計算產生的 NaN 值
 
data.dropna(subset=features, inplace=True)


X = data[features]
y = data['Price_Change_Ratio']




# --- 5. 模型訓練 ---
# 為了簡單起見，我們這裡不嚴格區分訓練集和測試集來做回測，
# 而是用所有可用歷史數據訓練模型，然後在這些數據上進行"預測"以產生訊號。
# 在實際應用中，應該劃分訓練集和測試集，並在測試集上評估模型。
# chronological split is better for time series
split_ratio = 0.75
split_index = int(len(X) * split_ratio)

#選項:最後?筆不訓練
split_index =  len(X)-last_n_days

X_train, X_test = X[:split_index], X[-300:]
y_train, y_test = y[:split_index], y[-300:]
data_train, data_test = data[:split_index], data[-300:]

print(f"全部數據大小: {X.shape}")
print(f"訓練數據大小: {X_train.shape}, 測試數據大小: {X_test.shape}")

if X_train.empty or X_test.empty:
    print("訓練集或測試集為空，請檢查數據分割。")
    exit()

# 特徵標準化 (可選，但對某些模型有益)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("正在訓練 AI 模型 (Random Forest Regressor)...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)


print("模型訓練完成。")

# 使用模型進行預測 (在測試集上)



predicted_price_change_ratio_test = model.predict(X_test_scaled)
data_test = data_test.copy() #避免SettingWithCopyWarning
data_test.loc[:, 'Predicted_Change'] = predicted_price_change_ratio_test

# --- 6. 生成交易訊號 (在測試集上) ---
print("正在生成交易訊號...")
data_test.loc[:, 'Signal'] = 0 # 0: No signal, 1: Buy, -1: Sell

# 買進訊號: 預測未來4週股價上漲5%
data_test.loc[data_test['Predicted_Change'] >= data_test['BUY_THRESHOLD'],  'Signal'] = 1

# 賣出訊號: 預測未來4週股價下跌5%
data_test.loc[data_test['Predicted_Change'] <= data_test['SELL_THRESHOLD'], 'Signal'] = -1

# data_test['SELL_THRESHOLD'] 之NA 補填入前值
data_test['SELL_THRESHOLD'] = data_test['SELL_THRESHOLD'].ffill() 
data_test['BUY_THRESHOLD'] = data_test['BUY_THRESHOLD'].ffill()

# 確保買賣訊號不會太頻繁 (可選，簡化版不加此邏輯)
# 例如，買入後直到賣出訊號出現才考慮下一次買入
print("交易訊號生成完畢。")

# --- 7. 回測與損益計算 (在測試集 data_test 上) ---
print("正在進行回測與損益計算...")
position = 0 # 0: 空倉, 1: 持倉
buy_price = 0
sell_price = 0
trades = [] # 記錄交易 [(buy_date, buy_price, sell_date, sell_price, profit_ratio)]

# 我們假設在訊號出現的「當日收盤價」執行買賣
# 實際上應該是訊號日的下一日開盤價，這裡為了簡化
for i in range(len(data_test)):
    current_date = data_test.index[i]
    current_price = data_test['Close'].iloc[i]
    signal = data_test['Signal'].iloc[i]

    if position == 0: # 如果目前空倉
        if signal == 1: # 買進訊號
            position = 1
            buy_price = current_price # 假設以當日收盤價買入
            buy_date = current_date
            print(f"{buy_date.date()}: 買入訊號 @ {buy_price:.2f}")
    elif position == 1: # 如果目前持倉
        # 優先處理賣出訊號
        if signal == -1: # 賣出訊號
            position = 0
            sell_price = current_price # 假設以當日收盤價賣出
            sell_date = current_date
            profit = sell_price - buy_price
            profit_ratio = (sell_price - buy_price) / buy_price
            trades.append({
                'Entry_Date': buy_date,
                'Entry_Price': buy_price,
                'Exit_Date': sell_date,
                'Exit_Price': sell_price,
                'Profit': profit,
                'Profit_Ratio': profit_ratio
            })
            print(f"{sell_date.date()}: 賣出訊號 @ {sell_price:.2f}, 獲利: {profit:.2f} ({profit_ratio*100:.2f}%)")
            buy_price = 0 # Reset buy price
        # 如果沒有賣出訊號，但出現新的買入訊號，可以選擇忽略或加倉 (這裡簡化為忽略)

# 如果回測結束時仍持倉，則以最後一日收盤價賣出 (可選)
if position == 1 and len(data_test) > 0:
    sell_price = data_test['Close'].iloc[-1]
    sell_date = data_test.index[-1]
    profit = sell_price - buy_price
    profit_ratio = (sell_price - buy_price) / buy_price
    trades.append({
        'Entry_Date': buy_date,
        'Entry_Price': buy_price,
        'Exit_Date': sell_date,
        'Exit_Price': sell_price,
        'Profit': profit,
        'Profit_Ratio': profit_ratio
    })
    print(f"{sell_date.date()}: 回測結束，強制賣出 @ {sell_price:.2f}, 獲利: {profit:.2f} ({profit_ratio*100:.2f}%)")

trades_df = pd.DataFrame(trades)
total_profit = 0
total_profit_ratio_product = 1 # 用來計算累積報酬率

if not trades_df.empty:
    total_profit = trades_df['Profit'].sum()
    # 計算累積報酬率: (1+r1)*(1+r2)*...*(1+rn) - 1
    # 這裡簡化為將每次交易的資金都再投入
    for ratio in trades_df['Profit_Ratio']:
        total_profit_ratio_product *= (1 + ratio)
    cumulative_return_percentage = (total_profit_ratio_product - 1) * 100

    print("\n--- 交易明細 ---")
    print(trades_df)
    print(f"\n總交易次數: {len(trades_df)}")
    print(f"總損益 (基於每次1股，未考慮資金變化): {total_profit:.2f}")
    if trades_df['Profit_Ratio'].mean() is not np.nan:
         print(f"平均每次交易報酬率: {trades_df['Profit_Ratio'].mean()*100:.2f}%")
    print(f"總累積報酬率 (假設利潤再投資): {cumulative_return_percentage:.2f}%")
else:
    print("\n回測期間無任何交易發生。")
    cumulative_return_percentage = 0.0

print("回測與損益計算完成。")

# --- 8. 繪製K線圖並標示買賣點 (在測試集 data_test 上) ---
print("正在繪製K線圖...")

# 準備買賣點標記
buy_signals_plot = [np.nan] * len(data_test)
sell_signals_plot = [np.nan] * len(data_test)

for i in range(len(data_test)):
    if data_test['Signal'].iloc[i] == 1:
        # 為了視覺效果，將買點標記在K線下方
        buy_signals_plot[i] = data_test['Low'].iloc[i] * 0.98
    elif data_test['Signal'].iloc[i] == -1:
        # 為了視覺效果，將賣點標記在K線集上方
        sell_signals_plot[i] = data_test['High'].iloc[i] * 1.02

ap_buy = mpf.make_addplot(buy_signals_plot, type='scatter', marker='^', color='green', markersize=40,alpha=0.2)
ap_sell = mpf.make_addplot(sell_signals_plot, type='scatter', marker='v', color='red', markersize=40,alpha=0.2)

# Create vertical line at the start of last_n_days
vertical_line_index = len(data_test) - last_n_days
vertical_line_date = data_test.index[vertical_line_index]

 


 
addplots = [ap_buy, ap_sell]
 

# 為了避免圖表過於擁擠，只繪製測試集部分
if len(data_test) > 450 : # 如果測試集數據點太多，只畫最後250個
    plot_data = data_test.iloc[-450:]
    plot_addplots = [
        mpf.make_addplot(buy_signals_plot[-450:], type='scatter', marker='^', color='green', markersize=40,alpha=0.3),
        mpf.make_addplot(sell_signals_plot[-450:], type='scatter', marker='v', color='red', markersize=40,alpha=0.3)
    ]
else:
    plot_data = data_test
    plot_addplots = addplots
#last buy threshold value
last_buy_threshold = data_test['BUY_THRESHOLD'].iloc[-1]*100

title=f'{STOCK_TICKER} with AI Trading Signals(MACD+KD)\n Total Cumulative Return: {cumulative_return_percentage:.2f}%  Test Threshold:{last_buy_threshold:.2f}% in {PREDICTION_PERIODS}*{DataFreq}' # Use 

if not plot_data.empty:
    fig, axes = mpf.plot(plot_data,
            type='candle',
            style='yahoo',
            title=title,
            addplot=plot_addplots,
            volume=True,
            figratio=(43,15),
            datetime_format='%Y-%m-%d',
            
            xrotation=90, # Rotate x-axis labels vertically
            returnfig=True # Add this to return the figure and axes
            )

    # Add vertical line
    # You need to get the index for the vertical line based on the current plot_data
    # The vertical line should be at the start of the "predictions begins" section
    # This index needs to be relative to the 'plot_data' if you are slicing it
    vertical_line_index_in_plot_data = len(plot_data) - last_n_days if len(data_test) > 450 else len(data_test) - last_n_days

    if vertical_line_index_in_plot_data >= 0: # Ensure index is valid for the sliced data
         axes[0].axvline(x=vertical_line_index_in_plot_data, color='blue', linestyle='--', linewidth=1)

        # Add text annotation
        # The x-coordinate for the annotation should also be based on the index in plot_data
         annotation_x_position = vertical_line_index_in_plot_data

         axes[0].annotate('predictions begins', xy=(annotation_x_position, axes[0].get_ylim()[1]),
                        xytext=(0, 10), textcoords='offset points', ha='right', va='bottom',
                        fontsize=11, color='blue')
  

    print("K線圖繪製完成。")
    plt.show()
else:
    print("測試集數據為空，無法繪製K線圖。")