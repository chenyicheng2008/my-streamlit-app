# new_ai_macd_kd_gradio.py
import yfinance as yf
import pandas as pd
import numpy as np
import ta # Technical Analysis library
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import mplfinance as mpf
import matplotlib.pyplot as plt
import gradio as gr
import scipy.stats as stats # For normal distribution

# --- Main Analysis and Plotting Function based on ai_macd_kd.py logic ---
def run_stock_analysis_v2(
    stock_ticker_input,
    start_date_input,
    end_date_input,
    data_freq_input,
    std_dev_factor_input, # Changed name from std_dev_threshold_input for clarity
    pred_periods_input,
    last_n_days_input # For training split and v-line position in plot
):
    log_messages = []

    log_messages.append(f"設定參數...")
    STOCK_TICKER = stock_ticker_input.upper()
    START_DATE = start_date_input
    END_DATE = end_date_input
    DataFreq = data_freq_input
    STD_FACTOR = std_dev_factor_input
    PREDICTION_PERIODS = int(pred_periods_input)
    LAST_N_DAYS_PARAM = int(last_n_days_input) # Used for training split and v-line

    # --- 2. 獲取數據 ---
    log_messages.append(f"正在下載 {STOCK_TICKER} 的股價數據 (頻率: {DataFreq})...")
    try:
        data_full = yf.download(STOCK_TICKER, start=START_DATE, end=END_DATE, interval=DataFreq)
        if data_full.empty:
            log_messages.append(f"錯誤：無法獲取 {STOCK_TICKER} 的數據，請檢查股票代碼或日期範圍。")
            return None, "\n".join(log_messages), None

        if isinstance(data_full.columns, pd.MultiIndex):
            if STOCK_TICKER in data_full.columns.get_level_values(1):
                 data_full = data_full.xs(STOCK_TICKER, axis=1, level='Ticker')
            elif ('Open', '') in data_full.columns and ('Close', '') in data_full.columns: # Fallback
                 data_full.columns = data_full.columns.droplevel(1)
            else: # If still multi-index and can't resolve, attempt to use first level if common names exist
                cols = [col[0] if isinstance(col, tuple) else col for col in data_full.columns]
                if 'Open' in cols and 'Close' in cols and 'High' in cols and 'Low' in cols and 'Volume' in cols:
                    data_full.columns = cols
                else:
                    raise ValueError("無法處理下載數據的列名結構。")


    except Exception as e:
        log_messages.append(f"下載數據時發生錯誤: {str(e)}")
        return None, "\n".join(log_messages), None

    if data_full.empty:
        log_messages.append(f"錯誤： {STOCK_TICKER} 的數據為空。")
        return None, "\n".join(log_messages), None
    log_messages.append("數據下載完成。")

    # --- 3. 計算技術指標 ---
    log_messages.append("正在計算 MACD 指標...")
    macd = ta.trend.MACD(data_full['Close'])
    data_full['MACD'] = macd.macd()
    data_full['MACD_signal'] = macd.macd_signal()
    data_full['MACD_hist'] = macd.macd_diff()

    macd_long = ta.trend.MACD(data_full['Close'], window_slow=26*4, window_fast=12*4, window_sign=9*4)
    data_full['MACD_long'] = macd_long.macd()
    data_full['MACD_signal_long'] = macd_long.macd_signal()
    data_full['MACD_hist_long'] = macd_long.macd_diff()
    log_messages.append("MACD 指標計算完成。")

    log_messages.append("正在計算 KD 指標...")
    stoch = ta.momentum.StochasticOscillator(high=data_full['High'], low=data_full['Low'], close=data_full['Close'])
    data_full['STOCH_K'] = stoch.stoch()
    data_full['STOCH_D'] = stoch.stoch_signal()
    data_full['K_gt_D'] = data_full['STOCH_K'] - data_full['STOCH_D']

    stoch_long = ta.momentum.StochasticOscillator(data_full['High'], data_full['Low'], data_full['Close'], window=9*4, smooth_window=3*4) # Corrected from stoch_short
    data_full['STOCH_K_long'] = stoch_long.stoch() # Corrected from stoch_short
    data_full['STOCH_D_long'] = stoch_long.stoch_signal() # Corrected from stoch_short
    log_messages.append("KD 指標計算完成。")

    # --- 4. 特徵工程 ---
    data_full['Future_Price'] = data_full['Close'].shift(-PREDICTION_PERIODS)
    data_full['Price_Change_Ratio'] = (data_full['Future_Price'] - data_full['Close']) / data_full['Close']

    mu_fit, std_fit = 0, 0.01 # Default values
    if 'Price_Change_Ratio' in data_full and not data_full['Price_Change_Ratio'].isnull().all():
        price_change_ratio_clean = data_full['Price_Change_Ratio'].dropna()
        if not price_change_ratio_clean.empty:
            mu_fit, std_fit = stats.norm.fit(price_change_ratio_clean)
        else:
            log_messages.append("警告: Price_Change_Ratio 清理後為空，無法計算常態分配。使用預設標準差。")
    else:
        log_messages.append("警告: Price_Change_Ratio 不存在或全為NaN，無法計算常態分配。使用預設標準差。")

    data_full['BUY_THRESHOLD'] = STD_FACTOR * std_fit / 2
    data_full['SELL_THRESHOLD'] = -1 * STD_FACTOR * std_fit / 2

    features = ['MACD', 'MACD_signal', 'MACD_hist', 'STOCH_K', 'STOCH_D', 'K_gt_D', 'MACD_hist_long', 'STOCH_D_long']
    # Drop NaNs from features and the target for the full dataset before splitting
    data_processed = data_full.dropna(subset=features + ['Price_Change_Ratio']).copy()


    if data_processed.empty:
        log_messages.append("錯誤：數據預處理後為空 (指標計算或dropna導致)。")
        return None, "\n".join(log_messages), None

    X_all = data_processed[features]
    y_all = data_processed['Price_Change_Ratio']

    # --- 5. 模型訓練 ---
    # Training data split: use data up to LAST_N_DAYS_PARAM from the end of available processed data
    # This LAST_N_DAYS_PARAM also defines the segment for the vertical line on the plot.
    
    min_train_size = 30 # Arbitrary minimum number of samples for training
    fixed_plot_window_size = 300 # As per ai_macd_kd.py's data_test = data[-300:]

    if len(X_all) < LAST_N_DAYS_PARAM + min_train_size:
        log_messages.append(f"錯誤：數據不足 ({len(X_all)} 筆) 無法根據 LAST_N_DAYS_PARAM ({LAST_N_DAYS_PARAM}) 和最小訓練樣本 ({min_train_size}) 分割訓練集。")
        return None, "\n".join(log_messages), None
    
    if len(X_all) < fixed_plot_window_size:
        log_messages.append(f"警告：數據總量 ({len(X_all)} 筆) 小於固定的繪圖/回測窗口 ({fixed_plot_window_size} 筆)。將使用所有可用數據進行繪圖/回測。")
        current_plot_window_size = len(X_all)
    else:
        current_plot_window_size = fixed_plot_window_size

    # Define training set
    train_end_index = len(X_all) - LAST_N_DAYS_PARAM
    if train_end_index < min_train_size : # check if training data is sufficient
        log_messages.append(f"錯誤：訓練數據不足 (僅 {train_end_index} 筆)。請減少 LAST_N_DAYS_PARAM 或提供更多歷史數據。")
        return None, "\n".join(log_messages), None

    X_train = X_all.iloc[:train_end_index]
    y_train = y_all.iloc[:train_end_index]
    
    # Data for prediction, signal generation, and plotting (fixed window of 300, or less if not enough data)
    data_for_signals_plot = data_processed.iloc[-current_plot_window_size:].copy()
    X_features_for_signals_plot = X_all.iloc[-current_plot_window_size:]

    log_messages.append(f"全部特徵數據大小: {X_all.shape}")
    log_messages.append(f"訓練數據大小: {X_train.shape}, 用於預測/繪圖的數據大小: {X_features_for_signals_plot.shape}")

    if X_train.empty or X_features_for_signals_plot.empty:
        log_messages.append("錯誤：訓練集或預測/繪圖集為空。")
        return None, "\n".join(log_messages), None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_features_for_signals_plot_scaled = scaler.transform(X_features_for_signals_plot)

    log_messages.append("正在訓練 AI 模型 (Random Forest Regressor)...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    log_messages.append("模型訓練完成。")

    predicted_change_ratios = model.predict(X_features_for_signals_plot_scaled)
    data_for_signals_plot.loc[:, 'Predicted_Change'] = predicted_change_ratios
    
    # --- 6. 生成交易訊號 (在 data_for_signals_plot 上) ---
    log_messages.append("正在生成交易訊號...")
    data_for_signals_plot.loc[:, 'Signal'] = 0
    
    # Thresholds should already be present from data_processed, fill any potential NaNs if any edge case
    data_for_signals_plot['BUY_THRESHOLD'] = data_for_signals_plot['BUY_THRESHOLD'].fillna(STD_FACTOR * std_fit / 2)
    data_for_signals_plot['SELL_THRESHOLD'] = data_for_signals_plot['SELL_THRESHOLD'].fillna(-1 * STD_FACTOR * std_fit / 2)

    data_for_signals_plot.loc[data_for_signals_plot['Predicted_Change'] >= data_for_signals_plot['BUY_THRESHOLD'], 'Signal'] = 1
    data_for_signals_plot.loc[data_for_signals_plot['Predicted_Change'] <= data_for_signals_plot['SELL_THRESHOLD'], 'Signal'] = -1
    log_messages.append("交易訊號生成完畢。")

    # --- 7. 回測与损益计算 ---
    log_messages.append("正在進行回測與損益計算...")
    position = 0
    buy_price = 0
    trades = []
    buy_date_mem = pd.NaT # Store buy date

    for i in range(len(data_for_signals_plot)):
        current_date = data_for_signals_plot.index[i]
        current_price = data_for_signals_plot['Close'].iloc[i]
        signal = data_for_signals_plot['Signal'].iloc[i]

        if position == 0:
            if signal == 1:
                position = 1
                buy_price = current_price
                buy_date_mem = current_date
                log_messages.append(f"{buy_date_mem.date()}: 買入訊號 @ {buy_price:.2f}")
        elif position == 1:
            if signal == -1:
                position = 0
                sell_price = current_price
                sell_date = current_date
                profit = sell_price - buy_price
                profit_ratio = (sell_price - buy_price) / buy_price if buy_price != 0 else 0
                trades.append({
                    'Entry_Date': buy_date_mem, 'Entry_Price': buy_price,
                    'Exit_Date': sell_date, 'Exit_Price': sell_price,
                    'Profit': profit, 'Profit_Ratio': profit_ratio
                })
                log_messages.append(f"{sell_date.date()}: 賣出訊號 @ {sell_price:.2f}, 獲利: {profit:.2f} ({profit_ratio*100:.2f}%)")
                buy_price = 0
                buy_date_mem = pd.NaT

    if position == 1 and not data_for_signals_plot.empty:
        # If still holding at the end, close position at last available price
        sell_price = data_for_signals_plot['Close'].iloc[-1]
        sell_date = data_for_signals_plot.index[-1]
        profit = sell_price - buy_price
        profit_ratio = (sell_price - buy_price) / buy_price if buy_price != 0 else 0
        trades.append({
            'Entry_Date': buy_date_mem, 'Entry_Price': buy_price,
            'Exit_Date': sell_date, 'Exit_Price': sell_price,
            'Profit': profit, 'Profit_Ratio': profit_ratio
        })
        log_messages.append(f"{sell_date.date()}: 回測結束，強制賣出 @ {sell_price:.2f}, 獲利: {profit:.2f} ({profit_ratio*100:.2f}%)")

    trades_df = pd.DataFrame(trades)
    summary_text_parts = []
    cumulative_return_percentage = 0.0
    total_profit_calc = 0.0

    if not trades_df.empty:
        total_profit_calc = trades_df['Profit'].sum()
        total_profit_ratio_product = 1
        for ratio in trades_df['Profit_Ratio']:
            total_profit_ratio_product *= (1 + ratio)
        cumulative_return_percentage = (total_profit_ratio_product - 1) * 100

        summary_text_parts.append("\n--- 交易明細 ---")
        summary_text_parts.append(f"\n總交易次數: {len(trades_df)}")
        summary_text_parts.append(f"總損益 (基於每次1股): {total_profit_calc:.2f}")
        if not trades_df['Profit_Ratio'].empty and not np.isnan(trades_df['Profit_Ratio'].mean()):
             summary_text_parts.append(f"平均每次交易報酬率: {trades_df['Profit_Ratio'].mean()*100:.2f}%")
        summary_text_parts.append(f"總累積報酬率 (假設利潤再投資): {cumulative_return_percentage:.2f}%")
    else:
        summary_text_parts.append("\n回測期間無任何交易發生。")
    log_messages.append("回測與損益計算完成。")

    # --- 8. 繪製K線圖 ---
    log_messages.append("正在繪製K線圖...")
    
    # Data for plotting is data_for_signals_plot (fixed 300 window or less)
    # The ai_macd_kd.py script would plot this data_test (which is 300 points).
    # If data_for_signals_plot is > 450 (not expected here as it's 300), it would be sliced.
    # For consistency with ai_macd_kd.py, if current_plot_window_size (e.g. 300) > 450, slice.
    # But since it's 300, no slicing by 450 is needed based on ai_macd_kd.py logic.
    final_plot_data = data_for_signals_plot # This is the 300-point (or less) window

    buy_signals_plot = [np.nan] * len(final_plot_data)
    sell_signals_plot = [np.nan] * len(final_plot_data)

    for i in range(len(final_plot_data)):
        if final_plot_data['Signal'].iloc[i] == 1:
            buy_signals_plot[i] = final_plot_data['Low'].iloc[i] * 0.98
        elif final_plot_data['Signal'].iloc[i] == -1:
            sell_signals_plot[i] = final_plot_data['High'].iloc[i] * 1.02
    
    ap_buy = mpf.make_addplot(buy_signals_plot, type='scatter', marker='^', color='green', markersize=40, alpha=0.6)
    ap_sell = mpf.make_addplot(sell_signals_plot, type='scatter', marker='v', color='red', markersize=40, alpha=0.6)
    plot_addplots = [ap_buy, ap_sell]

    last_buy_threshold_val = final_plot_data['BUY_THRESHOLD'].iloc[-1]*100 if not final_plot_data.empty else float('nan')

    title_str = (f'{STOCK_TICKER} AI Signals (MACD+KD) - Based on ai_macd_kd.py\n'
                 f'Cum.Ret: {cumulative_return_percentage:.2f}% | Thr:{last_buy_threshold_val:.2f}% in {PREDICTION_PERIODS}*{DataFreq}')

    fig = None
    if not final_plot_data.empty:
        try:
            fig, axes = mpf.plot(final_plot_data,
                                type='candle',
                                style='yahoo',
                                title=title_str,
                                addplot=plot_addplots,
                                volume=True,
                                figratio=(16,9),
                                datetime_format='%y-%m-%d',
                                xrotation=45,
                                returnfig=True,
                                tight_layout=True)
            
            # Potential manual adjustment if tight_layout is not enough:
            fig.subplots_adjust(left=0.03, right=0.9, bottom=0.15, top=0.92)
            price_ax = axes[0]
            price_ax.yaxis.tick_right()
            # Values are fractions of figure width/height. Needs experimentation.


            # Vertical line logic from ai_macd_kd.py:
            # Position is LAST_N_DAYS_PARAM from the end of the plotted data.
            # length of final_plot_data is current_plot_window_size (e.g. 300)
            vertical_line_plot_idx = len(final_plot_data) - LAST_N_DAYS_PARAM
            
            if vertical_line_plot_idx >= 0 and vertical_line_plot_idx < len(final_plot_data):
                 axes[0].axvline(x=vertical_line_plot_idx, color='blue', linestyle='--', linewidth=1)
                 axes[0].annotate('Predictions Begin Here', 
                                  xy=(vertical_line_plot_idx, axes[0].get_ylim()[1]*0.95),
                                  xytext=(5, -5), textcoords='offset points', 
                                  ha='left', va='top',
                                  fontsize=9, color='blue',
                                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", lw=0.5, alpha=0.7))
            elif LAST_N_DAYS_PARAM >= len(final_plot_data) : # If LAST_N_DAYS_PARAM covers the whole plot or more
                 axes[0].axvline(x=0, color='blue', linestyle='--', linewidth=1) # Place at beginning
                 axes[0].annotate('Entire Plot is Prediction Window', 
                                  xy=(0, axes[0].get_ylim()[1]*0.95),
                                  xytext=(5, -5), textcoords='offset points', 
                                  ha='left', va='top',
                                  fontsize=9, color='blue',
                                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", lw=0.5, alpha=0.7))


            log_messages.append("K線圖繪製完成。")
        except Exception as e:
            log_messages.append(f"繪製K線圖時發生錯誤: {str(e)}")
            fig = plt.figure()
            plt.text(0.5, 0.5, f"繪圖錯誤: {str(e)}", ha='center', va='center')
    else:
        log_messages.append("繪圖數據為空，無法繪製K線圖。")
        fig = plt.figure()
        plt.text(0.5, 0.5, "無數據可繪製K線圖", ha='center', va='center')

    return fig, "\n".join(log_messages) + "\n" + "\n".join(summary_text_parts), trades_df


# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(), title="股票分析與交易訊號 (v2 - ai_macd_kd base)") as demo_v2:
    gr.Markdown("# AI 股票分析與交易訊號 (MACD+KD) - v2 (Based on ai_macd_kd.py logic)")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 參數設定")
            stock_ticker_input = gr.Textbox(label="股票代碼 (例如 GOOG, 2330.TW)", value="^TWII")
            start_date_input = gr.Textbox(label="開始日期 (YYYY-MM-DD)", value="1990-01-01") # Adjusted start for more data
            end_date_input = gr.Textbox(label="結束日期 (YYYY-MM-DD)", value="2026-05-31") # Adjusted end
            data_freq_input = gr.Dropdown(label="數據頻率", choices=['1d', '1wk'], value='1d', info="日線='1d', 週線='1wk'")
            std_dev_factor_input = gr.Slider(label="買賣閾值標準差因子 (STD_FACTOR)", minimum=0.5, maximum=3.0, step=0.1, value=1.3, info="預測漲跌幅標準差的倍數/2")
            pred_periods_input = gr.Slider(label="預測目標期數 (N 期後)", minimum=1, maximum=40, step=1, value=4, info="預測未來N個K棒後的漲跌幅")
            last_n_days_input = gr.Slider(label="訓練數據排除期數 / 圖中預測區間期數", minimum=10, maximum=250, step=10, value=50, info="訓練時排除最後N期; 圖表上藍色虛線右側為此範圍 (在300期K線圖內)")
            
            analyze_button = gr.Button("📈 開始分析與繪圖", variant="primary")
            gr.Markdown("注意：此為AI模型輔助分析，非投資建議，請謹慎評估風險。結果僅供參考。圖表主要顯示最近約300期數據。")

        with gr.Column(scale=4):
            output_plot = gr.Plot(label="K線圖與交易訊號 (最近約300期)")
    
    with gr.Accordion("詳細日誌與回測結果", open=False):
        output_logs = gr.Textbox(label="日誌與摘要", lines=20, interactive=False, show_copy_button=True)
    
    with gr.Accordion("交易明細表", open=False):
        output_trades_df = gr.DataFrame(label="交易記錄", wrap=True)

    analyze_button.click(
        fn=run_stock_analysis_v2,
        inputs=[
            stock_ticker_input, 
            start_date_input, 
            end_date_input, 
            data_freq_input,
            std_dev_factor_input,
            pred_periods_input,
            last_n_days_input
        ],
        outputs=[output_plot, output_logs, output_trades_df]
    )

if __name__ == '__main__':
    # For testing, you might want to ensure the script can be run directly
    # demo_v2.launch(share=True) # Example with share=True
    demo_v2.launch(share=True) 