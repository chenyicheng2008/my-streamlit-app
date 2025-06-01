import streamlit as st
import pandas as pd
import numpy as np

# 設定頁面標題和圖示 (可選)
st.set_page_config(page_title="我的第一個 Streamlit 應用", page_icon="🎈")

# 應用程式標題
st.title("🎈 我的第一個 Streamlit 應用程式")

# 顯示一些文字
st.write("歡迎來到這個用 Streamlit 製作的簡單應用程式！")

# 滑桿小工具
x = st.slider("選擇一個值 (x)", 0, 100, 25)
st.write(f"您選擇的值是： {x}")

# 按鈕
if st.button("給我驚喜！"):
    st.balloons() # 顯示氣球動畫
    st.success("驚喜！🎉")

# 顯示一個簡單的圖表
st.subheader("隨機數據圖表")
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c'])
st.line_chart(chart_data)

# 顯示 DataFrame
st.subheader("隨機數據表格")
st.dataframe(chart_data.head())

# 顯示圖片 (假設您有一個名為 'image.jpg' 的圖片在同一個資料夾)
# from PIL import Image
# try:
#     img = Image.open("image.jpg")
#     st.image(img, caption="這是一張範例圖片", use_column_width=True)
# except FileNotFoundError:
#     st.warning("找不到 image.jpg，請確保圖片存在於專案目錄中。")
