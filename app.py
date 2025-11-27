import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# --- 1. 页面配置 ---
st.set_page_config(page_title="南昌房价预测系统", page_icon="🏠")

st.title("🏠 AI 南昌房价预测小助手")
st.markdown("这是一个基于机器学习的简易房价预测模型（学生作业演示版）。")

# --- 2. 模拟南昌房价数据 ---
data = {
    '区域': ['红谷滩区', '红谷滩区', '西湖区', '西湖区', '青山湖区', '青山湖区', '新建区', '新建区', '高新区', '高新区'] * 10,
    '面积': [80, 120, 90, 130, 70, 110, 85, 125, 95, 135] * 10,
    '房龄': [2, 5, 10, 15, 20, 3, 5, 8, 2, 6] * 10
}
df = pd.DataFrame(data)

# 价格逻辑
price_map = {'红谷滩区': 1.8, '西湖区': 1.4, '青山湖区': 1.2, '新建区': 1.0, '高新区': 1.5}
prices = []
for i in range(len(df)):
    area = df['面积'][i]
    region = df['区域'][i]
    age = df['房龄'][i]
    base_price = price_map[region] * area
    depreciation = age * 0.5
    random_fluctuation = np.random.randint(-10, 10)
    final_price = base_price - depreciation + random_fluctuation
    prices.append(final_price)
df['价格'] = prices

# --- 3. 训练模型 ---
# 这一步非常关键：我们要记录下训练时的列顺序！
X = df[['面积', '房龄']]
# 使用 get_dummies 转换区域，并确保保存这个列顺序
X_dummies = pd.get_dummies(df['区域'])
X = X.join(X_dummies)

# !!! 核心修复点：保存训练时的列名列表 !!!
model_columns = X.columns.tolist()

y = df['价格']
model = LinearRegression()
model.fit(X, y)

# --- 4. 侧边栏：用户输入 ---
st.sidebar.header("请设置房屋参数")
input_region = st.sidebar.selectbox("选择区域", list(price_map.keys()))
input_area = st.sidebar.slider("房屋面积 (平米)", 30, 300, 100)
input_age = st.sidebar.slider("房龄 (年)", 0, 50, 5)

# --- 5. 进行预测 ---
if st.button("开始预测房价"):
    try:
        # 1. 先创建一个只有面积和房龄的数据框
        input_data = pd.DataFrame([[input_area, input_age]], columns=['面积', '房龄'])
        
        # 2. 手动添加所有区域的列，初始化为0
        for col in model_columns:
            if col not in ['面积', '房龄']:
                input_data[col] = 0 # 先把所有区域都设为0
        
        # 3. 把用户选的那个区域设为1
        if input_region in input_data.columns:
            input_data[input_region] = 1
            
        # 4. !!! 核心修复点：强制重新排列列的顺序，必须和 model_columns 一模一样 !!!
        input_data = input_data[model_columns]
        
        # 5. 预测
        prediction = model.predict(input_data)[0]
        
        st.success(f"📍 区域：{input_region}")
        st.info(f"📏 面积：{input_area} 平米 | 🏚️ 房龄：{input_age} 年")
        st.metric(label="AI 估算总价", value=f"{prediction:.2f} 万元")
        
        unit_price = (prediction * 10000) / input_area
        st.write(f"折合单价约为：{unit_price:.0f} 元/平米")
        
    except Exception as e:
        st.error(f"预测出错: {e}")

# --- 6. 展示部分数据 ---
st.markdown("---")
st.subheader("📊 历史数据概览")
st.dataframe(df.head(10))
