import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# --- 1. 页面配置 ---
st.set_page_config(page_title="南昌房价预测系统", page_icon="🏠")

st.title("🏠 AI 南昌房价预测小助手")
st.markdown("这是一个基于机器学习的简易房价预测模型（学生作业演示版）。")

# --- 2. 模拟南昌房价数据 (为了作业演示，我们构造一些数据) ---
# 真实项目中，你应该读取 excel 文件: data = pd.read_excel("nanchang_house.xlsx")
# 这里我们模拟：红谷滩最贵，偏远地区便宜
data = {
    '区域': ['红谷滩区', '红谷滩区', '西湖区', '西湖区', '青山湖区', '青山湖区', '新建区', '新建区', '高新区', '高新区'] * 10,
    '面积': [80, 120, 90, 130, 70, 110, 85, 125, 95, 135] * 10,
    '房龄': [2, 5, 10, 15, 20, 3, 5, 8, 2, 6] * 10
}
df = pd.DataFrame(data)

# 给数据加上价格逻辑 (模拟真实规律：单价 * 面积 - 折旧)
# 假设基准单价：红谷滩 1.8万, 西湖 1.4万, 青山湖 1.2万, 新建 1.0万, 高新 1.5万
price_map = {'红谷滩区': 1.8, '西湖区': 1.4, '青山湖区': 1.2, '新建区': 1.0, '高新区': 1.5}

# 生成价格 (单位：万元) - 加上一点随机波动让它看起来真实
prices = []
for i in range(len(df)):
    area = df['面积'][i]
    region = df['区域'][i]
    age = df['房龄'][i]
    base_price = price_map[region] * area
    depreciation = age * 0.5  # 每年折旧 5000元
    random_fluctuation = np.random.randint(-10, 10) # 随机波动
    final_price = base_price - depreciation + random_fluctuation
    prices.append(final_price)

df['价格'] = prices

# --- 3. 训练模型 ---
# 机器学习不认识中文"红谷滩"，需要转换成数字 (One-Hot 编码)
X = df[['面积', '房龄']]
X = X.join(pd.get_dummies(df['区域'])) # 把区域变成 0/1 矩阵
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
    # 构造用户输入的数据，格式要和训练时一样
    input_data = pd.DataFrame([[input_area, input_age]], columns=['面积', '房龄'])
    
    # 处理区域的 One-Hot 编码
    for region in price_map.keys():
        input_data[region] = 1 if region == input_region else 0
        
    # 预测
    prediction = model.predict(input_data)[0]
    
    st.success(f"📍 区域：{input_region}")
    st.info(f"📏 面积：{input_area} 平米 | 🏚️ 房龄：{input_age} 年")
    st.metric(label="AI 估算总价", value=f"{prediction:.2f} 万元")
    
    # 算一下单价展示
    unit_price = (prediction * 10000) / input_area
    st.write(f"折合单价约为：{unit_price:.0f} 元/平米")

# --- 6. 展示部分数据 (增加作业丰富度) ---
st.markdown("---")
st.subheader("📊 历史数据概览")

st.dataframe(df.head(10)) # 展示前10条数据
