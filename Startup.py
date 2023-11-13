import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('startUp(1).csv')
data.head()

# this is use for duplicating the data
dx = data.copy()

dx.info()

dx.isnull().sum()

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()

# for i in dx.columns:
#     if dx[i].dtypes != 'O':
#         dx[[i]] = scaler.fit_transform(dx[[i]])

dx.drop('Unnamed: 0',axis = 1, inplace = True)
dx.head()

# drop state because it does satisfy the assumption of linearity
dx.drop('State',axis = 1, inplace = True)
# dx.columns

# TRAIN AND TEST SPLIT
x = dx.drop('Profit', axis = 1)
y = dx.Profit

from sklearn .model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size = 0.80, random_state = 57)
print(f'xtrain: {xtrain.shape}')
print(f'ytrain: {ytrain.shape}')
print(f'xtest: {xtest.shape}')
print(f'ytest: {ytest.shape}')


# -----------------------MODELLING--------------------------
from sklearn.linear_model  import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


lin_reg = LinearRegression()

lin_reg.fit(xtrain, ytrain) #---------------------create a linear regression model

#--------cross validation---------------
cross_validate = lin_reg.predict(xtrain)
score = r2_score(cross_validate, ytrain)
print(f'The cross validation Score is: {score.round(2)}')


import pickle
pickle.dump(lin_reg, open('StartUp_Model.pkl', 'wb'))


#---------------------- STEEAMLIT DEVELOPMENT ----------------------
st.markdown("<h1 style = 'color: #141E46; text-align: center; font-family:  Helvetica, sans-serif'>STARTUP PROJECT</h1>", unsafe_allow_html=True)

st.markdown("<h4 style = 'margin: -25px; color: #D6D46D; text-align: center; font-family: Times New Roman, Helvetica, sans-serif'>BUILT BY ADEKUNLE MOJEED</h4>", unsafe_allow_html=True)


st.image('pngwing.com (1).png',width = 800) #---- to give it image
st.markdown("<h2 style = 'color: #D6D46D; text-align: center;font-family: Arial, Helvetica, sans-serif; '>BACKGROUND OF STUDY </h2>", unsafe_allow_html= True)

st.markdown("<p>By analyzing a diverse set of parameters, including market trends, competitive landscape, financial indicators, and operational strategies, our team seeks to develop a robust predictive model that can offer valuable insights into the future financial performance of startups. This initiative not only empowers investors and stakeholders to make data-driven decisions but also provides aspiring entrepreneurs with a comprehensive framework to evaluate the viability of their business models and refine their strategies for long-term success.With a strong emphasis on data integrity, algorithmic precision, and industry-specific expertise, our project is poised to revolutionize the way we evaluate the prospects of startups, fostering a more dynamic and sustainable startup ecosystem that thrives on informed and strategic investments.<p>")

st.sidebar.image('pngwing.com (2).png')

data = pd.read_csv('startUp(1).csv')
st.write(data.head())

#--Select Your prefered Input Style
input_type = st.sidebar.radio("Select your prefered Input Style", ["Slider", "Number Input"])
if input_type == 'Slider':
    st.sidebar.header('Input Your Information')
    research = st.sidebar.slider("R&D Spend", data['R&D Spend'].min(), data['R&D Spend'].max())
    Admin = st.sidebar.slider("Administration", data['Administration'].min(), data['Administration'].max())
    mkt_spend = st.sidebar.slider("Marketing Spend", data['Marketing Spend'].min(), data['Marketing Spend'].max())
else:
    st.sidebar.header('Input Your Information')
    research = st.sidebar.number_input("R&D Spend", data['R&D Spend'].min(), data['R&D Spend'].max())
    Admin = st.sidebar.number_input("Administration", data['Administration'].min(), data['Administration'].max())
    mkt_spend = st.sidebar.number_input("Marketing Spend", data['Marketing Spend'].min(), data['Marketing Spend'].max())

st.header('Input Values')
# bring all the inputs into a dataframe 
input_variable = pd.DataFrame([{'R&D Spend': research, 'Administration': Admin, 'Marketing Spend': mkt_spend}])
st.write(input_variable)

#----------Standard scale the input variable.
# from sklearn.preprocessing import StandardScaler
# for i in input_variable.columns:
#    input_variable[i] = StandardScaler().fit_transform(input_variable[[i]])

#--------------------------- Load model
import pickle
model = pickle.load(open('StartUp_Model.pkl', "rb"))

# Insert containers separated into tabs:
tab1, tab2 = st.tabs(["Modelling", "Interpretation"])
with tab1:
    if st.button("Click me"):
        # profit = model.predict(input_variable)
        st.toast('Profitability Predicted')
        st.image('image.jpg', width = 200)
        st.success('Predicted . pls check the interpretation Tab for interpretation')




with tab2:
    st.subheader('Model Interpretation')
    profit = lin_reg.predict(input_variable)
    st.success(f'Predicted Profit is: {profit}')


    st.write(f"Profit = {lin_reg.intercept_.round(2)} + {lin_reg.coef_[0].round(2)} R&D Spend + {lin_reg.coef_[1].round(2)} Administration + {lin_reg.coef_[2].round(2)} Marketing Spend")

    st.markdown("<br>", unsafe_allow_html= True)

    st.markdown(f"- The expected Profit for a startup is {lin_reg.intercept_}")

    st.markdown(f"- For every additional 1 dollar spent on R&D Spend, the expected profit is expected to increase by ${lin_reg.coef_[0].round(2)}  ")

    st.markdown(f"- For every additional 1 dollar spent on Administration Expense, the expected profit is expected to decrease by ${lin_reg.coef_[1].round(2)}  ")

    st.markdown(f"- For every additional 1 dollar spent on Marketting Expense, the expected profit is expected to increase by ${lin_reg.coef_[2].round(2)}  ")






# st.markdown("<h4>Start Up Project Built By Emjay</h4>", unsafe_allow_html=True)
# st.header('START UP PROJECT')m
# st.subheader('Start Up Project Built By Emjay')