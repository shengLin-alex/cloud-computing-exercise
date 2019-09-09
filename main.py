import sys
import pymysql
import pandas as pd
import prediction

# 建立 db 連線
conn = pymysql.connect("140.124.42.180", "guest", "guest", "cloud_computing")

# 取資料並且將price 轉換為 int
df = pd.read_sql("SELECT * FROM `practice` WHERE `price` != '' AND `price` IS NOT NULL AND `price` != '-'", con=conn)
df['price'] = df['price'].str.replace(',', '')
df[['price']].astype(int)

# 取欄位
cols = ['bus_station', 'school', 'hospital', 'subway_station', 'parking', 'movie_theater', 'price', 'population']
x = df[cols]
y = df[['branch_o_int']]

if len(sys.argv) < 2:
    print('usage: python main.py [prediction option] option: dt, gnb. ex: python main.py gnb')
else:
    option = sys.argv[1]
    if option == 'dt':
        # 測試決策樹準確率
        dt_accu = prediction.Prediction(x=x, y=y, prediction_strategy=prediction.dtree_prediction).get_accuracy()
        print('accuracy(Decision Tree) : ', dt_accu)
    elif option == 'gnb':
        # 高斯單純貝氏分類器
        gnb_accu = prediction.Prediction(x=x, y=y, prediction_strategy=prediction.gaussian_nb_prediction).get_accuracy()
        print('accuracy(Gaussion Naive Bayes) : ', gnb_accu)
    elif option == 'rf':
        # 隨機森林
        rf_accu = prediction.Prediction(x=x, y=y, prediction_strategy=prediction.random_forest_prediction).get_accuracy()
        print('accuracy(Random Forest) : ', rf_accu)
