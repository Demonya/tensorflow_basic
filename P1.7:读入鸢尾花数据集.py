from sklearn.datasets import load_iris
import pandas as pd


x_data = load_iris().data
y_data = load_iris().target

print("x_data from dataset: \n", x_data)
print("y_data from dataset: \n", y_data)
x_data = pd.DataFrame(x_data, columns = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'])
# pd.set_option('display.unicode.east_asian_width', True) #设置列明对齐
print("x_data add index: \n", x_data)

x_data['类别'] = y_data
print("x_data add a columns: \n", x_data)
