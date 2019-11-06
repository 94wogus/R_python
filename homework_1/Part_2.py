from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

os.system("cls")

# part2
# answer 5)
csv_path = "./docs/boston_csv.csv"
# missing values 설정 후 적용
missing_values = ["n/a", "na", "--"]
raw_boston = pd.read_csv(csv_path, na_values=missing_values)
# 결측치 제거
boston = raw_boston.dropna(axis=0)
print(boston)

# answer 6)
# describe 변수 별 요약 통계 구하기
print(boston.describe())
# 상관관계 구하기
# colums name list
from scipy import stats
colum_names = [cn for cn in boston.columns]
boston_dict = {}
# stats로 상관 변수 구하고 dict형태로 저장
for cn in colum_names:
    clist = []
    for cn2 in colum_names:
        coef, p_value = stats.pearsonr(boston[cn], boston[cn2])
        clist.append(coef)
    boston_dict[cn] = clist
# dict를 활용하여 DataFrame 생성
coef_boston = pd.DataFrame(boston_dict)
# int 인덱스를 다시 colum에 해당하는 값으로 매칭
for i, cn in enumerate(colum_names):
    coef_boston = coef_boston.rename(index={i: cn})
# figure setting, save, print
plt.figure(figsize=(12,10))
ax = sns.heatmap(coef_boston, annot=True, fmt='.2f', linewidths=.5, annot_kws={"size": 7}, yticklabels=1)
plt.title("Heatmap of Boston", fontsize=10)
plt.ylim(-0.5, len(colum_names))
plt.savefig('./boston.png', dpi=298)
plt.show()

# answer 7)
# 독립변수
X = boston[['LSTAT']]
# 종속변수
Y = boston[['MEDV']]
# train set, test set split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

# Fit the data(train the model)
regression_model = LinearRegression()
regression_model.fit(X_train, Y_train)

# get R2
y_train_pred = regression_model.predict(X_train)
mse = mean_squared_error(Y_train, y_train_pred)
r2 = round(regression_model.score(X_train, Y_train), 2)

print("The model performance for training set")
print("--------------------------------------")
print("Estimated coefficients: {}".format(regression_model.coef_[0][0]))
print('mean squared error: {}'.format(mse))
print('R2 score: {}'.format(r2))


y_test_pred = regression_model.predict(X_test)
mse = mean_squared_error(Y_test, y_test_pred)
r2 = round(regression_model.score(X_test, Y_test), 2)
print("The model performance for testing set")
print("--------------------------------------")
print('mean squared error: {}'.format(mse))
print('R2 score: {}'.format(r2))
print("\n")
#
# # answer 8)
# 독립변수
X = boston[['LSTAT', 'TAX']]
# 종속변수
Y = boston[['MEDV']]
# train set, test set split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

# Fit the data(train the model)
regression_model = LinearRegression()
regression_model.fit(X_train, Y_train)

# get R2
y_train_pred = regression_model.predict(X_train)
mse = mean_squared_error(Y_train, y_train_pred)
r2 = round(regression_model.score(X_train, Y_train), 2)

print("The model performance for training set")
print("--------------------------------------")
print("Estimated coefficients: {}".format(regression_model.coef_[0][0]))
print('mean squared error: {}'.format(mse))
print('R2 score: {}'.format(r2))

y_test_pred = regression_model.predict(X_test)
mse = mean_squared_error(Y_test, y_test_pred)
r2 = round(regression_model.score(X_test, Y_test), 2)
print("The model performance for testing set")
print("--------------------------------------")
print('mean squared error: {}'.format(mse))
print('R2 score: {}'.format(r2))

