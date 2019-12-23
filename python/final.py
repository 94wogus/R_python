import FinanceDataReader as fdr

def split_print():
    print('------------------------------------------------------------------------------------------\n')

# KOSPI
index = 'KOSPI'
df_KOSPI = fdr.StockListing(index)
KOSPI = list(df_KOSPI['Symbol'])
print(df_KOSPI.head())
print("{} 상장 회사 수: {}".format(index, len(KOSPI)))

split_print()

# KOSDAQ
index = 'KOSDAQ'
df_KOSDAQ = fdr.StockListing(index)
KOSDAQ = list(df_KOSDAQ['Symbol'])
print(df_KOSDAQ.head())
print("{} 상장 회사 수: {}".format(index, len(KOSDAQ)))

split_print()

# 삼성전자 주가 그래프 얻기
SamSung_electronic = df_KOSPI[df_KOSPI['Name'] == "삼성전자"]
print(SamSung_electronic)
print('\n')
df_SamSung_electronic = fdr.DataReader(SamSung_electronic['Symbol'].values[0], '2018-01-01', '2018-03-30')
print(df_SamSung_electronic)
# 수익률은 (금일 종가 - 전일 종가) / 전일 종가 * 100 으로 dataframe의 change * 100이다.
change_SamSung = df_SamSung_electronic['Change'] * 100
print(change_SamSung.head())
#
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams["figure.figsize"] = (18, 12)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.color'] = 'r'
plt.rcParams['axes.grid'] = True
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
change_SamSung.plot()
plt.savefig('./static/change.png')
plt.show()

# KOSPI 기업의 연간 평균과 표준편차
import pandas as pd
import seaborn as sns
import numpy as np
import math
import os

if os.path.isfile("./KOSPI.csv"):
    KOSPI_mean_std = pd.read_csv("./KOSPI.csv")
else:
    KOSPI_mean_std = pd.DataFrame(columns=["mean", "std"])
    total = len(KOSPI)
    for x, s in enumerate(KOSPI):
        df = fdr.DataReader(s, '2018-01-01', '2018-12-31')
        mean = df['Change'].mean() * 100
        var = df['Change'].var()
        std = math.sqrt(var)
        KOSPI_mean_std.loc[s] = [mean, std]
        if x%50 == 0:
            print("{}/{} 개 완료".format(x, total))
    KOSPI_mean_std.dropna(how="any")
    KOSPI_mean_std.to_csv("./KOSPI.csv", mode='w')


sns.regplot(
    x=KOSPI_mean_std['mean'],
    y=KOSPI_mean_std['std'],
    fit_reg=True
)
plt.title('KOSPI Mean & Standard Deviation', fontsize=10)
plt.xlabel('Mean', fontsize=8)
plt.ylabel('Standard Deviation', fontsize=8)
plt.savefig("./static/KOSPI.png")
plt.show()

# KOSDAQ 기업의 연간 평균과 표준편차
if os.path.isfile("./KOSDAQ.csv"):
    KOSDAQ_mean_std = pd.read_csv("./KOSDAQ.csv")
else:
    KOSDAQ_mean_std = pd.DataFrame(columns=["mean", "std"])
    total = len(KOSDAQ)
    for x, s in enumerate(KOSDAQ):
        df = fdr.DataReader(s, '2018-01-01', '2018-12-31')
        mean = df['Change'].mean() * 100
        var = df['Change'].var()
        std = math.sqrt(var)
        KOSDAQ_mean_std.loc[s] = [mean, std]
        if x%50 == 0:
            print("{}/{} 개 완료".format(x, total))
    KOSDAQ_mean_std.dropna(how="any")
    KOSDAQ_mean_std.to_csv("./KOSDAQ.csv", mode='w')

sns.regplot(
    x=KOSDAQ_mean_std['mean'],
    y=KOSDAQ_mean_std['std'],
    fit_reg=True
)
plt.title('KOSDAQ Mean & Standard Deviation', fontsize=10)
plt.xlabel('Mean', fontsize=8)
plt.ylabel('Standard Deviation', fontsize=8)
plt.savefig("./static/KOSDAQ.png")
plt.show()

# Q7
#  KOSPI 기업 대상
import statsmodels.api as sm
if os.path.isfile("./KOSPI_Fitting.csv"):
    KOSPI_Fitting = pd.read_csv("./KOSPI_Fitting.csv")
else:
    Rf = 0.03/365
    df_KOSPI200 = fdr.DataReader('KS200', '2018')
    feature = df_KOSPI200['Change'] - Rf
    KOSPI_Fitting = pd.DataFrame(columns=["mean", "b1"])
    for x, s in enumerate(KOSPI):
        try:
            s_df = fdr.DataReader(s, '2018-01-01', '2018-12-31')
            target = s_df['Change'] - Rf
            df = pd.DataFrame({'target': target, "feature": feature})
            reg = sm.OLS.from_formula("target ~ feature", df).fit()
            mean = s_df['Change'].mean()
            print([mean, reg.params['feature']])
            KOSPI_Fitting.loc[s] = [mean, reg.params['feature']]
        except:
            continue
    KOSPI_Fitting.dropna(how="any")
    KOSPI_Fitting.to_csv("./KOSPI_Fitting.csv", mode='w')

sns.regplot(
    x=KOSPI_Fitting['mean'],
    y=KOSPI_Fitting['b1'],
    fit_reg=True
)
plt.title('KOSPI Regression Analysis', fontsize=10)
plt.xlabel('mean', fontsize=8)
plt.ylabel('b1', fontsize=8)
plt.savefig("./static/KOSPI_Fitting.png")
plt.show()

# KOSDAQ
if os.path.isfile("./KOSDAQ_Fitting.csv"):
    KOSDAQ_Fitting = pd.read_csv("./KOSDAQ_Fitting.csv")
else:
    Rf = 0.03/365
    df_KOSPI200 = fdr.DataReader('KS200', '2018')
    feature = df_KOSPI200['Change'] - Rf
    KOSDAQ_Fitting = pd.DataFrame(columns=["mean", "b1"])
    for x, s in enumerate(KOSDAQ):
        try:
            s_df = fdr.DataReader(s, '2018-01-01', '2018-12-31')
            target = s_df['Change'] - Rf
            df = pd.DataFrame({'target': target, "feature": feature})
            reg = sm.OLS.from_formula("target ~ feature", df).fit()
            mean = s_df['Change'].mean()
            print([mean, reg.params['feature']])
            KOSDAQ_Fitting.loc[s] = [mean, reg.params['feature']]
        except:
            continue
    KOSDAQ_Fitting.dropna(how="any")
    KOSDAQ_Fitting.to_csv("./KOSDAQ_Fitting.csv", mode='w')

sns.regplot(
    x=KOSDAQ_Fitting['mean'],
    y=KOSDAQ_Fitting['b1'],
    fit_reg=True
)
plt.title('KOSDAQ Regression Analysis', fontsize=10)
plt.xlabel('mean', fontsize=8)
plt.ylabel('b1', fontsize=8)
plt.savefig("./static/KOSDAQ_Fitting.png")
plt.show()