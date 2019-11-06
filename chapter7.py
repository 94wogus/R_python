import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def movies():
    f_path = "./docs/titles.csv"
    t = pd.read_csv(f_path)
    after85 = t[t['year'] > 1985]

    movies = t[ (t['year'] >= 1990) & (t['year'] < 2000) ]
    print(movies)

    macbeth = t[t['title'] == 'Macbeth']
    print(macbeth.head().sort_index(), "\n")
    print(t[t['title'] == 'Maa'], "\n")
    print(t[t['title'].str.startswith("Maa ")].head(3), "\n")
    print(t['year'].value_counts().sort_index(), "\n")

    p = t['year'].value_counts()
    # p.plot()
    # plt.show()
    p.sort_index().plot()
    plt.show()

def casts():
    casts = pd.read_csv("./docs/cast.csv", index_col=None)
    # print(casts.head())
    c = casts
    cg = c.groupby(['year']).size()
    # print(cg)
    # cg.plot()
    # plt.show()

    cf = c[c['name'] == 'Aaron Abrams']
    print(cf.groupby(['year']).size().head())
    print(c.groupby(['year']).n.max().head())
    print(c.groupby([c['year'] // 10 * 10, 'type']).size().head(8))

    c_decade = c.groupby(['type', c['year'] // 10 * 10]).size()
    print(c_decade)
    c_decade.unstack().plot(kind='bar')
    c_decade.plot()
    plt.show()


def df():
    df = pd.read_csv("./docs/property_data.csv")
    # print(df.head())
    # print(df['ST_NUM'])
    # print(df['ST_NUM'].isnull())
    # print(df['NUM_BEDROOMS'])
    # print(df['NUM_BEDROOMS'].isnull())
    cnt = 0
    for row in df['OWN_OCCUPIED']:
        try:
            int(row)
            df.loc[cnt, 'OWN_OCCUPIED'] = np.nan
        except ValueError:
            pass
        cnt += 1

    print(df['OWN_OCCUPIED'])


df()