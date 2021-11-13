import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


# Loading data
oced_bill = pd.read_csv("oecd_bli_2015.csv", thousands=",")
gdp_per_capita = pd.read_csv(
    "gdp_per_capita.csv", thousands=",", sep="\t", encoding="latin1", na_values="n/a"
)


# Taken directly from book's github
def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(
        left=oecd_bli, right=gdp_per_capita, left_index=True, right_index=True
    )
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", "Life satisfaction"]].iloc[
        keep_indices
    ]


country_stat = prepare_country_stats(oced_bill, gdp_per_capita)
X = country_stat["GDP per capita"].values.reshape(
    len(country_stat["GDP per capita"]), -1
)
y = country_stat["Life satisfaction"].values.reshape(
    len(country_stat["Life satisfaction"]), -1
)

country_stat.plot(kind="scatter", x="GDP per capita", y="Life satisfaction")


model_based = LinearRegression()
model_based.fit(X, y)


X_new = [[22587]]
print(model_based.predict(X_new))


model_based.coef_, model_based.intercept_


instance_based = KNeighborsRegressor(n_neighbors=3)
instance_based.fit(X, y)


print(instance_based.predict(X_new))


# According to instance based
country_stat[
    (country_stat["Life satisfaction"] >= 5.7)
    & (country_stat["Life satisfaction"] <= 5.8)
]


# According to Model Based
country_stat[
    (country_stat["Life satisfaction"] >= 5.9)
    & (country_stat["Life satisfaction"] <= 6.0)
]
