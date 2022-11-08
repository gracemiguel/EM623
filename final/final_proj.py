import pandas as pd


df15 = pd.read_csv("2015.csv")
df15["Year"] = 2015
df15.rename(columns={"Happiness score": "Score", "Economy: GDP per capita": "GDP per capita","Health (Life Expectancy)": "Healthy life expectancy", "Family": "Social Support", "Trust(Government Corruption)" :"Perceptions of corruption" })
df15.drop(columns=["Dystopia Residual"])
df16 = pd.read_csv("2016.csv")
df16.rename(columns={"Happiness score": "Score", "Economy: GDP per capita": "GDP per capita","Health (Life Expectancy)": "Healthy life expectancy", "Family": "Social Support", "Trust(Government Corruption)" :"Perceptions of corruption" })
df16["Year"] = 2016
df16.drop(columns=["Dystopia Residual"])
df17 = pd.read_csv("2017.csv")
df17.rename(columns={"Happiness.rank": "Happiness rank", "Happinesss.Score": "Score","Economy..GDP.per.Capita.": "Economy: GDP per capita", "Family": "Social Support", "Health..Life.Expectancy.": "Healthy life expectancy", "Trust..Government.Corruption.": "Perceptions of corruption" })
df17.drop(columns=["Dystopia.Residual"])
df17["Year"] = 2017
df18 = pd.read_csv("2018.csv")
df18["Year"] = 2018
df19 = pd.read_csv("2019.csv")
df19.rename(columns={"Freedom to make life choices": "Freedom"})
df19["Year"] = 2019
df20 = pd.read_csv("2020.csv")
df20.rename(columns={"Freedom to make life choices": "Freedom"})
df20["Year"] = 2020
df21 = pd.read_csv("2021.csv")
df21["Year"] = 2021
df21.rename(columns={"Ladder score": "Score", "Country name": "Country or region", "Logged GDP per capita": "GDP per capita"})

#Overall rank	Country or region	Score	GDP per capita	Social support	Healthy life expectancy	Freedom to make life choices	Generosity	Perceptions of corruption
# df21[["Country or region", "Score", ]]

df_1516 = df15.join(df16.set_index("Country"), on=("Country"))
print(df_1516)
