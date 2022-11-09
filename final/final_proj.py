import pandas as pd


df15 = pd.read_csv("2015.csv")
df15["Year"] = 2015
df15 = df15.rename(columns={"Happiness Score": "Score", "Health (Life Expectancy)": "Healthy life expectancy", "Family": "Social support", "Trust (Government Corruption)" :"Perceptions of corruption", "Economy (GDP per Capita)": "Economy: GDP per capita", "Freedom to make life choices": "Freedom", "Country": "Country or region"})
df15 = df15.drop(columns=["Dystopia Residual", "Region", "Standard Error"])
df16 = pd.read_csv("2016.csv")
df16 = df16.rename(columns={"Happiness Score": "Score", "Health (Life Expectancy)": "Healthy life expectancy", "Family": "Social support", "Trust (Government Corruption)" :"Perceptions of corruption",  "Economy (GDP per Capita)": "Economy: GDP per capita", "Freedom to make life choices": "Freedom", "Country": "Country or region"})
df16 = df16.drop(columns=["Dystopia Residual", "Region", "Lower Confidence Interval", "Upper Confidence Interval"])
df16["Year"] = 2016
df17 = pd.read_csv("2017.csv")
df17 = df17.rename(columns={"Happiness.Rank": "Happiness Rank", "Happiness.Score": "Score","Economy..GDP.per.Capita.": "Economy: GDP per capita", "Family": "Social support", "Health..Life.Expectancy.": "Healthy life expectancy", "Trust..Government.Corruption.": "Perceptions of corruption", "Country": "Country or region" })
df17 = df17.drop(columns=["Dystopia.Residual", "Whisker.high", "Whisker.low"])
df17["Year"] = 2017
df18 = pd.read_csv("2018.csv")
df18 = df18.rename(columns={"Overall rank": "Happiness Rank", "GDP per capita": "Economy: GDP per capita", "Freedom to make life choices": "Freedom"})
df18["Year"] = 2018
df19 = pd.read_csv("2019.csv")
df19 = df19.rename(columns={"Freedom to make life choices": "Freedom", "Overall rank": "Happiness Rank", "GDP per capita": "Economy: GDP per capita"})
df19["Year"] = 2019
df20 = pd.read_csv("2020.csv")
df20 = df20.rename(columns={"Freedom to make life choices": "Freedom", "Country name": "Country or region", "Ladder score": "Score", "Logged GDP per capita": "Economy: GDP per capita"})
df20["Year"] = 2020
df20 = df20[["Country or region", "Score", "Economy: GDP per capita", "Healthy life expectancy", "Social support", "Perceptions of corruption", "Year", "Freedom"]]
df21 = pd.read_csv("2021.csv")
df21["Year"] = 2021
df21 = df21.rename(columns={"Ladder score": "Score", "Country name": "Country or region", "Logged GDP per capita": "Economy: GDP per capita", "Freedom to make life choices": "Freedom"})


#Overall rank	Country or region	Score	GDP per capita	Social support	Healthy life expectancy	Freedom to make life choices	Generosity	Perceptions of corruption
df21_ = df21[["Country or region", "Score", "Economy: GDP per capita", "Healthy life expectancy", "Social support", "Perceptions of corruption", "Generosity", "Freedom", "Year" ]]

df_total = pd.concat([df15, df16, df17, df18, df19, df20, df21_], axis=0)

#ONLY COMPUTE THIS ONCE TO DOWNLOAD THE COMBINED FILE
# df_total.to_csv("world_happiness_15_21.csv")


print(df_total.info())
print(df_total.head())

