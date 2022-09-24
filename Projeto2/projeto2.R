# Projeto 2 - Machine Learning na Segurança do Trabalho Prevendo a Eficiência de Extintores de Incêndio
# Gabriel Araujo Carlos

setwd("C:/Users/Yoh/Desktop/R/Projeto2")
getwd()

# Pacotes
library(dplyr)
library(ggplot2)
require(randomForest)
library(tidyr)
require(plyr)

# Carrega o dataset
dados <- read.csv('dataset.csv',header=T,sep=";",na.strings=c(""," ","NA"))

# Visualiza os dados
View(dados)
str(dados)

# Verificando ocorrência de valores NA
colSums(is.na(dados))

#Separando o dataframe entre combustiveis liquidos e lgp
dadosLpg = dados[(dados$FUEL=="lpg"),]
dadosLiquidFuels  <- dplyr::anti_join(dados, dadosLpg, by = 'FUEL')
unique(dadosLiquidFuels$FUEL)
unique(dadosLpg$FUEL)

#Transformando os valor indicados de size para os tamanhos de fato indicados no dicionário.
dadosLiquidFuels["SIZE"][dadosLiquidFuels["SIZE"] == "1"] <- 7
dadosLiquidFuels["SIZE"][dadosLiquidFuels["SIZE"] == "2"] <- 12
dadosLiquidFuels["SIZE"][dadosLiquidFuels["SIZE"] == "3"] <- 14
dadosLiquidFuels["SIZE"][dadosLiquidFuels["SIZE"] == "4"] <- 16
dadosLiquidFuels["SIZE"][dadosLiquidFuels["SIZE"] == "5"] <- 20

#Transformar as variaveis
dadosLiquidFuels <- dadosLiquidFuels %>% 
  mutate(FUEL = as.factor(FUEL)) %>% 
  mutate(AIRFLOW = as.numeric(gsub(",", ".",AIRFLOW)))%>%
  mutate(STATUS = as.factor(STATUS))

View(dadosLiquidFuels)
str(dadosLiquidFuels)