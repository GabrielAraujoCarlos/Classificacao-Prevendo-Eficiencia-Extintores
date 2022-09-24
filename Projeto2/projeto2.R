# Projeto 2 - Machine Learning na Segurança do Trabalho Prevendo a Eficiência de Extintores de Incêndio
# Gabriel Araujo Carlos

setwd("C:/Users/Yoh/Documents/GitHub/projeto2DSA/Projeto2")
getwd()

# Pacotes
library(dplyr)
library(ggplot2)
require(randomForest)
library(tidyr)
require(plyr)
library(e1071)
library(rpart)
library(MASS)

# Carrega o dataset
dados <- read.csv('dataset.csv',header=T,sep=";",na.strings=c(""," ","NA"))

# Visualiza os dados
View(dados)
str(dados)

# Verificando ocorrência de valores NA
colSums(is.na(dados))


#### Verifiquei no dicionario de dados que o valor de tamanho das observacoes LPG significam coisas diferentes
#### dos demais, assim decidi separar o dataset entre combustiveis liquidos e LPG.
#### Irei entregar 2 modelos diferentes para combustiveis liquidos e LPG.


#Separando o dataframe entre combustiveis liquidos e lpg
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
dadosLiquidFuels["STATUS"][dadosLiquidFuels["STATUS"] == 0] <- "Falha"
dadosLiquidFuels["STATUS"][dadosLiquidFuels["STATUS"] == 1] <- "Sucesso"

#Transformar as variaveis
dadosLiquidFuels <- dadosLiquidFuels %>% 
  mutate(FUEL = as.factor(FUEL)) %>% 
  mutate(AIRFLOW = as.numeric(gsub(",", ".",AIRFLOW)))%>%
  mutate(STATUS = as.factor(STATUS))

View(dadosLiquidFuels)
str(dadosLiquidFuels)

#Analise exploratoria da distribuicao das variaveis
boxplot(dadosLiquidFuels$SIZE)
boxplot(dadosLiquidFuels$DISTANCE)
boxplot(dadosLiquidFuels$DESIBEL)
boxplot(dadosLiquidFuels$FREQUENCY)
#### As variaveis numericas nao apresentam outliers


#Analise exploratoria da distribuicao das variaveis
table(dadosLiquidFuels$FUEL)
table(dadosLiquidFuels$STATUS)
#### A variavel categorica de combustivel indica que os 3 combustiveis liqueidos estao igualmente contemplados
#### A variavel alvo esta euqilibrada entre valores positivos e negativos, assim nao se fazendo necessario
#### o balanceamento de variaveis


#Split entre treino e teste
set.seed(1)
dadosLiquidFuels$id <- 1:nrow(dadosLiquidFuels)
train1 <- dadosLiquidFuels %>% dplyr::sample_frac(0.7)
test1  <- dplyr::anti_join(dadosLiquidFuels, train1, by = 'id')
train1$id <- NULL
test1$id <- NULL
dadosLiquidFuels$id <- NULL

#Criacao do modelo KNN
dados_treino_labels <- train1[, 1]
dados_teste_labels <- test1[, 1]
#modelo_knn_v1 <- knn(train = train,test = test,cl = dados_treino_labels,k = 21)
#### O modelo KNN apresenta erro pois o algoritmo espera que todas as variaveis preditoras sejam numericas.
#### Assim, decidi nao utilizar este algoritmo pois nao vejo valor em converter a variavel de combustiveis 
#### em numerica.

#Criacao do modelo SVM
modelo_svm_v1 <- svm(STATUS ~ .,data = train1,type = 'C-classification',kernel = 'radial')
# Previsões nos dados de teste
pred_test <- predict(modelo_svm_v1, test1)
# Percentual de previsões corretas com dataset de teste
mean(pred_test == test1$STATUS)
# Verificando a confusion Matrix
table(pred_test, test1$STATUS)

# Criando o modelo Random Forest
modelo_rf_v1 = rpart(STATUS ~ ., data = train1, control = rpart.control(cp = .0005)) 
# Previsões nos dados de teste
tree_pred = predict(modelo_rf_v1, test1, type='class')
# Percentual de previsões corretas com dataset de teste
mean(tree_pred==test1$STATUS)
# Verificando a confusion Matrix
table(tree_pred, test1$STATUS)

# Criando o modelo Naive Bayes
modelo_nb_v1 <- naiveBayes(STATUS ~ ., data = train1)
# Previsões nos dados de teste
y_pred <- predict(modelo_nb_v1, newdata = test1)
# Percentual de previsões corretas com dataset de teste
mean(y_pred==test1$STATUS)
# Verificando a confusion Matrix
table(y_pred, test1$STATUS)

#### SVM = (0.9491011 & 141) / RandomForest = (0.9471518 & 109) / NaiveBayes = (0.8880225 & 333)
#### Irei trabalhar com estes 3 algoritmos, tentando maximizar a precisão e anular falsos positivos.
#### Falso positivos são resultados que prestarei mais atenção pois indicam que um extintor funciona
#### quando ele nao funciona.


# padronização dos dados
normalizar <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
dados_norm <- as.data.frame(lapply(dadosLiquidFuels[-c(2,7)], normalizar))
dados_norm = cbind(dados_norm,dadosLiquidFuels[c(2,7)])
dados_norm$id <- 1:nrow(dados_norm)
train2 <- dados_norm %>% dplyr::sample_frac(0.7)
test2 <- dplyr::anti_join(dados_norm, train2, by = 'id')
train2$id <- NULL
test2$id <- NULL

#Criacao de nova versao dos modelos
modelo_svm_v2 <- svm(STATUS ~ .,data = train2,type = 'C-classification',kernel = 'radial')
pred_test <- predict(modelo_svm_v2, test2)
mean(pred_test == test2$STATUS)
table(pred_test, test2$STATUS)

modelo_rf_v2 = rpart(STATUS ~ ., data = train2, control = rpart.control(cp = .0005)) 
tree_pred = predict(modelo_rf_v2, test2, type='class')
mean(tree_pred==test2$STATUS)
table(tree_pred, test2$STATUS)

modelo_nb_v2 <- naiveBayes(STATUS ~ ., data = train2)
y_pred <- predict(modelo_nb_v2, newdata = test2)
mean(y_pred==test2$STATUS)
table(y_pred, test2$STATUS)


#### SVM = (0.9484514 & 148) / RandomForest = (0.9434698 & 139) / NaiveBayes = (0.8797921 & 358)
#### As versoes do modelo com dados padronizados apresentaram performance ligeiramente menor que o original
#### e aumentou o numero de Falso Positivos


# Normalização dos dados
dados_z <- as.data.frame(scale(dadosLiquidFuels[-c(2,7)]))
dados_z = cbind(dados_z,dadosLiquidFuels[c(2,7)])
dados_norm$id <- 1:nrow(dados_norm)
train3 <- dados_norm %>% dplyr::sample_frac(0.7)
test3 <- dplyr::anti_join(dados_norm, train3, by = 'id')
train3$id <- NULL
test3$id <- NULL

#Criacao de nova versao dos modelos
modelo_svm_v3 <- svm(STATUS ~ .,data = train3,type = 'C-classification',kernel = 'radial')
pred_test <- predict(modelo_svm_v3, test3)
mean(pred_test == test3$STATUS)
table(pred_test, test3$STATUS)

modelo_rf_v3 = rpart(STATUS ~ ., data = train3, control = rpart.control(cp = .0005)) 
tree_pred = predict(modelo_rf_v3, test3, type='class')
mean(tree_pred==test3$STATUS)
table(tree_pred, test3$STATUS)

modelo_nb_v3 <- naiveBayes(STATUS ~ ., data = train3)
y_pred <- predict(modelo_nb_v3, newdata = test3)
mean(y_pred==test3$STATUS)
table(y_pred, test3$STATUS)


#### SVM = (0.9517002 & 126) / RandomForest = (0.9510505 & 141) / NaiveBayes = (0.8793589 & 343)
#### As versoes do modelo com dados normalizados apresentaram performance ligeiramente maior que o original
#### e aumentou o numero de Falso Positivos de 2 modelos.
#### Neste caso eu irei tentar otimizar hiperparametros com o modelo de SVM3 e RandomForest1 por 
#### apresentarem as melhores performances


#SVM = (0.9517002 & 126)
#modelo_svm_v3.1 <- svm(STATUS ~ .,data = train3,type = 'C-classification',kernel = 'linear')
#modelo_svm_v3.1 <- svm(STATUS ~ .,data = train3,type = 'C-classification',kernel = 'polynomial')
#modelo_svm_v3.1 <- svm(STATUS ~ .,data = train3,type = 'C-classification',kernel = 'sigmoid')
#modelo_svm_v3.1 <- svm(STATUS ~ .,data = train3,type = 'one-classification',kernel = 'radial')
#RandomForest = (0.9471518 & 109)
#modelo_rf_v1.1 = rpart(STATUS ~ ., data = train1, control = rpart.control(cp = .005))
#modelo_rf_v1.1 = rpart(STATUS ~ ., data = train1, control = rpart.control(cp = .00005))

#### Nenhuma das alteracoes em hiperparametros que testei melhorou a performance
#### Por nao compreender outras possiveis mudancas em hiperparametros, decidi por continuar com outra
#### estrategia para a otimizacao de modelo.
#### Suponho que se o modelo for apresentado a mais dados de negativo do que de positivos, ele ira
#### diminuir os falso positivos.


table(dadosLiquidFuels$STATUS)
# Como o pacote da funcao SMOTE(aprendida no curso) foi descontinuado,irei diminuir a quantidade da 
# variavel positiva
