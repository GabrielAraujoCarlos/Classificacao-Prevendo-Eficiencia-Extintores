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
#### A variavel categorica de combustivel indica que os 3 combustiveis liquidos estao igualmente contemplados
#### A variavel alvo esta euqilibrada entre valores positivos e negativos, assim nao se fazendo necessario
#### o balanceamento de variaveis


#Split entre treino e teste
dadosLiquidFuels$id <- 1:nrow(dadosLiquidFuels)
set.seed(1)
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
set.seed(1)
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


#### SVM = (0.9491011 & 141) / RandomForest = (0.9471518 & 109) / NaiveBayes = (0.8880225 & 333)
#### As versoes do modelo com dados padronizados apresentaram a mesma performance


# Normalização dos dados
dados_z <- as.data.frame(scale(dadosLiquidFuels[-c(2,7)]))
dados_z = cbind(dados_z,dadosLiquidFuels[c(2,7)])
dados_z$id <- 1:nrow(dados_z)
set.seed(1)
train3 <- dados_z %>% dplyr::sample_frac(0.7)
test3 <- dplyr::anti_join(dados_z, train3, by = 'id')
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


#### SVM = (0.9491011 & 141) / RandomForest = (0.9471518 & 109) / NaiveBayes = (0.8880225 & 333)
#### As versoes do modelo com dados padronizados apresentaram a mesma performance
#### Concluo que as padronizacao e normalizacao dos dados nao afetou o resultado deste dataset.
#### Neste caso eu irei tentar otimizar hiperparametros com o modelo de SVM1 e RandomForest1 que
#### foram criados com o dataset original


#modelo_svm_v1.1 <- svm(STATUS ~ .,data = train1,type = 'C-classification',kernel = 'linear')
#modelo_svm_v1.1 <- svm(STATUS ~ .,data = train1,type = 'C-classification',kernel = 'polynomial')
#modelo_svm_v1.1 <- svm(STATUS ~ .,data = train1,type = 'C-classification',kernel = 'sigmoid')
#modelo_svm_v1.1 <- svm(STATUS ~ .,data = train1,type = 'one-classification',kernel = 'radial')
#modelo_rf_v1.1 = rpart(STATUS ~ ., data = train1, control = rpart.control(cp = .005))
#modelo_rf_v1.1 = rpart(STATUS ~ ., data = train1, control = rpart.control(cp = .00005))


#### Nenhuma das alteracoes em hiperparametros que testei melhorou a performance
#### Por nao compreender outras possiveis mudancas em hiperparametros, decidi por continuar com outra
#### estrategia para a otimizacao de modelo.
#### Suponho que se o modelo for apresentado a mais dados de negativo do que de positivos, ele ira
#### diminuir os resultados falso positivos.


table(dadosLiquidFuels$STATUS)
# Como o pacote da funcao SMOTE(aprendida no curso) foi descontinuado,irei diminuir a quantidade da 
# variavel negativa
dadosDesbalanciados = dadosLiquidFuels[dadosLiquidFuels$STATUS == "Falha",]
auxiliar = dadosLiquidFuels[dadosLiquidFuels$STATUS == "Sucesso",]
auxiliar <- auxiliar %>% dplyr::sample_frac(0.9)
dadosDesbalanciados = rbind(dadosDesbalanciados,auxiliar)

dadosDesbalanciados$id <- 1:nrow(dadosDesbalanciados)
set.seed(1)
train1.1 <- dadosDesbalanciados %>% dplyr::sample_frac(0.7)
test1.1  <- dplyr::anti_join(dadosDesbalanciados, train1.1, by = 'id')
train1.1$id <- NULL
test1.1$id <- NULL

modelo_svm_v1.1 <- svm(STATUS ~ .,data = train1.1,type = 'C-classification',kernel = 'radial')
pred_test <- predict(modelo_svm_v1.1, test1.1)
mean(pred_test == test1.1$STATUS)
table(pred_test, test1.1$STATUS)

modelo_rf_v1.1 = rpart(STATUS ~ ., data = train1.1, control = rpart.control(cp = .0005)) 
tree_pred = predict(modelo_rf_v1.1, test1.1, type='class')
mean(tree_pred==test1.1$STATUS)
table(tree_pred, test1.1$STATUS)

#### SVM = (0.9453427 & 144) / RandomForest = (0.9453427 & 139)
#### A minha suposicao nao se confirmou, assim os modelos criados com variavel desbalanceada apresentou
#### desempenho pior.
#### Testarei alterar a taxa de split entre treino e teste
#### Como irei alterar a proporcao de valores para teste, irei utilizar o valor False Positive Rate para
#### avaliar(quanto menor melhor).
#### SVM = (0.9491011 & 141) / RandomForest = (0.9471518 & 109)
FPR <- function(x) {
  return (x[1,2]/ (x[1,2] + x[2,2]))
}
pred_test <- predict(modelo_svm_v1, test1)
tree_pred = predict(modelo_rf_v1, test1, type='class')
FPR(table(pred_test, test1$STATUS))
FPR(table(tree_pred, test1$STATUS))
#### SVM = (0.9491011 & 0.06197802) / RandomForest = (0.9471518 & 0.04791209)

dadosLiquidFuels$id <- 1:nrow(dadosLiquidFuels)
set.seed(1)
train1.2 <- dadosLiquidFuels %>% dplyr::sample_frac(0.65)
test1.2 <- dplyr::anti_join(dadosLiquidFuels, train1.2, by = 'id')
train1.2$id <- NULL
test1.2$id <- NULL
modelo_svm_v1.2 <- svm(STATUS ~ .,data = train1.2,type = 'C-classification',kernel = 'radial')
pred_test <- predict(modelo_svm_v1.2, test1.2)
mean(pred_test == test1.2$STATUS)
FPR(table(pred_test, test1.2$STATUS))
modelo_rf_v1.2 = rpart(STATUS ~ ., data = train1.2, control = rpart.control(cp = .0005)) 
tree_pred = predict(modelo_rf_v1.2, test1.2, type='class')
mean(tree_pred==test1.2$STATUS)
FPR(table(tree_pred, test1.2$STATUS))
#### SVM = (0.948756 & 0.06029579) / RandomForest = (0.947085 & 0.05119454)
set.seed(1)
train1.3 <- dadosLiquidFuels %>% dplyr::sample_frac(0.75)
test1.3 <- dplyr::anti_join(dadosLiquidFuels, train1.3, by = 'id')
train1.3$id <- NULL
test1.3$id <- NULL
modelo_svm_v1.3 <- svm(STATUS ~ .,data = train1.3,type = 'C-classification',kernel = 'radial')
pred_test <- predict(modelo_svm_v1.3, test1.3)
mean(pred_test == test1.3$STATUS)
FPR(table(pred_test, test1.3$STATUS))
modelo_rf_v1.3 = rpart(STATUS ~ ., data = train1.3, control = rpart.control(cp = .0005)) 
tree_pred = predict(modelo_rf_v1.3, test1.3, type='class')
mean(tree_pred==test1.3$STATUS)
FPR(table(tree_pred, test1.3$STATUS))
#### SVM = (0.9503638 & 0.05937993) / RandomForest = (0.9540021 & 0.05307409)
set.seed(1)
train1.4 <- dadosLiquidFuels %>% dplyr::sample_frac(0.8)
test1.4 <- dplyr::anti_join(dadosLiquidFuels, train1.4, by = 'id')
train1.4$id <- NULL
test1.4$id <- NULL
modelo_svm_v1.4 <- svm(STATUS ~ .,data = train1.4,type = 'C-classification',kernel = 'radial')
pred_test <- predict(modelo_svm_v1.4, test1.4)
mean(pred_test == test1.4$STATUS)
FPR(table(pred_test, test1.4$STATUS))
modelo_rf_v1.4 = rpart(STATUS ~ ., data = train1.4, control = rpart.control(cp = .0005)) 
tree_pred = predict(modelo_rf_v1.4, test1.4, type='class')
mean(tree_pred==test1.4$STATUS)
FPR(table(tree_pred, test1.4$STATUS))
#### SVM = (0.9519168 & 0.0571241) / RandomForest = (0.9483431 & 0.0571241)
set.seed(1)
train1.5 <- dadosLiquidFuels %>% dplyr::sample_frac(0.85)
test1.5 <- dplyr::anti_join(dadosLiquidFuels, train1.5, by = 'id')
train1.5$id <- NULL
test1.5$id <- NULL
modelo_svm_v1.5 <- svm(STATUS ~ .,data = train1.5,type = 'C-classification',kernel = 'radial')
pred_test <- predict(modelo_svm_v1.5, test1.5)
mean(pred_test == test1.5$STATUS)
FPR(table(pred_test, test1.5$STATUS))
modelo_rf_v1.5 = rpart(STATUS ~ ., data = train1.5, control = rpart.control(cp = .0005)) 
tree_pred = predict(modelo_rf_v1.5, test1.5, type='class')
mean(tree_pred==test1.5$STATUS)
FPR(table(tree_pred, test1.5$STATUS))
#### SVM = (0.952773 & 0.05487805) / RandomForest = (0.9475737 & 0.05400697)
dadosLiquidFuels$id <- NULL


#### A taxa de falso positivos diminui para os modelo de SVM porem aumentou para os modelo de Random Forest
#### O modelo com a melhor acuracia e menor FPR foi Random Forest1
summary(modelo_rf_v1)
#### que apresenta 0.9471518 de acuracia e 0.0479 de FPR(109 falsos positivos)

#### Agora inicio a parte do combustivel LPG


# Transformacoes similares
dadosLpg["SIZE"][dadosLpg["SIZE"] == "6"] <- "Half"
dadosLpg["SIZE"][dadosLpg["SIZE"] == "7"] <- "Full"
dadosLpg["STATUS"][dadosLpg["STATUS"] == 0] <- "Falha"
dadosLpg["STATUS"][dadosLpg["STATUS"] == 1] <- "Sucesso"
dadosLpg$FUEL = NULL
dadosLpg <- dadosLpg %>% 
  mutate(SIZE = as.factor(SIZE)) %>% 
  mutate(AIRFLOW = as.numeric(gsub(",", ".",AIRFLOW)))%>%
  mutate(STATUS = as.factor(STATUS))
boxplot(dadosLpg$DISTANCE)
boxplot(dadosLpg$DESIBEL)
boxplot(dadosLpg$FREQUENCY)
table(dadosLpg$SIZE)
table(dadosLpg$STATUS)
#### Tambem sem outliers nas variaveis numericas
#### As classes estao desbalanceadas, com Sucesso sendo ~56%
#### Testarei igual aos modelo de combustiveis liquidos utilizando a FPR


dadosLpg$id <- 1:nrow(dadosLpg)
set.seed(1)
trainlpg <- dadosLpg %>% dplyr::sample_frac(0.7)
testlpg  <- dplyr::anti_join(dadosLpg, trainlpg, by = 'id')
trainlpg$id <- NULL
testlpg$id <- NULL
dadosLpg$id <- NULL
modelo_svm_lpg_v1 <- svm(STATUS ~ .,data = trainlpg,type = 'C-classification',kernel = 'radial')
pred_test <- predict(modelo_svm_lpg_v1, testlpg)
mean(pred_test == testlpg$STATUS)
FPR(table(pred_test, testlpg$STATUS))
modelo_rf_lpg_v1 = rpart(STATUS ~ ., data = trainlpg, control = rpart.control(cp = .0005)) 
tree_pred = predict(modelo_rf_lpg_v1, testlpg, type='class')
mean(tree_pred==testlpg$STATUS)
FPR(table(tree_pred, testlpg$STATUS))
modelo_nb_lpg_v1 <- naiveBayes(STATUS ~ ., data = trainlpg)
y_pred <- predict(modelo_nb_lpg_v1, newdata = testlpg)
mean(y_pred==testlpg$STATUS)
FPR(table(y_pred, testlpg$STATUS))
#### SVM = (0.9237013 & 0.06508876) / RandomForest = (0.9366883 & 0.03846154) / NaiveBayes = (0.8522727 & 0.1893491)
# Padronização dos dados
dados_lpg_norm <- as.data.frame(lapply(dadosLpg[-c(1,6)], normalizar))
dados_lpg_norm = cbind(dados_lpg_norm,dadosLpg[c(1,6)])
dados_lpg_norm$id <- 1:nrow(dados_lpg_norm)
set.seed(1)
trainlpg2 <- dados_lpg_norm %>% dplyr::sample_frac(0.7)
testlpg2 <- dplyr::anti_join(dados_lpg_norm, trainlpg2, by = 'id')
trainlpg2$id <- NULL
testlpg2$id <- NULL
modelo_svm_lpg_v2 <- svm(STATUS ~ .,data = trainlpg2,type = 'C-classification',kernel = 'radial')
pred_test <- predict(modelo_svm_lpg_v2, testlpg2)
mean(pred_test == testlpg2$STATUS)
FPR(table(pred_test, testlpg2$STATUS))
modelo_rf_lpg_v2 = rpart(STATUS ~ ., data = trainlpg2, control = rpart.control(cp = .0005)) 
tree_pred = predict(modelo_rf_lpg_v2, testlpg2, type='class')
mean(tree_pred==testlpg2$STATUS)
FPR(table(tree_pred, testlpg2$STATUS))
modelo_nb_lpg_v2 <- naiveBayes(STATUS ~ ., data = trainlpg2)
y_pred <- predict(modelo_nb_lpg_v2, newdata = testlpg2)
mean(y_pred==testlpg2$STATUS)
FPR(table(y_pred, testlpg2$STATUS))
#### SVM = (0.9237013 & 0.06508876) / RandomForest = (0.9366883 & 0.03846154) / NaiveBayes = (0.8522727 & 0.1893491)
dados_lpg_z <- as.data.frame(scale(dadosLpg[-c(1,6)]))
dados_lpg_z = cbind(dados_lpg_z,dadosLpg[c(1,6)])
dados_lpg_z$id <- 1:nrow(dados_lpg_z)
set.seed(1)
trainlpg3 <- dados_lpg_z %>% dplyr::sample_frac(0.7)
testlpg3 <- dplyr::anti_join(dados_lpg_z, trainlpg3, by = 'id')
trainlpg3$id <- NULL
testlpg3$id <- NULL
modelo_svm_lpg_v3 <- svm(STATUS ~ .,data = trainlpg3,type = 'C-classification',kernel = 'radial')
pred_test <- predict(modelo_svm_lpg_v3, testlpg3)
mean(pred_test == testlpg3$STATUS)
FPR(table(pred_test, testlpg3$STATUS))
modelo_rf_lpg_v3 = rpart(STATUS ~ ., data = trainlpg3, control = rpart.control(cp = .0005)) 
tree_pred = predict(modelo_rf_lpg_v3, testlpg3, type='class')
mean(tree_pred==testlpg3$STATUS)
FPR(table(tree_pred, testlpg3$STATUS))
modelo_nb_lpg_v3 <- naiveBayes(STATUS ~ ., data = trainlpg3)
y_pred <- predict(modelo_nb_lpg_v3, newdata = testlpg3)
mean(y_pred==testlpg3$STATUS)
FPR(table(y_pred, testlpg3$STATUS))
#### SVM = (0.9237013 & 0.06508876) / RandomForest = (0.9366883 & 0.03846154) / NaiveBayes = (0.8522727 & 0.1893491)
table(dadosLpg$STATUS)
# Testarei equilibrando as classes preditoras
dadosDesbalanciadosLpg = dadosLpg[dadosLpg$STATUS == "Falha",]
auxiliar = dadosLpg[dadosLpg$STATUS == "Sucesso",]
auxiliar <- auxiliar %>% dplyr::sample_frac(0.80)
dadosDesbalanciadosLpg = rbind(dadosDesbalanciadosLpg,auxiliar)
table(dadosDesbalanciadosLpg$STATUS)
dadosDesbalanciadosLpg$id <- 1:nrow(dadosDesbalanciadosLpg)
set.seed(1)
train1.1 <- dadosDesbalanciadosLpg %>% dplyr::sample_frac(0.7)
test1.1  <- dplyr::anti_join(dadosDesbalanciadosLpg, train1.1, by = 'id')
train1.1$id <- NULL
test1.1$id <- NULL
modelo_svm_v1.1 <- svm(STATUS ~ .,data = train1.1,type = 'C-classification',kernel = 'radial')
pred_test <- predict(modelo_svm_v1.1, test1.1)
mean(pred_test == test1.1$STATUS)
FPR(table(pred_test, test1.1$STATUS))
modelo_rf_v1.1 = rpart(STATUS ~ ., data = train1.1, control = rpart.control(cp = .0005)) 
tree_pred = predict(modelo_rf_v1.1, test1.1, type='class')
mean(tree_pred==test1.1$STATUS)
FPR(table(tree_pred, test1.1$STATUS))
#### SVM = (0.9396709 & 0.07142857) / RandomForest = (0.9469835 & 0.04642857)
dadosLpg$id <- 1:nrow(dadosLpg)
set.seed(1)
train1.2 <- dadosLpg %>% dplyr::sample_frac(0.65)
test1.2 <- dplyr::anti_join(dadosLpg, train1.2, by = 'id')
train1.2$id <- NULL
test1.2$id <- NULL
modelo_svm_v1.2 <- svm(STATUS ~ .,data = train1.2,type = 'C-classification',kernel = 'radial')
pred_test <- predict(modelo_svm_v1.2, test1.2)
mean(pred_test == test1.2$STATUS)
FPR(table(pred_test, test1.2$STATUS))
modelo_rf_v1.2 = rpart(STATUS ~ ., data = train1.2, control = rpart.control(cp = .0005)) 
tree_pred = predict(modelo_rf_v1.2, test1.2, type='class')
mean(tree_pred==test1.2$STATUS)
FPR(table(tree_pred, test1.2$STATUS))
#### SVM = (0.9220056 & 0.06122449) / RandomForest = (0.9233983 & 0.04336735)
set.seed(1)
train1.3 <- dadosLpg %>% dplyr::sample_frac(0.75)
test1.3 <- dplyr::anti_join(dadosLpg, train1.3, by = 'id')
train1.3$id <- NULL
test1.3$id <- NULL
modelo_svm_v1.3 <- svm(STATUS ~ .,data = train1.3,type = 'C-classification',kernel = 'radial')
pred_test <- predict(modelo_svm_v1.3, test1.3)
mean(pred_test == test1.3$STATUS)
FPR(table(pred_test, test1.3$STATUS))
modelo_rf_v1.3 = rpart(STATUS ~ ., data = train1.3, control = rpart.control(cp = .0005)) 
tree_pred = predict(modelo_rf_v1.3, test1.3, type='class')
mean(tree_pred==test1.3$STATUS)
FPR(table(tree_pred, test1.3$STATUS))
#### SVM = (0.9161793 & 0.07142857) / RandomForest = (0.9473684 & 0.06428571)
set.seed(1)
train1.4 <- dadosLpg %>% dplyr::sample_frac(0.8)
test1.4 <- dplyr::anti_join(dadosLpg, train1.4, by = 'id')
train1.4$id <- NULL
test1.4$id <- NULL
modelo_svm_v1.4 <- svm(STATUS ~ .,data = train1.4,type = 'C-classification',kernel = 'radial')
pred_test <- predict(modelo_svm_v1.4, test1.4)
mean(pred_test == test1.4$STATUS)
FPR(table(pred_test, test1.4$STATUS))
modelo_rf_v1.4 = rpart(STATUS ~ ., data = train1.4, control = rpart.control(cp = .0005)) 
tree_pred = predict(modelo_rf_v1.4, test1.4, type='class')
mean(tree_pred==test1.4$STATUS)
FPR(table(tree_pred, test1.4$STATUS))
#### SVM = (0.9170732 & 0.0745614) / RandomForest = (0.9560976 & 0.04824561)
set.seed(1)
train1.5 <- dadosLpg %>% dplyr::sample_frac(0.85)
test1.5 <- dplyr::anti_join(dadosLpg, train1.5, by = 'id')
train1.5$id <- NULL
test1.5$id <- NULL
modelo_svm_v1.5 <- svm(STATUS ~ .,data = train1.5,type = 'C-classification',kernel = 'radial')
pred_test <- predict(modelo_svm_v1.5, test1.5)
mean(pred_test == test1.5$STATUS)
FPR(table(pred_test, test1.5$STATUS))
modelo_rf_v1.5 = rpart(STATUS ~ ., data = train1.5, control = rpart.control(cp = .0005)) 
tree_pred = predict(modelo_rf_v1.5, test1.5, type='class')
mean(tree_pred==test1.5$STATUS)
FPR(table(tree_pred, test1.5$STATUS))
#### SVM = (0.9058442 & 0.07647059) / RandomForest = (0.9577922 & 0.05882353)
dadosLpg$id <- NULL


#### Nenhuma das variacoes testadas apresentou um melhor desempenho em relacao ao modelo de Random Forest
#### utilizando os dados originais e divididos entre 0.7 treino e 0.3 teste.
#### Assim apresentaria o modelo:
summary(modelo_rf_lpg_v1)
#### que apresenta 0.9366883 de acuracia e 0.03846154 de FPR