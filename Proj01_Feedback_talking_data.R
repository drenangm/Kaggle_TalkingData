setwd("C:/MyBkp/Cursos/DSA/DSA_Formacao_Cientista_Dados/Curso01_Big-Data-Analytics-com-R-e-Microsoft-Azure-Machine-Learning/Projeto_Feedback/Projetos-1-2/Proj01")
getwd()

# -------------------------#
# data information:
# -------------------------#
  
# ip: ip address of click.
# app: app id for marketing.
# device: device type id of user mobile phone (e.g., iphone 6 plus, iphone 7, huawei mate 7, etc.)
# os: os version id of user mobile phone
# channel: channel id of mobile ad publisher
# click_time: timestamp of click (UTC)
# attributed_time: if user download the app for after clicking an ad, this is the time of the app download
# is_attributed: the target that is to be predicted, indicating the app was downloaded

#--------------------------#
# necessidades identificadas no dataset de treino:
# -> divisão do dataset de treino para alocação em memoria
# -> balanceamento dos dados de treino para a variável target
#--------------------------#

install.packages('bigreadr')
install.packages('fasttime')
install.packages('tidyverse')
install.packages('fastAdaboost')
install.packages('xgboost')
library(bigreadr)
library(utils)
library(dplyr)
library(data.table)
library(ggplot2)
library(fasttime)
library(lubridate)
library(tidyverse)
library(lattice)
library(caret)
library(e1071)
library(randomForest)
library(pROC)
library(fastAdaboost)
library(ROCR)

# Os arquivos de treino e teste sao muito grandes para serem carregados de uma so vez em memoria
# Portanto sera necessario adotar um metodo de divisao destes arquivos

# Variavel de divisao:
numChunks <- 5

# vamos desconsiderar o cabecalho:
totalLines <- nlines('train.csv') - 1
maxLinesChunks <- round(totalLines / numChunks)

# Numero de linhas por arquivo: 36.980.778

# apos divisao do dataset em partes (chunks), determinaremos o nome destas partes
chunks <- c('train_part1.csv', 'train_part2.csv', 'train_part3.csv', 'train_part4.csv', 'train_part5.csv')

# carregando o primeira parte do dataset
chunk <- fread(chunks[1])

# checando as colunas do dataset
colNames <- colnames(chunk)

# checando uma amostra do dataset
head(chunk)

#-----------------Analise Exploratoria--------------#

# Verificando o balanceamento dos dados da variavel target
# Carregaremos os chunks, removeremos valores duplicados, caso existam, e retornaremos a quantidade de cada classe da variavel target

targetCount <- sapply(chunks, function(x) {
  chunk <- fread(x, col.names = colNames)
  chunk <- chunk[!duplicated(chunk), ]
  table(chunk$is_attributed)
})

# sumarizando os resultados obtidos com a transposicao das linhas e colunas
# para melhor visualizacao da variavel target
chunksSummary <- t(targetCount)

# Renomeando o nome das colunas para melhor entendimento
colnames(chunksSummary) <- c('No', 'Yes')

# Apresentando os resultados obtidos 
prop.table(chunksSummary)
View(chunksSummary)


#--------------------------------------------#
#                    No         Yes
#train_part1.csv 0.1995894 0.0005177438
#train_part2.csv 0.1996269 0.0004779610
#train_part3.csv 0.1991522 0.0005639860
#train_part4.csv 0.1995156 0.0004459230
#train_part5.csv 0.1995958 0.0005144285
#--------------------------------------------#


# O resultado mostrou proporcoes bem parecidas entre cada chunk do dataset
# Para melhorar a leitura dos resultados vamos agrupar os valores obtidos pasra termos uma analise de todo o dataset

# Somando os resultados de cada coluna do dataset
targetTotal <- sapply(as.data.frame(chunksSummary), sum)

# Retornando os resultados obtidos apos a soma de todos os valores
# valores absolutos
print(targetTotal)
# valores proporcionais
print(prop.table(targetTotal))

# Proporcao final sumarizada: 
# No          Yes 
# 180827810   456845
# 0.997479958 0.002520042

# Plot dos resultados obtidos na variavel target

data.frame(downloadedApp = c('No','Yes'), counts = targetTotal) %>%
  ggplot(aes(x = downloadedApp, y = counts, fill = downloadedApp)) +
  geom_bar(stat = 'identity') +
  labs(title = 'Proportion of downloaded Apps by users') +
  theme_bw()

#-------------Balanceamento dos dados de treino---------------#

# Claramente a proporcao dos dados da variavel target e desequilibrada 
# o que dificultaria o aprendizado do modelo para casos em que usuarios fizeram download
# Iremos corrigir esta distorcao criando um novo conjunto de dados 

# Extraindo de cada parte do dataset todos os registros de download efeutado ('is_attributed == 1')

# Funcao para carregar os dados, remocao das linhas duplicadas e capturar usuarios que fizeram download
sapply(chunks, function(c) {
  chunk <- fread(c, col.names = colNames)
  chunk <- chunk[!duplicated(chunk), ]
  fwrite(chunk[chunk$is_attributed == 1,], 'train_sample.csv', append = TRUE)
  return('Saved file!')
})

# Sumarizando o numero de linhas em que is_attributed == 1
usersYes <- t(targetTotal)[2]

# Definicao do numeor de linhas que deverao ser amostradas em cada chunk para usuarios que nao fizeram download
usersSampleNo <- round(usersYes / length(chunks))

# Aplicando o seed para reproducao da amostragem
set.seed(42)

# amostrando em cada chunk linhas as quais a classe is_attributed == 0

## Funcao para carregar os dados, remocao das linhas duplicadas e capturar usuarios que nao fizeram download (por amostragem)
sapply(chunks, function(c) {
  chunk <- fread(c, col.names = colNames)
  chunk <- chunk[!duplicated(chunk), ]
  chunk <- chunk[chunk$is_attributed == 0, ]
  fwrite(chunk[sample(1:nrow(chunk), usersSampleNo),], 'train_sample.csv', append = TRUE)
  return('Saved file!')
})

# Carregando o dataset 
dataTarget = fread('train_sample.csv')

# Plotando os dados obtidos em um grafico

dataTarget %>%
  mutate(downloadedApp = factor(is_attributed, labels = c('No', 'Yes'))) %>%
  ggplot(aes(x = downloadedApp, fill = downloadedApp)) +
  geom_bar() +
  labs(title = 'Proportion of downloaded Apps by users') +
  ylab('count') +
  theme_bw()

#-----------------Carregando os dados de teste-----------------#

# Utilizaremos a mesma solucao adotada para os dados de treino
# Faremos isso para termos proporcoes adequadas quando formos aplicar a fase de treino/teste do modelo

#  definicao da quantidade de chunks de teste
numTestChunks <- 5
totalTestLines <- nlines('test.csv') - 1
maxLinesTestChunks <- round(totalTestLines / numTestChunks)

# Da mesma forma feita com os dados de treino, utilizaremos o comando split (Linux) para fazer a divisao
# dos dados de teste em 5 chunks -> volume total por chunk de 3.758.094 linhas
# drgm@sandbox:~/Big_Data_Analytics_R/Project_Feedback$ split test.csv -l 3758094

# determinando os nomes dos chunks com o dataset de teste
testChunks <- c('test_part1.csv', 'test_part2.csv', 'test_part3.csv', 'test_part4.csv', 'test_part5.csv')

# fazendo o carregamento do primeiro chunks de dados de teste
testChunk <- fread(testChunks[1])

# checando as 1as linhas
head(testChunk)

#-------------------Analise Exploratoria-------------------#

# Verificando a estrutura dos dados no dataset
str(dataTarget)
View(dataTarget)

# checando a existencia de valores NA's no dataset
sapply(dataTarget, function(v) {anyNA(v)})

# apenas attributed_time possui valores NA`s
# vamos contar quantos valores NA's temos na coluna "attributed_time"
sum(is.na(dataTarget$attributed_time))

# temos 456845 registros de NA's no dataset
# concluimos que estes registros tem relacao com o fato do usuario nao ter feito download
# faz todo sentido nao haver registro de uma acao que nao foi praticada pelo usuario

# vamos verificar registro unicos das variaveis int (ip, app, device, os, channel)
dataTarget %>%
  select('ip') %>%
  n_distinct()
dataTarget %>%
  select('app') %>%
  n_distinct()
dataTarget %>%
  select('device') %>%
  n_distinct()
dataTarget %>%
  select('os') %>%
  n_distinct()
dataTarget %>%
  select('channel') %>% 
  n_distinct()

# ip: 253051
# app: 336
# device: 1881
# OS: 190
# channel: 180

# pela quantidade mapeada de ips e possivel concluir que podem existir multiplos
# anuncios associados ao registro de um unico usuario

# Confirmando se a variavel target e representada por apenas duas classes diferentes
dataTarget %>%
  select('is_attributed') %>% 
  n_distinct()

#----------------Data Munging----------------#

# attributed_time para is_attributed = 0 -> NA
# estrutra de dados de attributed_time e click_time -> POSIXct


#--------------------------------------------#

# Avaliando a variavel click_time 
# Checando a distribuicao de datas em que ocorreram os cliques
summary(dataTarget$click_time)

# verificando o periodo de tempo em que ocorreram os cliques
max(dataTarget$click_time) - min(dataTarget$click_time)
# Periodo de clicks ocorreram em um periodo de aproximadamente 3 dias

# Criaremos um grafico com serie temporal para checarmos o numero de cliques em anuncios por hora 
# e que levaram ao download de apps durante todo o intervalo de tempo
dataTarget %>%
  mutate(datesFix = floor_date(click_time, unit = 'hour')) %>%
  group_by(datesFix) %>%
  summarise(downloadsDone = sum(is_attributed)) %>%
  ggplot(aes(x = datesFix, y = downloadsDone)) +
  geom_line() +
  scale_x_datetime(date_breaks = '4 hours', date_labels = '%d %b /%H hrs') +
  theme_light() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  xlab('Clicks done by users') +
  ylab('Downloads') +
  labs(title = 'Downloads done by period of time')

# A serie mostra que existe consistencia de downloads ao longo do dia dentro do periodo de 3 dias
# o pico de downloads ocorreu das 00:00 as 13:00 hrs

# Faremos a mesma analise considerando cliques que nao levaram a downloads
dataTarget %>%
  mutate(datesFix = floor_date(click_time, unit = 'hour')) %>%
  group_by(datesFix) %>%
  summarise(notDownloaded = sum(!is_attributed)) %>%
  ggplot(aes(x = datesFix, y = notDownloaded)) +
  geom_line() +
  scale_x_datetime(date_breaks = '4 hours', date_labels = '%d %b /%H hrs') +
  theme_light() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  xlab('Clicks done by users') +
  ylab('Not downloaded') +
  labs(title = 'Downloads not done by period of time')

# Os periodos em que ocorreram cliques e nao foram feitos downloads seguem uma distribuicao
# parecida em relacao ao grafico anterior

# Vamos sobrepor os dois graficos

dataTarget %>%
  mutate(datesFix = floor_date(click_time, unit = 'hour')) %>%
  group_by(datesFix) %>%
  summarise(downloadsDone = sum(is_attributed),
            notDownloaded = sum(!is_attributed)) %>%
  ggplot() +
  geom_line(aes(x = datesFix, y = downloadsDone, color = 'Yes')) +
  geom_line(aes(x = datesFix, y = notDownloaded, color = 'No')) +
  scale_x_datetime(date_breaks = '4 hours', date_labels = '%d %b /%H hrs') +
  theme_light(base_size = 20) +
  theme(axis.text.x = element_text(angle = 45, hjust =1)) +
  xlab('Clicks done by users') +
  ylab('Clicks Attributed') +
  labs(title = 'Cross Evaluation Downloaded x not Downloaded', colour = 'Downloaded')

#-------------------Feature Engineering---------------#

# Vamos preparar os dados de treino para criacao do modelo preditivo:
# A variavel attributed_time sera removida pois tem relacao direta com o que desejamos prever na variavel target (is_attributed = 1)
# Transformaremos a variavel click_time a qual vamos remover a unidade "Mes", pois nao e relevante nesta amostragem
# Vamos agrupar a variavel ip com as demais variaveis do dataset devido a grande quantidade de valores unicos que ela possui
dataTarget <- dataTarget %>%
  select(-c(attributed_time)) %>%
  mutate(day = day(click_time), hour = hour(click_time)) %>%
  select(-c(click_time)) %>%
  add_count(ip, day, hour) %>% rename('ip_day_hour' = n) %>%
  add_count(ip, hour, app) %>% rename('ip_hour_APP' = n) %>%
  add_count(ip, hour, device) %>% rename('ip_hour_DEVICE' = n) %>%
  add_count(ip, hour, os) %>% rename('ip_hour_OS' = n) %>%
  add_count(ip, hour, channel) %>% rename('ip_hour_CHANNEL' = n) %>%
  select(-c(ip))

# O mesmo processo de feature engineering sera aplicado a cada chunk do dataset de teste
colTestNames <- colnames(testChunk)
sapply(testChunks, function(x) {
  testChunk <- fread(x, col.names = colTestNames)
  testChunk <- testChunk %>%
    mutate(day = day(click_time), hour = hour(click_time)) %>%
    select(-c(click_time)) %>%
    add_count(ip, day, hour) %>% rename('ip_day_hour' = n) %>%
    add_count(ip, hour, app) %>% rename('ip_hour_APP' = n) %>%
    add_count(ip, hour, device) %>% rename('ip_hour_DEVICE' = n) %>%
    add_count(ip, hour, os) %>% rename('ip_hour_OS' = n) %>%
    add_count(ip, hour, channel) %>% rename('ip_hour_CHANNEL' = n) %>%
    select(-c(ip))
  fwrite(testChunk, paste('modif_', x, sep = ''), append = TRUE)
  return('Test chunk saved!')
})

# Converteremos a variavel target do tipo int para tipo factor pois resolveremos um problema em que nossa variavel target
# tera como resultado um valor do tipo categorico
dataTarget$is_attributed <- as.factor(dataTarget$is_attributed)

# Salvando o resultado em um arquivo consolidado
fwrite(dataTarget, 'dataTarget.csv')
str(dataTarget)

###########-------------------#############

# para "resgatar" a variavel dataTarget, caso necessario
dataTarget <- fread('dataTarget.csv')

###########-------------------#############

# Avaliacao da importancia de cada variavel
# Iremos utilizar o algoritmo RandomForest para avaliar peso que cada variavel tem na predicao da variavel target 
# Sera definido um range de definicao dos valores dos hiperparametros para posterior avaliacao
# sobre qual configuracao e mais eficiente

# definindo um dataframe com os resultados das avaliacoes das configuracoes
featureEvalRF <- data.frame()

# estabelecendo o range de arvores e nos
RFTrees <- 1:25
RFNodes <- 1:10

# quantidade de modelos que serao criados
modelComb <- length(RFTrees) * length(RFNodes) 

# vamos criar uma variavel para que possamos acompanhar a avaliacao em andamento
# em seguida sera feita a avaliacao dos atributos de maior importancia e geracao da matriz de confusao
count <- 0
for(t in RFTrees) {
  for(n in RFNodes) {
    set.seed(100)
    modelEvalRF <- randomForest(is_attributed ~ .,
                                data = dataTarget,
                                ntree = t,
                                nodesize = n,
                                importance = TRUE)
    confusionRF <- confusionMatrix(table(
      data = modelEvalRF$y,
      reference = modelEvalRF$predicted
    ))
    
    featureEvalRF <- rbind(featureEvalRF, data.frame(
      nodes = n,
      trees = t,
      accuracy = unname(confusionRF$overall['Accuracy'])
    ))
    
    count <- count + 1
  }
}

# Resultados serao salvos em um arquivo .csv
fwrite(featureEvalRF, 'featureEvalRF.csv')

# carregando o dataframe com os resultados obtidos
featureEvalRF <- fread('featureEvalRF.csv')

# Impressao da configuracao que apresentou melhor desempenho
bestFeatureRF <- featureEvalRF[featureEvalRF$accuracy == max(featureEvalRF$accuracy),]
bestFeatureRF

#    nodes trees  accuracy
# 1:    10    25 0.9185053

modelRF <- randomForest(is_attributed ~ .,
                        data = dataTarget,
                        ntree = bestFeatureRF$trees,
                        nodesize = bestFeatureRF$nodes,
                        importance = TRUE)

# Retornando os resultados do modelo
modelRF

# Type of random forest: classification
# Number of trees: 25
# No. of variables tried at each split: 3

# OOB estimate of  error rate: 8.12%
# Confusion matrix:
#        0      1 class.error
# 0 440114  16729  0.03661871
# 1  57425 399419  0.12569936

# Criaremos um grafico para visualizar o nivel de importancia de cada variavel na determinacao da variavel target
varEval <- as.data.frame(varImpPlot(modelRF))

# retornando uma lista com o nivel de importancia de cada variavel em ordem decrescente
varEval[order(varEval$MeanDecreaseAccuracy, decreasing = TRUE),]

#                 MeanDecreaseAccuracy MeanDecreaseGini
# app                        38.517955       194270.067
# hour                       24.078986         7547.198
# day                        23.425862         2150.136
# ip_day_hour                19.625139         4225.207
# channel                    16.329030        90920.962
# ip_hour_APP                14.782068         2674.303
# ip_hour_CHANNEL            12.496946         7251.455
# ip_hour_DEVICE             11.885089        11480.762
# ip_hour_OS                 11.423202         2516.927
# device                      5.617553        22971.100
# os                          4.337295        18479.965

#------------------Criacao do Modelo Preditivo-----------------#

# Sera feita a selecao de 3 algoritmos para treinamento do modelo preditivo
# que melhor se adaptam a conjunto de dados muito grande:

# Naive-Bayes
# Random Forest
# Adaboost

#---------------Algoritmo Naive-Bayes------------#

# criando o modelo com o Naive-Bayes
modelNB <- naiveBayes(is_attributed ~ ., data = dataTarget)

# fazendo previsoes
predictionsNB <- predict(modelNB, dataTarget, type = c('class', 'raw'))

# Confusion Matrix
confusionMatrix(table(pred = predictionsNB, data = dataTarget$is_attributed))

#################################

# Accuracy : 0.648  

# pred      0      1
#    0 440345 305082
#    1  16500 151763

# 'Positive' Class : 0 

#################################

# Gerando a curva ROC
rocNB <- prediction(as.numeric(predictionsNB), dataTarget$is_attributed)
perfNB <- performance(rocNB, 'tpr', 'fpr')
plot(perfNB, col = 'blue', main = 'Curva ROC - Naive-Bayes')
abline(a = 0, b = 1)

# Precision/Recall
prNB <- performance(rocNB, 'prec', 'rec')
plot(prNB, main = 'Curva Precision/Recall - Naive-Bayes')


#---------------Algoritmo Random Forest--------------#

# Criando o modelo com o Random Forest
modelAppRF <- randomForest(is_attributed ~ .,
                           ntree = 25,
                           nodesize = 10,
                           data = dataTarget)

# Fazendo previsoes
predictionsAppRF <- predict(modelAppRF, dataTarget, type = 'response')

# Confusion Matrix
confusionMatrix(table(pred = predictionsAppRF, data = dataTarget$is_attributed))

###################################

# Accuracy : 0.929  

#     data
# pred      0      1
#    0 444632  52686
#    1  12213 404159

# 'Positive' Class : 0 

###################################

# Gerando a curva ROC
rocRF <- prediction(as.numeric(predictionsAppRF), dataTarget$is_attributed)
perfRF <- performance(rocRF, 'tpr', 'fpr')
plot(perfRF, col = 'green', main = 'Curva ROC - Random Forest')
abline(a = 0, b = 1)

# Precision/Recall
prRF <- performance(rocRF, 'prec', 'rec')
plot(prRF, main = 'Curva Precision/Recall - Random Forest')

str(predictionsAppRF)

#---------------Algoritmo AdaBoost------------#

# Criando o modelo com o AdaBoost
modelAdaB <- adaboost(is_attributed ~ ., data = as.data.frame(dataTarget), nIter = 10, method = 'adaboost')

# Fazendo previsoes
predictionsAdaB <- predict(modelAdaB, dataTarget, type = 'class')

# Confusion Matrix
confusionMatrix(table(pred = predictionsAdaB$class, data = dataTarget$is_attributed))

######################

# Accuracy :  0.9569

# pred      0      1
#    0 436693  19221
#    1  20152 437624

# 'Positive' Class : 0 

######################

# Gerando a curva ROC
rocAdaB <- prediction(as.numeric(predictionsAdaB$class), dataTarget$is_attributed)
perfAdaB <- performance(rocAdaB, 'tpr', 'fpr')
plot(perfAdaB, main = 'Curva ROC - AdaBoost')
abline(a = 0, b = 1)

# Precison/Recall
prAdaB <- performance(rocAdaB, 'prec', 'rec')
plot(prAdaB, main = 'Curva Precision/Recall - AdaBoost')

#-----------------Otimizacao do modelo com melhor desempenho--------------#

# Buscaremos otimizar o desempenho de modelo desenvolvido com o AdaBoost alterando alguns parametros
modelOptAda <- adaboost(is_attributed ~ .,
                         data = as.data.frame(dataTarget),
                         nIter = 50,
                         method = 'adaboost')

# Fazendo previsoes com o modelo otimizado
predictionsOptAda <- predict(modelOptAda, dataTarget, type = 'class')

confusionMatrix(table(pred = predictionsOptAda$class, data = dataTarget$is_attributed))

######################

# Accuracy : 0.9617  

# pred      0      1
#    0 429889   8011
#    1  26956 448834

######################

# Gerando a curva ROC
rocOptAdaB <- prediction(as.numeric(predictionsOptAda$class), dataTarget$is_attributed)
perfOptAdaB <- performance(rocOptAdaB, 'tpr', 'fpr')
plot(perfOptAdaB, main = 'Curva ROC - AdaBoost Optimized')
abline(a = 0, b = 1)

# Precison/Recall
prOptAdaB <- performance(rocOptAdaB, 'prec', 'rec')
plot(prOptAdaB, main = 'Curva Precision/Recall - AdaBoost Optimized')


#-------------------Previsoes com os dados de teste--------------------#

# Realizaremos as previsoes utilizando o modelo com Adaboost Otimizado
# Os resultados obtidos serao salvos em um arquivo .csv

# Chamando algumas variaveis ja utilizadas e que foram modificadas devido a transformacoes no dataset de teste
testModifChunks <- c('modif_test_part1.csv', 'modif_test_part2.csv', 'modif_test_part3.csv', 
                     'modif_test_part4.csv', 'modif_test_part5.csv')
testModifChunk <- fread(testModifChunks[1])
colModifTestNames <- colnames(testModifChunk)

sapply(testModifChunks, function(x) {
  chunk <- fread(x, col.names = colModifTestNames)
  predictionsTest <- predict(modelOptAda, chunk[,!'click_id'], type = 'class')
  fwrite(data.frame(click_id = as.integer(chunk$click_id),
                    is_attributed = as.numeric(predictionsTest$class)), 'predictions.csv', append = TRUE)
  rm(chunk, predictionsTest)
  return('Saved results!')
})

