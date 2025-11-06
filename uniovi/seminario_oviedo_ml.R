# Métodos de Aprendizaje Automático para la Economía Aplicada
# Juan D. Montoro-Pons
# Oviedo, 7 de Noviembre de 2025



# 1. Modelos lineales con penalización: ejemplo 1 (datos sintéticos) ------------------------------------

library(glmnet)
library(dplyr)
library(caret)
library(ggplot2)

# Conjunto de datos sintético, con 400 predictores y decrecimiento 
# exponencial de los coef beta poblacionales

url <- "https://juandmontoro.github.io/bigDataEco/data/regularized_regression.csv"
betas <- "https://juandmontoro.github.io/bigDataEco/data/betas.csv"
data <- read.csv(url)
data |> head()
dim(data)
# Rango no completo

# Visualización de los verdaderos parámetros poblacionales 
# (coeficientes del modelo lineal generador de datos sintéticos)
betas <- read.csv(betas)
betas |> head(10)
betas |>  ggplot() + geom_point(aes(x=X,y=X0),col='salmon') 

# Respuesta (y) y matriz de predictores (X)
X = data|> select(-y) |> as.matrix()
y = data$y

# El paquete glmnet estima modelos lineales generalizados con regularizacion 
# A continuación se estima un modelo lineal con penalización Lasso
# Ajustaremos el modelo para 100 valores distintos del parámetro lambda 
# (valor por defecto que se puede modificar)
fit_lasso <- glmnet(X, y, family = "gaussian",alpha=1)

# Un plot del objeto resultante muestra la senda de los coeficientes 
# estimados conforme pasamos de una menor regularización (valores más altos de norma L1) 
# a una mayor regularización. Lasso actúa como selector de variables 
# al forzar a que determiandos coeficientes (para un valor suficiente de lambda) sean 0. 
plot(fit_lasso)

# La norma L1 (o L2 para Ridge) puede verse como una restricción 
# presupuestaria para "gastar" en los coeficientes estimador. 
# Un mayor valor del presupuesto  implica una menor penalización. Existe un valor
# para el cual los coeficientes Lasso (o Ridge) son equivalentes a los OLS
plot(fit_lasso,xvar='norm')

# ¿Qué valor de lambda elegir? Aquél con mejor comportamiento predictivo fuera de muestra
# Para ello usamos validación cruzada (en este caso con 5 particiones o folds)
# El paquete glmnet gestiona internamente el proceso de cv
fit_lasso_cv <- cv.glmnet(X, y, family = "gaussian", alpha = 1, nfolds = 5)

# Minimización del EC para un valor de lambda
plot(fit_lasso_cv)

# Valores reultantes del proceso de validación cruzada
fit_lasso_cv$lambda.min # lambda que minimiza ecm
fit_lasso_cv$lambda.1se # lambda min + 1 se (añade algo más de regularización y es estándar en la literatura ML)

# R2(obs,pred) para el modelo Lasso. Realizamos las predicciones sobre muestra de ajuste
# Esto devuelve una evaluación del modelo optimista
R2(y,predict(fit_lasso_cv,newx = X)) # por defecto usa el modelo más regularizado
R2(y,predict(fit_lasso_cv,newx = X,s=fit_lasso_cv$lambda.min)) # usando lambda.min


# 2. Modelos lineales con penalización: ejemplo 2 ------------------------------------

# Datos: alojamientos AIRBNB en la ciudad de Valencia
url <- "https://juandmontoro.github.io/bigDataEco/data/airbnb.csv"
data <- read.csv(url)
dim(data)
data |> head(2)

# Para el anñalisis se realiza una partición de datos: entrenamiento(70%)/prueba(30%)
# Semilla aleatoria para reproducibilidad
set.seed(123)  
train_ratio <- 0.8
n <- nrow(data)
train_indices <- sample(seq_len(n), size = floor(train_ratio * n))
train <- data[train_indices, ]
test <- data[-train_indices, ]

# Se define respuesta y matriz de predictores
X_train = train|> select(-price) |> as.matrix()
y_train= log(train$price)
X_test = test|> select(-price) |> as.matrix()
y_test= log(test$price)

# Ajuste de un modelo lasso (alpha=1)
fit_lasso <- glmnet(X_train, y_train, family = "gaussian",alpha=1)
plot(fit_lasso,xvar='norm')

# Se observa inestabilidad de los predictores conforme algunos se anulan
# (por colinealidad). Elección del valor óptimo de lambda por CV
fit_lasso_cv <- cv.glmnet(X_train, y_train, family = "gaussian", alpha = 1, nfolds = 5)
plot(fit_lasso_cv)
fit_lasso_cv$lambda.min
fit_lasso_cv$lambda.1se

# Ajuste de modelo ridge
fit_ridge<- glmnet(X_train, y_train, family = "gaussian", alpha = 0)
plot(fit_ridge,xvar = 'norm')

# Los coeficientes en Ridge se 'encojen' pero no se anulan totalmente
# Seleccionamos lambda por CV
fit_ridge_cv <- cv.glmnet(X_train, y_train, family = "gaussian", alpha = 0, nfolds = 5)
plot(fit_ridge_cv)

# EVALUACION: métrica de bondad de ajuste dentro/fuera muestra
# Benchmark: modelo lineal (LM)
# Nota: LM devuelve warning por problema de colinealidad perfecta por 
# categorias no presentes en datos de entrenamiento
fit_lm <- lm(log(price)~., data=train)
r2_lm_train <- R2(predict(fit_lm), y_train)
r2_lm_test <- R2(predict(fit_lm,newdata=as.data.frame(X_test)), y_test)

# Modelo LASSO
r2_lasso_train <- R2(predict(fit_lasso_cv,newx = X_train), y_train)
r2_lasso_test <- R2(predict(fit_lasso_cv, newx = X_test), y_test)

# Modelo Ridge
r2_ridge_train <- R2(predict(fit_ridge_cv,newx = X_train), y_train)
r2_ridge_test <- R2(predict(fit_ridge_cv, newx = X_test), y_test)

# Tabla de resultados
lm_resultados <- data.frame(
  Modelo = c("Lineal", "LASSO", "Ridge"),
  R2_train=c(r2_lm_train, r2_lasso_train, r2_ridge_train),
  R2_test = c(r2_lm_test, r2_lasso_test, r2_ridge_test))

print(lm_resultados)

# El modelo lineal se comporta marginalmente mejor que Ridge pero si aumentamos 
# complejidad incluyendo términos no lineales y/o interacciones en los predictores
# se resiente (verificar) 

# 3. Árboles --------------------------------------------------------------

library(rpart)
library(rpart.plot)
library(dplyr)
library(caret)

# 3.1 Ajuste de un árbol poco profundo a los datos AIRBNB valencia
fit_trees <- rpart(log(price)~., train, maxdepth = 2, cp=0)
rpart.plot(fit_trees, leaf.round = 1, 
           space = 2, yspace = 2, 
           split.space = 2, 
           shadow.col = "gray", 
           trace = 1)

# Las particiones próximas al nodo raíz se realizan con variables
# que tienen una elevada asociación con la respuesta (particiones que más
# reducen el error cuadrático). En el ejemplo las variables private.room 
# (que un alojamiento sea una habitación o un apartamento completo) y 
# el número de personas que se pueden alojar son sobre las que se realizan 
# las dos primeras particiones en el árbol

# 3.2 Crecimiento del árbol sin restricciones. Ya que paquete rpart limita max.depth a 30
# se opta por estrategias alternativas: fijar número obs en nodo términal y/o num. 
# obs mínimas para poder hacer una partición a su valor mínimo para expandir la
# profundidad del árbol más allá del límite de rpart
# minsplit: the minimum number of observations that must exist in a node in order 
#           for a split to be attempted.
# minbucket: the minimum number of observations in any terminal <leaf> node. 

fit_unc <- rpart(log(price)~., train, minsplit=2, minbucket=1,cp=0)

# En este caso se produce un sobreajuste (hemos ajustado ruido de la muestra de entrenamiento)
R2(predict(fit_unc),log(y_train)) 
R2(predict(fit_unc,newdata = test),log(y_test))

# 3.3 Poda del árbol: Poda por coste de complejidad (o post-poda)
# Estrategia: hacer que el árbol crezca hasta una complejidad alta 
# (permitiendo muchas particiones), y después podar hojas/subárboles en función 
# del aumento del error promedio por nodo eliminado: el coste de la complejidad 
# o cp (cost-complexity parameter). Durante la poda cp representa el aumento
# medio del error por nodo que estamos dispuestos aceptar al simplificar el árbol.
# La elección del valor óptimo de cp se realiza mediante validación cruzada
# con el objetivo de maximizar el rendimiento predictivo fuera 
# de muestra (minimizar el error de validación).

# El árbol estimado por rpart incluye la tabla cptable que recoge para distintos valores de cp (mayor a menor):
#   * el número splits  del árbol para ese cp
#   * el error relativo de entrenamiento penalizado (error modelo podado/error modelo unc)
#   * el error de validación cruzada (xerror)
#   * su desviación estándar (xstd)
# A medida que se reducen las particiones, se reduce la ganancia en reducción del MSE
# y cp nos indica cuándo la reducción ya no compensa el aumento de complejidad.

fit_unc$cptable |> head()
fit_unc$cptable |> as.data.frame() |> ggplot(aes(x=CP,y=xerror)) +
  geom_point(col='salmon',alpha=0.7) + xlim(0,0.03) +ylim(0.5,0.7)

# plotcp ofrece información adicional 
plotcp(fit_unc, minline = FALSE)

# Elegimos el CP que minimiza el error de cv
bestcp <- fit_unc$cptable[which.min(fit_unc$cptable[, "xerror"]), "CP"]
# Y a continuacion podamos el árbol basada en el parámetro cp
fit_post <- prune(fit_unc, cp = bestcp)


df <- as.data.frame(fit_unc$cptable)
ggplot(df, aes(CP, xerror)) +
  geom_line() +
  geom_point(color = "red", size = 2) +
  geom_errorbar(aes(ymin = xerror - xstd, ymax = xerror + xstd),
                width = 0.01,
                color = "gray50",
                alpha = 0.04,        # barra más tenue
                linewidth = 0.5) +  # más fina
  scale_x_log10() +
  labs(x = "cp (log scale)", y = "Cross-validated rel.error") +
  theme_minimal()

# Vemos que el rendimiento predictivo en train y test se aproximan
r2_post_train <- R2(predict(fit_post),log(y_train)) 
r2_post_test <- R2(predict(fit_post,newdata=test),log(y_test))
r2_post_train
r2_post_test

# 3.4 Poda del árbol: pre-poda. Se eligen valores de parámetros de ajuste
#     (p.e.profundidad máxima, número mínimo de obs en un nodo final o 
#     número mínimo de obs para realizar una partición) ANTES de entrenar
#     el árbol. Evaluamos por validación cruzada.

# En este ejemplo evaluamos una secuencia de max.depth para el árbol.
# Se usa el paquete caret, un framework unificado para entrenar, evaluar 
# y seleccionar modelos de aprendizaje automático. Caret simplifica el 
# preprocesamiento, la selección de hiperparámetros y la 
# evaluación de modelos mediante una interfaz consistente para múltiples
# algoritmos (es decir, paquetes).

set.seed(123)
# Control del proceso de entrenamiento: cv con 5 particiones
ctrl <- trainControl(method = "cv", number = 5)

# Secuencia de profundidades a evaluar
grid <- expand.grid(maxdepth = 5:30)

# Entrenamiento del modelo
modelo_pre <- train(
  log(price)~.,
  data= train, 
  method = "rpart2",           
  trControl = ctrl,
  tuneGrid = grid,
  control = rpart.control(cp = 0) 
)

# Resultados de CV (la media de la métrica pero también SD, para ver si inestabilidad)
modelo_pre$results

# Mejor valor de max.depth (y modelo)
modelo_pre$bestTune
plot(modelo_pre)

# Elección del mejor modelo por CV
fit_pre <- modelo_pre$finalModel

r2_pre_train <- R2(predict(fit_pre),log(y_train)) 
r2_pre_test <- R2(predict(fit_pre,newdata=test),log(y_test))

# Crear tabla de resultados para todos los árboles entrenados
arbol_resultados <- data.frame(
  Modelo = c("Árbol unc.", "Árbol post","Árbol pre"),
  R2_train = c(r2_unc_train, r2_post_train,r2_pre_train),
  R2_test = c(r2_unc_test, r2_post_test,r2_pre_test)
)

arbol_resultados


# 4. Ensambles (i): bosques aleatorios ----------------------------------------

library(randomForest)

# 4.1 Entrenamiento de un bosque aleatorio
# Hiperparámetros: 
#   ntrees: número de árboles a estimar (=número de muestras bootstrap)
#           Aumentar el número de estimadores mejora la calidad predictiva sin  
#           riesgo de sobreajuste (el error de generalización tiende a 
#           estabilizarse). Elegir valor elevado (100 o más pero coste tiempo computacion).
#   mtry: porcentaje de predictores que se seleccionan aleatoriamente para partición.
#         Valores más bajos descorrelacionan más los árboles: estos difieren más
#         al entrenarse sobre conjunto menor y muy probablemente más diferente de
#         predictores (menor varianza pero mayor sesgo al agregar modelos más 
#         simples). Valores más altos reducen el sesgo pero reducen el beneficio
#         de la agregación se reduce 

# Prueba con p/3 predictores en cada partición  
m <- dim(X_train)[2]/3                         
fit_rf <- randomForest(log(price)~.,
                       ntree = 100,   
                       mtry=m,  
                       data = train)           

# Visualización de info básica sobre el modelo estimado
fit_rf
# %var. explained es R2 para observaciones out of bag (OOB)

r2_rf_train <- R2(predict(fit_rf,X_train),y_train)
r2_rf_test <-R2(predict(fit_rf,X_test),y_test)

r2_rf_train
r2_rf_test

# 4.2 Selección de mejor hiperparámetro mtry por CV
# Conviene búsqueda exhaustiva de % de predictores elegidos aleatoriamente 
# en una partición: recurso a paquete caret

# Como el proceso es lento doParallel paraleliza
library(doParallel)
# Detectar núcleos disponibles y usar todos menos uno
cl <- makeCluster(parallel::detectCores() - 1)
registerDoParallel(cl)

# Parámetros que controlan el ajuste
ctrl <- trainControl(
  method = "cv", number = 5, 
  summaryFunction = defaultSummary,  # RMSE, Rsquared, MAE
  verboseIter = TRUE,
  allowParallel = TRUE
)

# Grid para mtry
p <- dim(X_train)[2]
grid_mtry <- expand.grid(
  mtry = round(seq(0.1,0.9,0.05)*p))

# Muestreo CV
set.seed(123)
# Entrenamiento
fit_rf_cv <- train(
  log(price) ~ .,               
  data = train,
  method = "rf",           # usa randomForest
  trControl = ctrl,
  tuneGrid = grid_mtry,
  ntree = 100,              
  importance = TRUE
)

stopCluster(cl)

# Visualización RMSE cv vs. mtry
plot(fit_rf_cv)

# Resultados
print(fit_rf_cv)
plot(fit_rf_cv)              # gráfico scoring vs mtry
fit_rf_cv$bestTune           # mejor valor de mtry
fit_rf_cv <- fit_rf_cv$finalModel

# Evaluación del modelo
r2_rf_cv_train <- R2(predict(fit_rf_cv,X_train),y_train)
r2_rf_cv_test <- R2(predict(fit_rf_cv,X_test),y_test)
r2_rf_cv_train 
r2_rf_cv_test


# 5. Ensambles (ii): extreme gradient boosting ---------------------------------

# 5.1 Entrenamiento de xgboost 
library(xgboost)

# DMatrix (mejor gestión de datos en memoria)
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest  <- xgb.DMatrix(data = X_test,  label = y_test)

# Parámetros básicos para regresión
params <- list(
  eta = 0.005,          # learning rate (pequeño; candidato a ajuste por cv)
  max_depth = 6,        # profundidad máxima árboles
  min_child_weight = 1, # número de instancias en nodo terminal
  subsample = 0.8,      # muestreo (% de obs) para generar arboles. Evita sobreajuste
  colsample_bytree = 0.8, # muestreo sobre predictores por arbol
  lambda = 1,           # regularización L2
  alpha = 0             # regularización L1
)

# Early stopping: parar proceso si en la secuencia de árboles
# entrenados tras t iteraciones no se producen mejoras en la calidad
# del estimador en un conjunto de prueba. Previene overfitting y
# evita tiempo de procesamiento innecesario.
# Pasamos a watch el conjunto de prueba

watch <- list(train = dtrain, eval = dtest)

# Entrenamos
fit_xgb <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 5000,           # número máximo de árboles. Fijar número elevado
  watchlist = watch,
  early_stopping_rounds = 50, # rondas para stop si no mejora rmse en eval
  print_every_n = 50 # permite ver la evolución del ajuste cada 50 iter
)

# Objeto resultante: gráfico rmse (train/eval) vs. n_trees
fit_xgb$evaluation_log |> ggplot() +  
  geom_point(aes(x=iter,y=train_rmse), alpha=0.15, color='salmon') +
  geom_point(aes(x=iter,y=eval_rmse), alpha=0.15,color='navy') + 
  xlab('Iteración') + ylab('rmse train/rmse eval') +    
  geom_vline(xintercept = 2351, color = "brown", linetype = "dashed", size = 1) + 
  annotate("text",label=paste0('n_trees = ',fit_xgb$best_iteration), x = fit_xgb$best_iteration, y=1) +
  theme_minimal() + 
  scale_y_log10()  

fit_xgb$best_iteration
fit_xgb$best_score

# Predicción y métricas
r2_xgb_train <-R2(predict(fit_xgb, dtrain),y_train)
r2_xgb_test <- R2(predict(fit_xgb, dtest),y_test)
r2_xgb_train
r2_xgb_test


# 5.2 xgboost: elección de learning rate por CV

# Warning: ejecución larga
etas <- c(0.0025,0.005, 0.01,0.05, 0.1)
# Aquí usa watchlist interno (el fold de validación)
results <- lapply(etas, function(e) {
  params$eta <- e
  set.seed(123)
  cv <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 10000, # dejamos un número elevado de iter (al fijar etas bajos) 
    nfold = 5,
    early_stopping_rounds = 50,
    metrics = "rmse",
    verbose = FALSE
  )
  
  tibble(eta = e, 
         best_rmse = min(cv$evaluation_log$test_rmse_mean),
         best_iter = cv$best_iteration)
})

tabla <- bind_rows(results)
tabla

# Grafico de los resultados de CV
ggplot(tabla, aes(x = eta, y = best_rmse)) + geom_line(color='navy') +
  geom_point(aes(size = best_iter, color = best_iter)) + 
  scale_x_log10() +
  labs(x = "eta", y = "RMSE", size = "Iteraciones", color = "Iteraciones",
       title = "Trade-off: RMSE vs eta según iteraciones") + 
  annotate('text',x=tabla$eta,y=tabla$best_rmse+0.0002,label=tabla$best_iter) +
  theme_minimal() +
  theme(legend.position = "none")



# Elección mejor eta
best_eta <- tabla %>% slice_min(best_rmse) %>% pull(eta)
best_eta
best_iter <- tabla %>% slice_min(best_rmse) %>% pull(best_iter)

# Entrenamiento modelo final (mejor eta)
params$eta <- best_eta

fit_xgb_cv <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = best_iter
)

r2_xgb_cv_train <- R2(predict(fit_xgb_cv,dtrain),y_train)
r2_xgb_cv_test <- R2(predict(fit_xgb_cv,dtest),y_test)
r2_xgb_cv_train
r2_xgb_cv_test


# 6. Tabulación resultados ------------------------------------------------


Modelo = c("LM","Lasso","Ridge","Árbol unc.", "Árbol post",
           "Árbol pre","Random forest","Random forest cv","xgb","xgb cv")
R2_train = c(r2_lm_train,r2_lasso_train,r2_ridge_train,
             r2_unc_train, r2_post_train,r2_pre_train,r2_rf_train,
             r2_rf_cv_train,r2_xgb_train,r2_xgb_cv_train)
R2_test = c(r2_lm_test,r2_lasso_test,r2_ridge_test,
            r2_unc_test, r2_post_test,r2_pre_test,r2_rf_test,
            r2_rf_cv_test,r2_xgb_test,r2_xgb_cv_test)


resultados <- data.frame(Modelo,R2_train,R2_test,R2_cv)
resultados


# 7 Otros  -----------------------------------------------------------------------

# install.packages('quanteda','DoubleML','grf')
# Paquetes en R que permiten trabajo con datos no estructurados o inferencia causal ML
#   quanteda o tm: procesamiento datos texto y creación de matrices documentos-términos
#   DoubleML: double/debiased machine learning of causal effects
#   grf: generalized random forests. Incluye causal_forest para estimación de efectos condicionales
