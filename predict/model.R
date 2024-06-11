## REQUIRED PACKAGES
## -----------------
library(MASS, exclude = 'select')
library(lme4)
library(arm)
library(tidyverse)
library(pubtheme)
library(corrplot)
library(glmnet)
library(pROC)

## LOADING THE DATA
## ----------------
load('data/labeled_points.Rdata')

labeled = labeled %>% 
  select(ID, landcover)

dl <- labeled_train %>%
  left_join(labeled, by = 'ID') %>%
  mutate(veg = ifelse(landcover %in% c('natforest', 'orchard', 'cropland'), 1, 0), 
         NDVI100 = NDVI*100, 
         NDBI100 = NDBI*100)

## COMPUTING MEAN VALUES PER LOCATION
## -----------------------------------
dm <- dl %>%
  group_by(ID) %>%
  summarise(
    B1   = mean(B1, na.rm=T),
    B2   = mean(B2, na.rm=T),
    B3   = mean(B3, na.rm=T),
    B4   = mean(B4, na.rm=T),
    B5   = mean(B5, na.rm=T),
    B6_VCID_1 = mean(B6_VCID_1, na.rm=T),
    B6_VCID_2 = mean(B6_VCID_2, na.rm=T),
    B7        = mean(B7, na.rm=T),
    NDVI100   = mean(NDVI100, na.rm=T), 
    # NDBI100   = mean(NDBI100, na.rm=T), ## causes warnings with lasso
    # EVI  = mean(EVI, na.rm=T),          ## not relevant
    landcover = unique(landcover), 
    veg = unique(veg)) %>%
  select(-ID, -landcover)

## PREDICTIVE MODELS
## -----------------
## logistic regression with no regularization using only NDVI100
m1 <- glm(veg ~ NDVI100, data = dm, family = binomial)
dm$predm1 <- predict(m1, type = 'response', newdata = dm)
summary(m1)

## logistic regression with no regularization using all band values
m2 <- glm(veg ~ ., data = dm, family = binomial)
dm$predm2 <- predict(m2, type = 'response', newdata = dm)
summary(m2)

## logistic regression with ridge regularization using all band values
x <- model.matrix(veg ~ B1 + B2 + B3 + B4 + B5 + 
                    B6_VCID_1 + B6_VCID_2 + B7 + NDVI100, data = dm)[,-1]
y <- dm$veg
m3 <- cv.glmnet(x, y, family = 'binomial', alpha = 0)
dm$predm3 <- predict(m3, newx = x, s = 'lambda.1se', type = 'response')
summary(m3)

## logistic regression with lasso regularization using all band values
m4 <- cv.glmnet(x, y, family = 'binomial', alpha = 1)
dm$predm4 <- predict(m4, newx = x, s = 'lambda.1se', type = 'response')
summary(m4)

## COEFFICIENTS AND TRACE CURVES
## -----------------------------
coefs.m3 <- coef(m3, s = m3$lambda)
coefs.m4 <- coef(m4, s = m4$lambda)
colnames(coefs.m3) = paste0('lambda', round(m3$lambda,6))
colnames(coefs.m4) = paste0('lambda', round(m4$lambda,6))

coefs.m3   <- coefs.m3   %>% 
  as.matrix() %>% 
  as.data.frame() %>% 
  rownames_to_column() %>% 
  pivot_longer(cols=-rowname) %>%
  mutate(model='Ridge')

coefs.m4   <- coefs.m4   %>% 
  as.matrix() %>% 
  as.data.frame() %>% 
  rownames_to_column() %>% 
  pivot_longer(cols=-rowname) %>%
  mutate(model='Lasso')

# bind rows
coefs1 <- bind_rows(coefs.m3, coefs.m4) %>%
  mutate(name = as.numeric(gsub('lambda', '', name))) %>%
  filter(rowname!='(Intercept)') %>%
  rename(lambda=name, 
         var = rowname)

# lambda values
lambda.lines <- data.frame(model=c('Ridge', 'Lasso'), 
                           lambda.min = c(m3$lambda.min, m4$lambda.min), 
                           lambda.1se = c(m3$lambda.1se, m4$lambda.1se))
dg <- coefs1

g <- ggplot(data=dg, aes(x=lambda, y=value, group=var, color=var))+
  geom_line(alpha=1, linewidth=1)+
  facet_wrap(~model, ncol=1, scales='free_y')+
  geom_vline(data=lambda.lines, aes(xintercept=lambda.min),
             color = pubblue)+
  geom_vline(data=lambda.lines, aes(xintercept=lambda.1se), 
             color = pubmediumgray)+
  scale_x_log10()+
  geom_hline(yintercept = 0)+
  scale_color_manual(values=cb.pal) +
  theme_bw() +
  labs(title='Trace Curves', x='Lambda', y='Coefficient')


## CROSS VALIDATION
## ----------------
# for reproducibility
set.seed(123)

# set number of folds
k=5

# create a vector of fold numbers, repeating 1:k until reaching the number of rows in the data
folds <- rep(1:k, length.out = nrow(dm))

# create a new column in the data frame that randomly assigns a fold to each row without replacement 
# (so that we don't have a row assigned to two or more folds)
dm$fold <- sample(folds, 
                  row(dm), 
                  replace = F)

# create a data frame to keep track of the metrics
metrics <- data.frame(fold = 1:k,
                      llr1se = NA, 
                      llrmin = NA,
                      lllas1se = NA,
                      lllasmin = NA,
                      llm1 = NA,
                      llm2 = NA)

# initialize the prediction columns
dm[, c('r.1se', 'las.1se', 'r.min', 'las.min', 'm1pred', 'm2pred')] = NA

# loop through the folds
for (j in 1:k){
  cat(j,'')
  
  ## train rows are the ones not in the j-th fold
  train.rows <- dm$fold != j
  
  # test rows are the observations in the j-th fold
  test.rows  <- dm$fold == j
  
  ## ======
  ## fit models to training data
  ## ======
  
  ## logistic regression with no regularization using only NDVI100
  m1 <- glm(veg ~ NDVI100, data = dm[train.rows, 1:10], family = "binomial")
  
  ## logistic regression with no regularization using all band values
  m2 <- glm(veg ~ ., data = dm[train.rows,1:10], family = "binomial")
  
  ## logistic regression with ridge regularization using all band values
  r.train <- cv.glmnet(x = x[train.rows,], y = dm$veg[train.rows], family = 'binomial', alpha = 0)
  
  ## logistic regression with lasso regularization using all band values
  las.train <- cv.glmnet(x = x[train.rows,], y = dm$veg[train.rows], family = 'binomial', alpha = 1)
  
  ## ======
  ## make predictions
  ## ======
  
  dm$m1pred[test.rows] = predict(m1, newdata=dm[test.rows,], type='response')
  dm$m2pred[test.rows] = predict(m2, newdata=dm[test.rows,], type='response')
  dm$r.1se[test.rows] = predict(r.train, newx=x[test.rows,], s='lambda.1se', type='response')
  dm$r.min[test.rows] = predict(r.train, newx=x[test.rows,], s='lambda.min', type='response')
  dm$las.1se[test.rows] = predict(las.train, newx=x[test.rows,], s='lambda.1se', type='response')
  dm$las.min[test.rows] = predict(las.train, newx=x[test.rows,], s='lambda.min', type='response')
  
  ## Test logloss for each fold
  metrics[j,'llr1se'] = -mean(dm$veg[test.rows]*
                                log(dm$r.1se[test.rows]) +
                                (1-dm$veg[test.rows])*
                                log(1-dm$r.1se[test.rows]))
  metrics[j,'llrmin'] = -mean(dm$veg[test.rows]*
                                log(dm$r.min[test.rows]) +
                                (1-dm$veg[test.rows])*
                                log(1-dm$r.min[test.rows]))
  metrics[j,'lllas1se'] = -mean(dm$veg[test.rows]*
                                  log(dm$las.1se[test.rows]) +
                                  (1-dm$veg[test.rows])*
                                  log(1-dm$las.1se[test.rows]))
  metrics[j,'lllasmin'] = -mean(dm$veg[test.rows]*
                                  log(dm$las.min[test.rows]) +
                                  (1-dm$veg[test.rows])*
                                  log(1-dm$las.min[test.rows]))
  metrics[j,'llm1'] = -mean(dm$veg[test.rows]*
                              log(dm$m1pred[test.rows]) +
                              (1-dm$veg[test.rows])*
                              log(1-dm$m1pred[test.rows]))
  metrics[j,'llm2'] = -mean(dm$veg[test.rows]*
                              log(dm$m2pred[test.rows]) +
                              (1-dm$veg[test.rows])*
                              log(1-dm$m2pred[test.rows]))
  
}


## LOG LOSS
## --------
logloss <- data.frame(m1 = NA,
                      m2 = NA,
                      m3 = NA,
                      m4 = NA,
                      m1out = NA,
                      m2out = NA,
                      r1se = NA,
                      rmin = NA,
                      las1se = NA,
                      lasmin = NA)

# Log loss of every model (in sample)
logloss$m1 <- (-mean(dm$veg*log(dm$predm1) + (1-dm$veg)*log(1-dm$predm1)))
logloss$m2 <- (-mean(dm$veg*log(dm$predm2) + (1-dm$veg)*log(1-dm$predm2)))
logloss$m3 <- (-mean(dm$veg*log(dm$predm3) + (1-dm$veg)*log(1-dm$predm3)))
logloss$m4 <- (-mean(dm$veg*log(dm$predm4) + (1-dm$veg)*log(1-dm$predm4)))

# Log Loss of every model (out of sample)
logloss$r1se <- (-mean(dm$veg*log(dm$r.1se) + (1-dm$veg)*log(1-dm$r.1se)))
logloss$rmin <- (-mean(dm$veg*log(dm$r.min) + (1-dm$veg)*log(1-dm$r.min)))
logloss$las1se <- (-mean(dm$veg*log(dm$las.1se) + (1-dm$veg)*log(1-dm$las.1se)))
logloss$lasmin <- (-mean(dm$veg*log(dm$las.min) + (1-dm$veg)*log(1-dm$las.min)))
logloss$m1out <- (-mean(dm$veg*log(dm$m1pred) + (1-dm$veg)*log(1-dm$m1pred)))
logloss$m2out <- (-mean(dm$veg*log(dm$m2pred) + (1-dm$veg)*log(1-dm$m2pred)))

logloss <- logloss %>% 
  pivot_longer(cols = everything(), names_to = 'model', values_to = 'logloss') %>% 
  mutate(sample = ifelse(model %in% c('m1', 'm2', 'm3', 'm4'), 'in', 'out'))
