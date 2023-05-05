library(deepregression)
library(deeptrafo)

d <- data.frame(y = rnorm(10), x = rnorm(10))
m1 <- BoxCoxNN(y ~ x, data = d)
m2 <- BoxCoxNN(x ~ y, data = d)
for(layer in m2$model$layers)
  layer$`_name` = paste0(layer$name, "_2")
big_mod <- keras_model(inputs = list(m1$model$input, m2$model$input),
                       outputs = list(m1$model$output, m2$model$output))

big_mod
