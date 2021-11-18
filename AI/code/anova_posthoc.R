setwd("/home/lkh256/Studio/Asthma/AI/code/")

library(data.table)
library(moonBook)

df_orig <- fread("../result/selected_metric/selected_metric.csv")
View(df_orig)

results <- pairwise.t.test(x = df_orig$test_roc_auc, g = df_orig$feature, p.adjust.method = "bonferroni")
View(results)

results$p.value
