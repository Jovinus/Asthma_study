library(stats)
library(readxl)
library(xlsx)
library(data.table)
library(tidyverse)
library(dplyr)
library(moonBook)

setwd("/Users/kyunghyunlee/Studio/Asthma_study/AI/")
metrics <- fread('./result/selected_metric/selected_metric.csv')
View(metrics)

results <- pairwise.t.test(x = metrics$test_roc_auc, g = metrics$feature, p.adjust.method = "bonferroni")
results

output_table <- mytable(feature ~ test_auroc + test_auprc + test_accuracy + test_sensitivy + test_specificity , data=metrics, digits=2)
output_table <- mytable(feature ~ . , data=metrics, digits=2)
output_table <- summary(output_table)

mycsv(x=output_table, file="./descriptive_metrics.csv")

metrics <- fread('./descriptive_metrics.csv')
write.xlsx(x=metrics, file="./descriptive_metrics.xlsx")