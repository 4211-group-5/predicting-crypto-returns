library(dplyr)
library(randomForest)
library(fastshap)


# 1. LOAD DATA + RESULTS

df <- read.csv("datasets/10_dropped_variables_and_renamed.csv")

df$week <- as.Date(df$week, format="%d/%m/%Y")

best_results <- readRDS("results/random_forest/results_all_coins_15000.rds")


# 2. SETTINGS

coins_to_run <- c("BCH", "BTC", "DASH", "EOS", "ETC", "ETH",
                  "IOTA", "LTC", "OMG", "XMR", "XRP", "ZEC")  

train_start <- as.Date("2018-01-01")
train_end   <- as.Date("2022-12-30")
test_end    <- as.Date("2026-01-08")


df_BCH <- df %>% filter(coin == "BCH")

train_BCH_num <- nrow(df_BCH %>% filter(week <= train_end, week >= train_start))
test_BCH_num  <- nrow(df_BCH %>% filter(week > "2024-12-30", week <= test_end))


# 3. SHAP FUNCTION 

runrf_shap <- function(Y, indice, ntree, mtry) {
  
  y <- Y[, indice]
  X <- Y[, -indice]
  
  X.train <- X[-nrow(X), ]
  y.train <- y[-length(y)]
  X.out <- X[nrow(X), , drop = FALSE]
  
  model <- randomForest(
    x = X.train,
    y = y.train,
    ntree = ntree,
    mtry = mtry,
    importance = TRUE
  )
  
  shap_values <- fastshap::explain(
    model,
    X = X.train,
    newdata = X.out,
    pred_wrapper = function(object, newdata) {
      predict(object, newdata)
    },
    nsim = 50
  )
  
  return(shap_values)
}


# 4. ROLLING SHAP LOOP

results_shap <- list()

for (coin in coins_to_run) {
  
  cat("Running SHAP for:", coin, "\n")
  
  best_ntree <- best_results[[coin]]$best_ntree
  best_mtry  <- best_results[[coin]]$best_mtry
  
  coin_df <- df %>%
    filter(coin == coin) %>%
    arrange(week)
  
  preds_df <- coin_df %>%
    filter(week >= train_start, week <= test_end) %>%
    tail(test_BCH_num + train_BCH_num) %>%
    select(-c("week", "coin"))
  
  Y <- as.matrix(preds_df)
  
  shap_list <- list()
  
  for (i in test_BCH_num:1) {
    
    Y_window <- Y[(1 + test_BCH_num - i):(nrow(Y) - i), ]
    
    shap_val <- runrf_shap(Y_window, indice = 1, best_ntree, best_mtry)
    
    idx <- (1 + test_BCH_num - i)
    shap_list[[idx]] <- shap_val
    
    cat("iteration", idx, "\n")
  }
  
 
  # 5. AGGREGATE SHAP

  shap_matrix <- do.call(rbind, shap_list)
  
  shap_importance <- colMeans(abs(shap_matrix))
  shap_direction  <- colMeans(shap_matrix)
  
  shap_summary <- data.frame(
    feature = names(shap_importance),
    mean_abs_shap = shap_importance,
    mean_shap = shap_direction
  ) %>%
    arrange(desc(mean_abs_shap))
  
  print(head(shap_summary, 10))
  
  results_shap[[coin]] <- shap_summary
  
  # save per coin
  write.csv(shap_summary, paste0("shap_summary_", coin, ".csv"), row.names = FALSE)
}


# 6. COMBINE ALL COINS

final_shap <- bind_rows(results_shap, .id = "coin")

write.csv(final_shap, "shap_summary_all_coins.csv", row.names = FALSE)
