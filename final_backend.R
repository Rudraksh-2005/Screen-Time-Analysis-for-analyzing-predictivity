# Load libraries
library(tidyverse)# For data manipulation (dplyr) and visualization (ggplot2)
library(caret) # For ML preprocessing, train-test split, dummy encodinga
library(xgboost) # To train and use XGBoost model
library(MLmetrics) # For custom metrics like F1-Score
library(ggplot2) # For visualizations
library(reshape2)   # To reshape data for the correlation heatmap


# Ensure final/ directory exists for model and preprocessing objects
dir.create("C:/Users/Lenovo/OneDrive/Documents/College/SEM4/FUNDAMENTALS OF DATA SCIENCE sem4/Screen-Time-Analysis-for-analyzing-predictivity", showWarnings = FALSE)

# Load data
data <- read.csv("C:/Users/Lenovo/OneDrive/Documents/College/SEM4/FUNDAMENTALS OF DATA SCIENCE sem4/Screen-Time-Analysis-for-analyzing-predictivity/data.csv")

# Define target and features explicitly
target <- "Productivity"
feature_cols <- c("Age.Group", "Gender", "Education.Level", "Occupation", 
                  "Average.Screen.Time", "Device", "Screen.Activity", 
                  "App.Category", "Screen.Time.Period", "Notification.Handling")
data <- data[, c(feature_cols, target)]

# Drop rows with NA values
data <- na.omit(data)

# Set target to factor
data[[target]] <- as.factor(data[[target]])

# --- Data Visualization ---

# 1. Target Distribution (Bar Plot)
p_target <- ggplot(data, aes(x = Productivity, fill = Productivity)) +
  geom_bar() +
  theme_minimal() +
  scale_fill_manual(values = c("#FF4C4C", "#FFD700", "#4CAF50")) +
  labs(title = "Distribution of Productivity Classes", x = "Productivity", y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
print(p_target)







# Define the color scheme for Productivity classes (consistent with your code)
productivity_colors <- c("#FF4C4C", "#FFD700", "#4CAF50")

# 1. Notification Handling vs. Productivity (Stacked Bar Plot)
p_notif <- ggplot(data, aes(x = Notification.Handling, fill = Productivity)) +
  geom_bar(position = "fill") +
  theme_minimal() +
  scale_fill_manual(values = productivity_colors) +
  labs(title = "Notification Handling vs. Productivity", 
       x = "Notification Handling", 
       y = "Proportion") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
print(p_notif)

# 2. Average Screen Time vs. Productivity (Stacked Bar Plot)
p_screen <- ggplot(data, aes(x = Average.Screen.Time, fill = Productivity)) +
  geom_bar(position = "fill") +
  theme_minimal() +
  scale_fill_manual(values = productivity_colors) +
  labs(title = "Average Screen Time vs. Productivity", 
       x = "Average Screen Time", 
       y = "Proportion") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
print(p_screen)

# Stacked Bar Plot for App Category vs. Productivity
p_app_category <- ggplot(data, aes(x = App.Category, fill = Productivity)) +
  geom_bar(position = "fill") +
  theme_minimal() +
  scale_fill_manual(values = productivity_colors) +
  labs(title = "App Category vs. Productivity", 
       x = "App Category", 
       y = "Proportion") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
print(p_app_category)


# 4. Boxplots for Average.Screen.Time and Notification.Handling (No Color)
# Boxplot for Average.Screen.Time vs. Productivity
p_box_screen <- ggplot(data, aes(x = Productivity, y = as.numeric(factor(Average.Screen.Time)))) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Average Screen Time vs. Productivity", x = "Productivity", y = "Screen Time (Ordered Categories)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
print(p_box_screen)

# Boxplot for Notification.Handling vs. Productivity
p_box_notif <- ggplot(data, aes(x = Productivity, y = as.numeric(factor(Notification.Handling)))) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Notification Handling vs. Productivity", x = "Productivity", y = "Notification Handling (Ordered Categories)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
print(p_box_notif)




# One-hot encoding (before correlation heatmap)
dummy_vars <- dummyVars(~ ., data = data %>% select(-all_of(target)))
x_data <- predict(dummy_vars, newdata = data %>% select(-all_of(target)))
x_data <- as.data.frame(x_data)

# 4. Correlation Heatmap (Post-Encoding)
cor_matrix <- cor(x_data)
cor_melted <- melt(cor_matrix)
p_cor <- ggplot(cor_melted, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  theme_minimal() +
  labs(title = "Correlation Heatmap of One-Hot Encoded Features") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), 
        axis.text.y = element_text(size = 6))
print(p_cor)

# Save feature names for frontend
feature_names <- colnames(x_data)
save(feature_names, file = "C:/Users/Lenovo/OneDrive/Documents/College/SEM4/FUNDAMENTALS OF DATA SCIENCE sem4/Screen-Time-Analysis-for-analyzing-predictivity/feature_names.Rdata")

# Prepare target
y_data <- data[[target]]
y_labels <- levels(y_data)  # Save class labels

# Train-test split
set.seed(123)
train_index <- createDataPartition(y_data, p = 0.8, list = FALSE)
x_train <- x_data[train_index, ]
x_test <- x_data[-train_index, ]
y_train <- y_data[train_index]
y_test <- y_data[-train_index]

# Convert target to numeric for XGBoost (0-based indexing)
y_train_numeric <- as.numeric(y_train) - 1
y_test_numeric <- as.numeric(y_test) - 1

# Train XGBoost model
xgb_train <- xgb.DMatrix(data = as.matrix(x_train), label = y_train_numeric)
xgb_model <- xgboost(
  data = xgb_train,
  objective = "multi:softprob",
  num_class = length(unique(y_train)),
  nrounds = 100,
  verbose = 0
)

# Save model
saveRDS(xgb_model, "C:/Users/Lenovo/OneDrive/Documents/College/SEM4/FUNDAMENTALS OF DATA SCIENCE sem4/Screen-Time-Analysis-for-analyzing-predictivity/xgb_model.rds")

# 5. Feature Importance (XGBoost)
importance_matrix <- xgb.importance(model = xgb_model)
p_importance <- ggplot(importance_matrix, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_bar(stat = "identity", fill = "#4CAF50") +
  coord_flip() +
  theme_minimal() +
  labs(title = "XGBoost Feature Importance", x = "Feature", y = "Gain") +
  theme(axis.text.y = element_text(size = 6))
print(p_importance)

# Predictions on test set
xgb_test <- xgb.DMatrix(data = as.matrix(x_test))
xgb_predictions <- predict(xgb_model, newdata = xgb_test)
xgb_pred_labels <- factor(max.col(matrix(xgb_predictions, ncol = length(unique(y_train)))) - 1, 
                          labels = y_labels)

# Evaluate model
xgb_conf_matrix <- confusionMatrix(xgb_pred_labels, y_test)
xgb_precision <- mean(xgb_conf_matrix$byClass[, "Precision"], na.rm = TRUE)
xgb_recall <- mean(xgb_conf_matrix$byClass[, "Recall"], na.rm = TRUE)
xgb_f1 <- mean(xgb_conf_matrix$byClass[, "F1"], na.rm = TRUE)

cat("\nXGBoost Model Evaluation:\n")
cat("Accuracy: ", xgb_conf_matrix$overall["Accuracy"], "\n")
cat("Precision: ", xgb_precision, "\n")
cat("Recall: ", xgb_recall, "\n")
cat("F1-Score: ", xgb_f1, "\n")

# Save preprocessing objects
saveRDS(dummy_vars, "C:/Users/Lenovo/OneDrive/Documents/College/SEM4/FUNDAMENTALS OF DATA SCIENCE sem4/Screen-Time-Analysis-for-analyzing-predictivity/dummy_vars.rds")
save(y_labels, file = "C:/Users/Lenovo/OneDrive/Documents/College/SEM4/FUNDAMENTALS OF DATA SCIENCE sem4/Screen-Time-Analysis-for-analyzing-predictivity/y_labels.Rdata")

cat("✅ Model and preprocessing objects saved to `final/`\n")
