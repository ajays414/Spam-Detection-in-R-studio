library(pROC)
library(caret)
library(tm)
library(randomForest)
library(e1071)
library(ggplot2)

# Read the CSV file
data <- read.csv("SPAM text message 20170820 - Data.csv", stringsAsFactors = FALSE)
pie(table(data$Category), 
    labels = c("Ham", "Spam"), 
    col = c("skyblue", "salmon"), 
    main = "Distribution of Ham and Spam Messages")

# Preprocess the text data
corpus <- Corpus(VectorSource(data$Message))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
dtm <- DocumentTermMatrix(corpus)

# Convert the dtm to a matrix
dtm_matrix <- as.matrix(dtm)

#Feature Engineering
dtm_tfidf <- weightTfIdf(dtm)
dtm_ngrams <- DocumentTermMatrix(corpus, control = list(tokenize = NGramTokenizer, ngrams = 2))
dtm_combined <- cbind(as.matrix(dtm_tfidf), as.matrix(dtm_ngrams))

# Train/test split
set.seed(123)
train_index <- createDataPartition(data$Category, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Hyper parametric tuning
svm_grid <- expand.grid(C = seq(0.1, 1, by = 0.1),
                        kernel = c("linear", "radial", "sigmoid"),
                        gamma = c(0.1, 0.01, 0.001))
plot(svm_grid)
svm_tune <- tune(svm, Category ~ ., data = train_data[train_index, ],
                 ranges = list(C = svm_grid$C, kernel = svm_grid$kernel, gamma = svm_grid$gamma),
                 tunecontrol = tune.control(sampling = "cross", cross = 10))
best_svm <- svm_tune$best.parameters

# Train
svm_model <- train(
  x = dtm_matrix[train_index, ],
  y = train_data$Category,
  method = "svmLinear",
  trControl = trainControl(method = "cv"),
  preProcess = c("scale", "center")
)

#Accuracy
svm_accuracy <- svm_model$results$Accuracy
print(paste("SVM Model Accuracy:", svm_accuracy))

str(test_data)
str(svm_model)


# Plotting
resample_results <- svm_model$resample

plot_data <- data.frame(
  Fold = factor(resample_results$Resample, levels = unique(resample_results$Resample)),
  Accuracy = resample_results$Accuracy,
  Kappa = resample_results$Kappa
)

ggplot(plot_data, aes(x = Fold)) +
  geom_point(aes(y = Accuracy, color = "Accuracy"), size = 3) +
  geom_line(aes(y = Accuracy, color = "Accuracy", group = 1), size = 1) + 
 
  scale_color_manual(values = c("Accuracy" = "blue")) +
  labs(title = "Performance across Folds",
       x = "Fold",
       y = "Value") +
  theme_minimal()




# Define the function to preprocess the message
preprocess_message <- function(message) {
  corpus <- Corpus(VectorSource(message))
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removePunctuation)
  dtm <- DocumentTermMatrix(corpus, control = list(dictionary = Terms(dtm)))
dtm_matrix <- as.matrix(dtm)
return(dtm_matrix)
}

# Function to convert numeric prediction to label
get_label <- function(prediction) {
  ifelse(prediction == "spam", "spam", "ham")
}

# Test the model on custom message
custom_message <- c("Reminder: Your appointment with Dr. Smith is scheduled for tomorrow at 10:00 AM.")
custom_dtm <- preprocess_message(custom_message)

# Define the SVM model
svm_model <- svm_model

# Predict the category for custom message
prediction <- predict(svm_model, newdata = custom_dtm)

# Convert prediction to label
predicted_label <- get_label(prediction)

# Output the prediction
print(paste("Predicted category for custom message:", predicted_label))
