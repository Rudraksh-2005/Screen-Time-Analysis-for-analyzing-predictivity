library(shiny)
library(ggplot2)
library(dplyr)
library(xgboost)
library(caret)
library(readr)

# UI
ui <- fluidPage(
  titlePanel("📱 Personalized Productivity Analysis Dashboard"),
  
  tags$head(
    tags$style(HTML("
      body {
        background-color: #f9f9ff;
        color: #2c3e50;
      }
      h2, h3 {
        color: #4b0082;
      }
      .well {
        background-color: #f0f8ff;
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
      }
    "))
  ),
  
  sidebarLayout(
    sidebarPanel(
      h3("🧠 Answer a few quick questions"),
      
      selectInput("age", "Your Age Group:",
                  choices = c("Below 18", "18–24", "25–34", "35–44", "45 and above")),
      
      selectInput("gender", "Your Gender:",
                  choices = c("Male", "Female")),
      
      selectInput("education", "Your Education Level:",
                  choices = c("High school or below", "Undergraduate", "Graduate")),
      
      selectInput("occupation", "Your Occupation:",
                  choices = c("Student", "Professional")),
      
      selectInput("screen_time", "Your Avg Daily Screen Time:",
                  choices = c("Less than 2", "2–4", "4–6", "6–8", "8-10", "More than 10")),
      
      selectInput("device", "Primary Device:",
                  choices = c("Smartphone", "Laptop/PC", "Tablet", "Television")),
      
      selectInput("screen_activity", "Main Screen Activity:",
                  choices = c("Entertainment (gaming, streaming, social media, etc.)", "Academic/Work-related")),
      
      selectInput("app_category", "Primary App Category:",
                  choices = c("Social Media (e.g., Facebook, Instagram, LinkedIn, Twitter)", 
                              "Streaming (e.g., YouTube, Netflix)", 
                              "Productivity (e.g., Microsoft Office, Notion)", 
                              "Messaging (e.g., WhatsApp, Messenger)", 
                              "Gaming")),
      
      selectInput("screen_time_period", "When do you use your device most?:",
                  choices = c("Morning (6 AM–12 PM)", "Afternoon (12 PM–6 PM)", 
                              "Evening (6 PM–10 PM)", "Late night (10 PM–6 AM)")),
      
      selectInput("notif", "How do you handle notifications?:",
                  choices = c("Check them briefly and resume my work", 
                              "Ignore them until my task is completed", 
                              "Spend time interacting with the notifications", 
                              "Turn off notifications altogether")),
      
      actionButton("analyze", "Analyze My Productivity", class = "btn-primary")
    ),
    
    mainPanel(
      h2("📊 Personalized Insights"),
      verbatimTextOutput("result_text"),
      plotOutput("prob_plot"),
      br(),
      h3("🌟 Recommendations"),
      uiOutput("recommendation")
    )
  )
)

server <- function(input, output, session) {
  
  # Load the pre-trained XGBoost model, dummyVars object, feature names, and class labels
  xgb_model <- readRDS("C:/Users/Lenovo/OneDrive/Documents/College/SEM4/FUNDAMENTALS OF DATA SCIENCE sem4/Screen-Time-Analysis-for-analyzing-predictivity/xgb_model.rds")
  dummy_vars <- readRDS("C:/Users/Lenovo/OneDrive/Documents/College/SEM4/FUNDAMENTALS OF DATA SCIENCE sem4/Screen-Time-Analysis-for-analyzing-predictivity/dummy_vars.rds")
  load("C:/Users/Lenovo/OneDrive/Documents/College/SEM4/FUNDAMENTALS OF DATA SCIENCE sem4/Screen-Time-Analysis-for-analyzing-predictivity/feature_names.Rdata")
  load("C:/Users/Lenovo/OneDrive/Documents/College/SEM4/FUNDAMENTALS OF DATA SCIENCE sem4/Screen-Time-Analysis-for-analyzing-predictivity/y_labels.Rdata")
  
  observeEvent(input$analyze, {
    
    # Prepare user input data with exact column names and levels as backend
    user_data <- data.frame(
      Age.Group = input$age,
      Gender = input$gender,
      Education.Level = input$education,
      Occupation = input$occupation,
      Average.Screen.Time = input$screen_time,
      Device = input$device,
      Screen.Activity = input$screen_activity,
      App.Category = input$app_category,
      Screen.Time.Period = input$screen_time_period,
      Notification.Handling = input$notif
    )
    
    # Ensure factor levels align with training data
    factor_levels <- list(
      Age.Group = c("Below 18", "18–24", "25–34", "35–44", "45 and above"),
      Gender = c("Male", "Female"),
      Education.Level = c("High school or below", "Undergraduate", "Graduate"),
      Occupation = c("Student", "Professional"),
      Average.Screen.Time = c("Less than 2", "2–4", "4–6", "6–8", "8-10", "More than 10"),
      Device = c("Smartphone", "Laptop/PC", "Tablet", "Television"),
      Screen.Activity = c("Entertainment (gaming, streaming, social media, etc.)", "Academic/Work-related"),
      App.Category = c("Social Media (e.g., Facebook, Instagram, LinkedIn, Twitter)", 
                       "Streaming (e.g., YouTube, Netflix)", 
                       "Productivity (e.g., Microsoft Office, Notion)", 
                       "Messaging (e.g., WhatsApp, Messenger)", 
                       "Gaming"),
      Screen.Time.Period = c("Morning (6 AM–12 PM)", "Afternoon (12 PM–6 PM)", 
                             "Evening (6 PM–10 PM)", "Late night (10 PM–6 AM)"),
      Notification.Handling = c("Check them briefly and resume my work", 
                                "Ignore them until my task is completed", 
                                "Spend time interacting with the notifications", 
                                "Turn off notifications altogether")
    )
    
    for (col in names(factor_levels)) {
      user_data[[col]] <- factor(user_data[[col]], levels = factor_levels[[col]])
    }
    
    # Transform input using dummyVars
    user_data_dummies <- predict(dummy_vars, newdata = user_data)
    user_data_dummies <- as.data.frame(user_data_dummies)
    
    # Align with feature_names
    for (col in setdiff(feature_names, colnames(user_data_dummies))) {
      user_data_dummies[[col]] <- 0
    }
    user_data_dummies <- user_data_dummies[, feature_names, drop = FALSE]
    
    # Make prediction
    xgb_input <- xgb.DMatrix(data = as.matrix(user_data_dummies))
    pred_probs <- predict(xgb_model, newdata = xgb_input)
    pred_class_idx <- max.col(matrix(pred_probs, ncol = length(y_labels))) - 1
    pred_class <- y_labels[pred_class_idx + 1]
    
    # Display prediction
    output$result_text <- renderPrint({
      cat("📌 Your Predicted Productivity Level: ", pred_class, "\n")
    })
    
    # Plot probabilities
    output$prob_plot <- renderPlot({
      df <- data.frame(
        Class = y_labels,
        Probability = pred_probs * 100
      )
      ggplot(df, aes(x = Class, y = Probability, fill = Class)) +
        geom_bar(stat = "identity") +
        ylim(0, 100) +
        ylab("Probability (%)") +
        theme_minimal() +
        scale_fill_manual(values = c("#FF4C4C", "#FFD700", "#4CAF50")) +
        geom_text(aes(label = paste0(round(Probability), "%")), vjust = -0.5, size = 5) +
        theme(legend.position = "none")
    })
    
    # Recommendations based on predicted class
    output$recommendation <- renderUI({
      if (pred_class == "Unproductive, i might not have completed the task and got carried away") {
        HTML(paste(
          "<b>🔴 Low Productivity Zone - Here's what you can do to improve:</b>",
          "1. ⏳ Reduce screen time by setting daily limits on entertainment apps.",
          "2. 🚫 Turn off non-essential notifications to minimize distractions.",
          "3. 🍅 Try the Pomodoro Technique: 25 mins work, 5 mins break.",
          "4. 📵 Use Focus Mode or app blockers during work hours.",
          "5. 🧘‍♂ Start your day with 10 mins of mindfulness to boost focus.",
          "6. 📓 Track your habits with a digital wellness journal.",
          sep = "<br>")
        )
      } else if (pred_class == "Moderately productive") {
        HTML(paste(
          "<b>🟡 Moderate Productivity Zone - You're on the right path:</b>",
          "1. 📊 Review your screen time weekly to identify distractions.",
          "2. 🎯 Use Time Blocking with short breaks to maintain focus.",
          "3. 📵 Keep your device out of sight during deep work sessions.",
          "4. 💡 Pair productivity tasks with existing routines (habit stacking).",
          "5. ✅ Plan tomorrow’s tasks at the end of each day.",
          "6. 🔁 Stay consistent with small daily improvements.",
          sep = "<br>")
        )
      } else {
        HTML(paste(
          "<b>🟢 High Productivity Zone - You're thriving! To stay consistent:</b>",
          "1. 🧠 Continue using effective strategies like Time Blocking.",
          "2. 📶 Periodically check app usage to maintain balance.",
          "3. 🧘‍♀ Keep up routines like journaling or meditation.",
          "4. 🔄 Rotate productivity techniques to prevent burnout.",
          "5. 🛑 Balance work with rest to sustain performance.",
          "6. 🚀 Share your productivity tips with peers to reinforce habits.",
          sep = "<br>")
        )
      }
    })
  })
}

# Run the app
shinyApp(ui = ui, server = server)
