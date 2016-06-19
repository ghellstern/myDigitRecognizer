#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)
library(png)
library(colorspace)

ui <- shinyUI(fluidPage(
   
   # Application title
   titlePanel("Digit Recognizer"),
   
   # Sidebar with a file input
   sidebarLayout(
      sidebarPanel(
         fileInput("digitInput",
                     "Please upload a digit in .csv or .png format:",
                   accept = c(".csv", ".png")
                   )
      ),
      
      # Show image of uploaded digit and prediction
      mainPanel(
         plotOutput("digitPlot"),
         textOutput("prediction")
      )
   )
))

# Load r scripts & saved model
source("feedforward.R")
source("predict_digit.R")
trained_parameters <- readRDS("./savedModel/nn_200_10.rds")

# Define server logic required to draw a histogram
server <- shinyServer(function(input, output) {
  
  # Render the uploaded digit based on input$digitInput from ui.R
   output$digitPlot <- renderPlot({
       req(input$digitInput)
       inputPath <- as.character(input$digitInput$datapath[1])
     
      # For .csv format
      if(grepl(".csv", input$digitInput$name[1], ignore.case=TRUE)){
        digit <- read.csv(inputPath, header=FALSE)
        ob <- matrix(as.integer(-digit), nrow=28, ncol=28, byrow=TRUE)
        ob <- t(apply(ob, MARGIN = 2, FUN = rev))
        image(ob, col = grey(seq(0, 1, length = 255)), axes=TRUE)
        title(main="Your digit")
      }
      
      # For .png format
      else if(grepl(".png", input$digitInput$name[1], ignore.case=TRUE)){
        x <- readPNG(inputPath)
        y <- rgb(x[,,1], x[,,2], x[,,3])
        yg <- desaturate(y)
        yn <- col2rgb(yg)[1, ]
        ob <- matrix(as.integer(yn), nrow=28, ncol=28, byrow=FALSE)
        ob <- t(apply(ob, MARGIN = 2, FUN = rev))
        image(ob, col = grey(seq(0, 1, length = 255)), axes=TRUE)
        title(main="Your digit")
      }
   })
   
   # Predict the uploaded digit
   output$prediction <- renderText({
     req(input$digitInput)
     inputPath <- as.character(input$digitInput$datapath[1])
     
     # For .csv format
     if(grepl(".csv", input$digitInput$name[1], ignore.case=TRUE)){
       digit <- read.csv(inputPath, header=FALSE)
       # Normalize pixel to [0,1]
       digit <- (as.integer(digit)+1)/256
       ff <- feedforward(input = matrix(digit, nrow=1), parameter.list = trained_parameters)
       paste("Model Prediction: ", predict_digit(ff$a[[3]]))
     }
     
     # For .png format
     else if(grepl(".png", input$digitInput$name[1], ignore.case=TRUE)){
       x <- readPNG(inputPath)
       y <- rgb(x[,,1], x[,,2], x[,,3])
       yg <- desaturate(y)
       yn <- col2rgb(yg)[1, ]
       digit <- -(yn-255)
       digit <- matrix(digit, nrow=28, byrow = TRUE)
       digit <- (as.integer(digit)+1)/256
       ff <- feedforward(input = matrix(digit, nrow=1), parameter.list = trained_parameters)
       paste("Model Prediction: ", predict_digit(ff$a[[3]]))
     }
   })
})

# Run the application 
shinyApp(ui = ui, server = server)

