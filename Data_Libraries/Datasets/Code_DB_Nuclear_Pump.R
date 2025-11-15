pump_data <- data.frame(
  Pump = 1:10,
  Failures = c(5, 1, 5, 14, 3, 19, 1, 1, 4, 22),
  Time = c(94.32, 15.72, 62.88, 125.76, 5.24, 31.44, 1.048, 1.048, 2.096, 10.48)
)

file_path <- "C:/Users/mussi/Documents/Manhattan/Leuven/MCQMC/Data_Libraries/DB_Nuclear_Pumps.csv"
write.csv(pump_data, file = file_path, row.names = FALSE)

