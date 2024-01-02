rm(list = ls())

# packages
if (!require("loadflux")){
    install.packages("devtools",repos = "http://cran.us.r-project.org")
    devtools::install_github("atsyplenkov/loadflux")
}
if (!require("dplyr")){
    install.packages("dplyr",repos = "http://cran.us.r-project.org")
}

library(dplyr)
library(loadflux)

# get output folder and input from outside
args = commandArgs(trailingOnly=TRUE)
data_gauge = args[1]
event_list = args[2]
save_plots = args[3]
print(data_gauge)
print(event_list)

# create output file names
stem = gsub(".csv","",event_list)
pdf_file = paste0(stem,"_loadflux_hysteresis_plots.pdf")
csv_file = paste0(stem,"_hysteresis_index.csv")

# discharge and sediment data
df = read.csv(data_gauge)
df$time <- as.POSIXct(df$time,"%Y-%m-%d %H:%M:%OS")
# list of event start and end dates
myEvents = read.csv(event_list)
myEvents$start <- as.POSIXct(myEvents$start)
myEvents$end <- as.POSIXct(myEvents$end)

# plot hysteresis of each event
if (save_plots == "TRUE" | save_plots == "True"){
  pdf(pdf_file)
  for (event_no in 1:nrow(myEvents)){
    myEvent = df[df$time >= myEvents[event_no,]$start & df$time <= myEvents[event_no,]$end,]
    hp <- hysteresis_plot(dataframe = myEvent, q = streamflow, ssc = suspended_sediment)
    print(hp)
  }
  dev.off()
}


# get hysteresis index
hi_simple = c() # empty vector
hi_aich = c() # empty vector
#hi_mid = c() # empty vector

for (event_no in 1:nrow(myEvents)){
  myEvent = df[df$time >= myEvents[event_no,]$start & df$time <= myEvents[event_no,]$end,]
  # calculate hysteresis indeces
  # simple hysteresis index
  shi = tryCatch(SHI(dataframe = myEvent, q = streamflow, ssc = suspended_sediment),error=function(e) e)
  if(inherits(shi, "error")) {
    hi_simple = append(hi_simple,NA)
  } else {
    hi_simple = append(hi_simple,shi)
  }
  # Aich's hysteresis index
  ahi = tryCatch(AHI(dataframe = myEvent, q = streamflow, ssc = suspended_sediment),error=function(e) e)
  if(inherits(ahi, "error")) {
    hi_aich = append(hi_aich,NA)
  } else {
    hi_aich = append(hi_aich,ahi)
  }

  # Mid Hysteresis Index
  #himid = tryCatch(HImid(dataframe = myEvent, q = streamflow, ssc = suspended_sediment),error=function(e) e)
  #if(inherits(himid, "error")) {
  #  hi_mid = append(hi_mid,NA)
  #} else {
  #  hi_mid = append(hi_mid,himid)
  #}
  
}

#hysteresis_index = cbind(myEvents,SHI=hi_simple,AHI=hi_aich,HIMID=hi_mid)
hysteresis_index = cbind(myEvents,SHI=hi_simple,AHI=hi_aich)
write.csv(hysteresis_index,csv_file,row.names = FALSE)
print(paste("Results saved in",csv_file))