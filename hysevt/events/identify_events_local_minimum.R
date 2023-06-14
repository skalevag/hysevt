rm(list = ls())

# packages
if (!require("loadflux")){
  install.packages("devtools")
  devtools::install_github("atsyplenkov/loadflux")
}
if (!require("dplyr")){
  install.packages("dplyr")
}

library(dplyr)
library(loadflux)

# get output folder and input from outside
args = commandArgs(trailingOnly=TRUE)
data_gauge = args[1]
outfile = args[2]
print(data_gauge)

# discharge and sediment data
df = read.csv(data_gauge)
df$time <- as.POSIXct(df$time)

# identify new events
myHydroEvents <- hydro_events(
  dataframe = df,
  q = streamflow,
  datetime = time,
  window = 21
)

# get the start and end of identified events
# get hysteresis index
eventStart = c() # empty vector
eventEnd = c() # empty vector

for (event_no in unique(myHydroEvents$he)){
  # extract event only
  myEvent = myHydroEvents[myHydroEvents$he==event_no,]
  # append end
  eventEnd = append(eventEnd,tail(myEvent$time,1))
  # append start
  eventStart = append(eventStart,head(myEvent$time,1))
}
# merge to dataframe
event_list = data.frame(start=eventStart,end=eventEnd)

# save to file
dir.create(dirname(outfile),recursive=TRUE)
write.csv(event_list,outfile,row.names = FALSE)
print(paste("Results saved in",outfile))