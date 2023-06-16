# Hydro-sediment event types

Detection, characterisation, clustering and type identification of hydro-sediment events. Only input for detection and clustering is sub-daily time series data of both streamflow (discharge) and (suspended) sediment concentration. To interpret the identified event clusters into *event types*, i.e. relating to certain processes or conditions requires expert knowledge and additional data.

A manuscript is in development.

## Getting started

### Installation
**The package requires python and R.**

Most of the package is in **python**. However, the event detection routine and calculation of the hysteresis indices for the event characterisation requires **R**. Therefore please have the latest version of R installed on your computer. You can download R [here](https://cran.r-project.org).

Install package:
```
git clone https://gitup.uni-potsdam.de/skalevag2/hysevt.git
cd hysevt
pip install .
```

Alternatively, the package can be installed in "development" mode. This is useful if you plan to make changes to the module code.
```
git clone https://gitup.uni-potsdam.de/skalevag2/hysevt.git
cd hysevt
pip install -e .
```

### Demo notebooks
There are several notebooks demonstrating the use of the package under `demo/`.

- **Event detection** using *loadflux* [R-package](atsyplenkov.github.io/loadflux)
- **Event characterisation** with metrics, based on SSC and Q time series of each event
- **Clustering** events based on metrics, using GMM or another approach
- **Evaluation** of metrics clustering

Additionally, there are notebooks showing the use of METS clustering, to identify events (see [Javed et al., 2021](https://doi.org/10.1016/j.jhydrol.2020.125802) for details).

- METS preprocessing: standardising event magnitude and length for the METS cluster alghorithm
- METS clustering: clustering events based on the shape of **M**ulitvariate **E**vent **T**ime **S**eries

### Calling the scripts from the terminal
The routines for characterising events can be called from the terminal.

```
python hysevt/events/metrics.py -g SSC-Q-timeseries.csv -e events.csv -s annual_sediment_yield.csv -q annual_water_yield.csv -o output.csv
```


## Contributing
Contributions and applications in research is welcome. Please don't hesitate to contact me if you want to use the module in you analysis.

## Authors and acknowledgment
Amalie Skålevåg, skalevag2@uni-potsdam.de

## License
GNU General Public License version 3


## Potential issues and limitations
Currently the R scripts are called from python by calling on the R script in the terminal with csv-files of the timeseries as input. This has **only** been tested on a Linux operating system. Should you run into problems related to this, a workaround would be to make a copy of the R-script, open this in your preferred IDE, e.g. R-Studio, and running the analysis directly yourself. **WARNING**: This will only get the *hydro events* based on the streamflow. 


## Project status

- Manuscript with description and application to an alpine catchment in the European Alps are in development.
- **Current version will likely have bugs related to versioning and operating systems.**


### Planned future developments

-[] Add tests
-[] Add demo-notebooks
