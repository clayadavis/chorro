#geoCPM

This is an implementation of a large-q Potts model for algorithmically assigning voting districts based on some objective criteria.
It has a lengthy list of dependencies including Numba, pandas, matplotlib, and networkx. 

Notably, this folder contains an updated version of pyshp (shapefile.py) which deals with missing data and is able to load data into a pandas DataFrame. 
Also contained in the geoCPM folder are iPython notebooks documenting some of the algorithms used in the geoCPM code and the speedups gained with Numba.
