chorro
======

Chorro is an effort to make effortless one-page webapps from Python
simulations. It depends on the Flask web microframework, and uses server-side
events for realtime snapshots of in-progress simulations.

Currently there is only a simple example of the approach using Conway's game of
life. That app depends on Numpy and Scipy. To take a look, check it out and
then just run 'python life-webapp.py' and navigate to localhost:12345/life in your browser of choice.

Soon to follow is an implementation of GeoCPM in this framework.
