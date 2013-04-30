chorro
======

Chorro is an effort to make effortless one-page webapps from Python
simulations. It depends on the Flask web microframework, and uses server-side
events for realtime snapshots of in-progress simulations.

This effort is still heavily under development and should be considered more of
a proof-of-concept or a design statement as opposed to a real software release.
Further work needs to be done to abstract out many of the details involved.

The intended workflow goes something like this:

0.  Import a previously-built simulation
1.  Define the numerical parameters for the simulation
2.  Wrap your simulation in a generator function that yields snapshots of the
simulation state.

This repo has two examples of this workflow/framework. The first is a simple
example implementation of Conway's game of life. That app depends on Numpy and
Scipy. The second example, geoCPM, has a much longer list of dependencies
including Numba, pandas, matplotlib, and networkx. It is an implementation of a
large-q Potts model for algorithmically assigning voting districts based on
some objective criteria.

To take a look at either example, check out the repo, navigate to the
appropriate subfolder, and run the '\_webapp.py'. Then navigate to
'localhost:12345/life' or 'localhost:12345/geocpm' in your browser of choice.
