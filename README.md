# SodShockTube
A comparison of Sod's shock tube as simulated by [Aither](https://github.com/mnucci32/aither) with the exact solution.

The **sod.py** script reads in three CSV data files that were generated using Aither and [ParaView](http://www.paraview.org). 
The data files show the results for a simulaton of Sod's shock tube with constant reconstruction, MUSCL reconstruction, and 
WENO reconstruction. These results are compared to the exact solution. A blog post detailing the simulation and results can 
be found [here](http://aithercfd.com/2017/01/29/sod-shock-tube.html). 
