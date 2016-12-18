# spinkyDemo

This python package demonstrates the use of the [Matlab toolbox SPINKY](https://github.com/TarekLaj/SPINKY) for the automatic detection of sleep spindles and K-complexes using Python 3 and the [python-matlab-bridge package](https://github.com/arokem/python-matlab-bridge). To use, simply download the example notebook. For example, on a linux machine:  
```
$ wget https://github.com/christian-oreilly/spinkyDemo/blob/master/notebook/finalDemo.ipynb
```
Then, open the downloaded notebook using Jupyter:
```
$ jupyter notebook finalDemo.ipynb
```
The execution of this notebook requires Matlab and Python 3 to be install on your workstation. All other resources and python packages will be downloaded and install (if necessary) by running the notebook. These include:
* Python packages:
  * numpy, scipy, pandas : For manipulating data more easily. 
  * matplotlib, seaborn: For plotting the result.
  * pymatbridge: To interact with the Matlab kernel.
  * gitpython : To clone the SPINKY git repository from Python.
  * spinkyDemo: This package. It is used to embed the code of some utility functions that we import instead of cluttering the notebook with boilerplate code that would make the example harder to follow.
  * requests : To download files from internet.
* SPINKY : The Matlab code of the SPINKY program. 
* EEG + scoring data from the [DREAMS database](http://www.tcts.fpms.ac.be/~devuyst/Databases/DatabaseSpindles/) to demonstrate spindle and K-complex detection.
