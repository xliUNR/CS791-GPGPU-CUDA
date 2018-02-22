# load libraries
import pandas as pd 
import pycuda.driver as cuda
import pycuda.autinit
from pycuda.compiler import SourceModule

# = SourceModule()
inputData = pandas.read_csv('PA3_nrdc_data.csv')

