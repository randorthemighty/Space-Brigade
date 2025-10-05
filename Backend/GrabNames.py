import requests
import json
import numpy as np
import pandas as pd
import Backend.read as read
#########################
fp = file_path = ""
file = pd.read(fp)
# API Request -> list of names

def GetAsteroidNames(url):
    