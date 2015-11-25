# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 23:28:53 2015

@author: konrad
"""

import pandas as pd
import numpy as np
import os
import subprocess
from collections import defaultdict, Counter
from datetime import datetime, date
from csv import DictReader
import math
from glob import glob
import copy

CHUNK_SIZE = 1000

def to_vw(loc_csv, loc_out, train=True, threshold = 2):
  print("\nConverting %s"%loc_csv)
  with open(loc_out,"wb") as outfile:
    
    for linenr, row in enumerate( DictReader(open(loc_csv,"rb")) ):
      # initialize strings for particular namespaces: coverage,
      # sales, personal, property, geography, field
      n_c = n_s = n_pe = n_pr = n_g = n_f = ""
      
      for k in row:
        if k != "id":
	  # coverage features
          if k in coverage_features:
            xnam = k.replace("CoverageField","CF"); xval = row[k]
            n_c += " %s_%s"%(xnam,xval)
	  # sales features
          if k in sales_features:
            xnam = k.replace("SalesField","SF"); xval = row[k]
            n_s += " %s_%s"%(xnam,xval)
        # personal features
          if k in personal_features:
            xnam = k.replace("PersonalField","PF"); xval = row[k]
            n_pe += " %s_%s"%(xnam,xval)
        # property features
          if k in property_features:
            xnam = k.replace("PropertyField","PR"); xval = row[k]
            n_pr += " %s_%s"%(xnam,xval)    
        # geographical features
          if k in geography_features:
            xnam = k.replace("GeographicField","GF"); xval = row[k]
            n_g += " %s_%s"%(xnam,xval)     
        # field features
          if k in field_features:
            xnam = k.replace("Field","F"); xval = row[k]
            n_f += " %s_%s"%(xnam,xval)     
            
            
      if train:
        label = 2 * int(row["QuoteConversion_Flag"]) - 1
      else:
        label = 1

      id = row["QuoteNumber"]
      outfile.write("%s '%s |c%s |s%s |p%s |r%s |g%s |f%s \n"%(label,id,n_c, n_s, n_pe, n_pr, n_g, n_f) )
      if linenr % (2 * CHUNK_SIZE) == 0:
        print("%s"%(linenr))        
        
# Data locations
projPath = '/Users/konrad/Documents/projects/homesite'
loc_train = projPath + '/input/train.csv'
loc_test = projPath + "/input/test.csv"
loc_train_vw = projPath + "/input/xtrain.vw"
loc_test_vw = projPath + "/input/xtest.vw"

# groups of features
coverage_features = ["CoverageField1A","CoverageField1B","CoverageField2A","CoverageField2B",
                           "CoverageField3A","CoverageField3B","CoverageField4A","CoverageField4B",
                           "CoverageField5A","CoverageField5B","CoverageField6A","CoverageField6B",
                           "CoverageField8","CoverageField9","CoverageField11A","CoverageField11B"]                          
sales_features = ["SalesField1A","SalesField1B","SalesField2A","SalesField2B","SalesField3",
                  "SalesField4","SalesField5","SalesField6","SalesField7","SalesField8",
                  "SalesField9","SalesField10","SalesField11","SalesField12","SalesField13",
                  "SalesField14","SalesField15"]                           
personal_features = ["PersonalField1","PersonalField2","PersonalField4A","PersonalField4B",
                     "PersonalField5","PersonalField6","PersonalField7","PersonalField8",
                     "PersonalField9","PersonalField10A","PersonalField10B","PersonalField11",
                     "PersonalField12","PersonalField13","PersonalField14","PersonalField15",
                     "PersonalField16","PersonalField17","PersonalField18","PersonalField19",
                     "PersonalField22","PersonalField23","PersonalField24","PersonalField25",
                     "PersonalField26","PersonalField27","PersonalField28","PersonalField29",
                     "PersonalField30","PersonalField31","PersonalField32","PersonalField33",
                     "PersonalField34","PersonalField35","PersonalField36","PersonalField37",
                     "PersonalField38","PersonalField39","PersonalField40","PersonalField41",
                     "PersonalField42","PersonalField43","PersonalField44","PersonalField45",
                     "PersonalField46","PersonalField47","PersonalField48","PersonalField49",
                     "PersonalField50","PersonalField51","PersonalField52","PersonalField53",
                     "PersonalField54","PersonalField55","PersonalField56","PersonalField57",
                         "PersonalField58","PersonalField59","PersonalField60","PersonalField61",
                         "PersonalField62","PersonalField63","PersonalField64","PersonalField65",
                         "PersonalField66","PersonalField67","PersonalField68","PersonalField69",
                         "PersonalField70","PersonalField71","PersonalField72","PersonalField73",
                         "PersonalField74","PersonalField75","PersonalField76","PersonalField77",
                         "PersonalField78","PersonalField79","PersonalField80","PersonalField81",
                         "PersonalField82","PersonalField83","PersonalField84"]                  
property_features = ["PropertyField1A",
                     "PropertyField1B","PropertyField2A","PropertyField2B","PropertyField3",
                     "PropertyField4","PropertyField5","PropertyField6","PropertyField7",
                     "PropertyField8","PropertyField9","PropertyField10","PropertyField11A",
                     "PropertyField11B","PropertyField12","PropertyField13","PropertyField14",
                     "PropertyField15","PropertyField16A","PropertyField16B","PropertyField17",
                     "PropertyField18","PropertyField19","PropertyField20","PropertyField21A",
                     "PropertyField21B","PropertyField22","PropertyField23","PropertyField24A",
                     "PropertyField24B","PropertyField25","PropertyField26A","PropertyField26B",
                     "PropertyField27","PropertyField28","PropertyField29","PropertyField30",
                     "PropertyField31","PropertyField32","PropertyField33","PropertyField34",
                     "PropertyField35","PropertyField36","PropertyField37","PropertyField38",
                     "PropertyField39A","PropertyField39B"]
geography_features = ["GeographicField1A","GeographicField1B","GeographicField2A","GeographicField2B",
                      "GeographicField3A","GeographicField3B","GeographicField4A","GeographicField4B",
                      "GeographicField5A","GeographicField5B","GeographicField6A","GeographicField6B",
                      "GeographicField7A","GeographicField7B","GeographicField8A","GeographicField8B",
                      "GeographicField9A","GeographicField9B","GeographicField10A","GeographicField10B",
                      "GeographicField11A","GeographicField11B","GeographicField12A","GeographicField12B",
                      "GeographicField13A","GeographicField13B","GeographicField14A","GeographicField14B",
                      "GeographicField15A","GeographicField15B","GeographicField16A","GeographicField16B",
                      "GeographicField17A","GeographicField17B","GeographicField18A","GeographicField18B",
                      "GeographicField19A","GeographicField19B","GeographicField20A","GeographicField20B",
                      "GeographicField21A","GeographicField21B","GeographicField22A","GeographicField22B",
                      "GeographicField23A","GeographicField23B","GeographicField24A","GeographicField24B",
    "GeographicField25A","GeographicField25B","GeographicField26A","GeographicField26B",
    "GeographicField27A","GeographicField27B","GeographicField28A","GeographicField28B",
    "GeographicField29A","GeographicField29B","GeographicField30A","GeographicField30B",
    "GeographicField31A","GeographicField31B","GeographicField32A","GeographicField32B",
    "GeographicField33A","GeographicField33B","GeographicField34A","GeographicField34B",
    "GeographicField35A","GeographicField35B","GeographicField36A","GeographicField36B",
    "GeographicField37A","GeographicField37B","GeographicField38A","GeographicField38B",
    "GeographicField39A","GeographicField39B","GeographicField40A","GeographicField40B",
    "GeographicField41A","GeographicField41B","GeographicField42A","GeographicField42B",
    "GeographicField43A","GeographicField43B","GeographicField44A","GeographicField44B",
    "GeographicField45A","GeographicField45B","GeographicField46A","GeographicField46B",
    "GeographicField47A","GeographicField47B","GeographicField48A","GeographicField48B",
    "GeographicField49A","GeographicField49B","GeographicField50A","GeographicField50B",
    "GeographicField51A","GeographicField51B","GeographicField52A","GeographicField52B",
    "GeographicField53A","GeographicField53B","GeographicField54A","GeographicField54B",
    "GeographicField55A","GeographicField55B","GeographicField56A","GeographicField56B",
    "GeographicField57A","GeographicField57B","GeographicField58A","GeographicField58B",
    "GeographicField59A","GeographicField59B","GeographicField60A","GeographicField60B",
    "GeographicField61A","GeographicField61B","GeographicField62A","GeographicField62B",
    "GeographicField63","GeographicField64"]
field_features = ["Field6","Field7","Field8","Field9","Field10","Field11","Field12"]

# process train and test
if __name__ == "__main__":
    to_vw(loc_train, loc_train_vw, train=True)
    to_vw(loc_test, loc_test_vw, train=False)




