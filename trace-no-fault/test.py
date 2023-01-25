import numpy as np
import pandas as pd
import os
import pickle
import json
import string


def read_json(filepath):
    if os.path.exists(filepath):
        assert filepath.endswith('.json')
        with open(filepath, 'r') as f:
            return json.loads(f.read())
    else: 
        logging.raiseExceptions("File path "+filepath+" not exists!")
        return


zeors_array = np.zeros((2, 3))
A = zeors_array.flatten()
string1 = "1650297414" + ":"
for i in range(A.size):
   string1 = string1 + str(int(A[i])) + ','
string1 = string1.strip( ',' )
print(string1)




"""
invok_list = []
trace_json = read_json("trace_invok_lat-3.json")
for key in trace_json.keys():
    for invok in trace_json[key].keys():
        if invok not in invok_list:
            invok_list.append(invok)
print(invok_list)

print("number of timestamp:",len(trace_json))


start = 16502970
end = 16502990
#end = 1650301097971000
chunk_lenth = 10

intervals = [(s, s+chunk_lenth-1) 
             for s in range(start, end-chunk_lenth+1)]

for chunk_idx, (s, e) in enumerate(intervals):
    slots = [t for t in range(s, e+1)]
print(slots)
#print(intervals)
"""