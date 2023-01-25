
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


span_data = {
    "spanID":'',
    "parentSpanID":'',
    "duration":'',
    "serviceName":'',
    "caller":'',
    "callee":''
}

#get a span-level file
def trace_info_extract_func(filepath, index):
    print("*** Extracting key features from the raw spans data...")
    spans_json = read_json(filepath)

    service_list = []
    for trace in spans_json:
        process_dict = trace["processes"]
        for key in process_dict:
            service_list.append(process_dict[key]["serviceName"])
    #去重
    service_list = list(set(service_list))
    #print(service_list)

    #为每个service生成全局ID
    i = 1
    service_dict = {}
    for service in service_list:
        if i == len(service_list)+1: break
        service_dict[service] = i
        i += 1
    print("service_dict:", service_dict)

    trace_dict = {}
    for trace in spans_json:
        #遍历提取service_name
        process_dict = trace["processes"]
        service_name_dict = {}
        for key in process_dict:
            service_name_dict[key] = process_dict[key]["serviceName"]
        #print(service_name_dict)
              
        #遍历提取重要的span信息
        span_list = []
        for span in trace["spans"]:
            span_data = {}
            span_data["spanID"] = span["spanID"]
            span_data["sendingTime"] = round(span["startTime"]/1000)
            span_data["duration"] = (span["duration"]/1000)
            if span["references"]:
                span_data["parentSpanID"] = span["references"][0]["spanID"]
            else:
                # The list is empty
                span_data["parentSpanID"] = ""
            span_data["serviceName"] = service_name_dict[span["processID"]]
            span_data["callee"] = service_dict.get(service_name_dict[span["processID"]])
            span_data["caller"] = ""
            span_list.append(span_data)
            #print(span_data)
        #print(span_list)
        #print("\n")
        trace_dict[trace["traceID"]] = span_list
    
        for key in trace_dict.keys():
            for span in trace_dict[key]:
                if span["parentSpanID"] == "":
                    continue
                else:
                    for par_span in trace_dict[key]:
                        if span["parentSpanID"] == par_span["spanID"]:
                            span["caller"] = par_span["callee"]

    #print(trace_dict)
    with open("trace_info-" + index + ".json", "w") as json_file:
        json.dump(trace_dict, json_file)
        print("trace_info-" + index + ".json write done!")
    return trace_dict


'''
trace_lat_dic = {
	"slot1": {
		"1-1": [lat1, lat2, lat3],
		"2-2": [lat1, lat2, lat3, lat4],
		"3-1": [lat1]
	},
	"slot2": {
		"1-1": [lat1, lat2, lat3],
		"2-2": [lat1, lat2, lat3, lat4],
		"3-1": [lat1]
	}
}
'''

def get_trace_invok_lat_func(trace_dict, index):
    trace_lat_dic = {}
    for key in trace_dict.keys():
        for span in trace_dict[key]:
            if span["sendingTime"] not in trace_lat_dic.keys():
                trace_lat_dic[span["sendingTime"]] = {}
            if span["caller"] == "":
                continue
            invok = str(span["caller"]) + '-' + str(span["callee"])
            if invok not in trace_lat_dic[span["sendingTime"]].keys():
                trace_lat_dic[span["sendingTime"]][invok] = []           
            trace_lat_dic[span["sendingTime"]][invok].append(span["duration"])
    for key in list(trace_lat_dic.keys()):
        if trace_lat_dic[key] == {}: 
            del trace_lat_dic[key]
    with open("trace_invok_lat-" + index + ".json", 'w') as json_file:
        json.dump(trace_lat_dic, json_file)
        print("trace_invok_lat-" + index + ".json write done!")
    return trace_lat_dic


#func2
#trace_invok_lat.json的时间来计算
def get_start_end_intervels(filepath, chunk_lenth):
    trace_json = read_json(filepath)
    sendingTime = []
    for key in trace_json.keys():
        sendingTime.append(int(key))
    print(len(sendingTime))
    start = min(sendingTime)
    end = max(sendingTime)
    print("start:",start, ",end:",end)
    return start, end
    '''
    intervals = [(s, s+chunk_lenth-1) 
             for s in range(start, end-chunk_lenth+1)]
    #print(intervals)
    return start, end, intervals
    '''


def deal_traces(intervals, node_num, chunk_lenth):
    """
    Input:
        intervals=[(s,e)], the chunks covers the period of [s, e].
    Return:
        a dict containing info for each interval:
        -- cell of invok list : a dict contains invocations inside the given time period == as invocation-based edge-level features
            {s-t:[lat1, lat2, ...]}
        -- cell of latency list: a dict contains a np.array [chunk_lenth] denoting the average latency (per time slot) for each node 
                                === as trace-based node-level features
            {nid:np.array([lat_1, ..., lat_tau, ..., lat_chunk_lenth}])}
    """
    print("*** Dealing with traces...")
    traces = read_json("trace_invok_lat.json")
    invocations = [] # the number of invocations
    latency = np.zeros((len(intervals), node_num, chunk_lenth, 2))

    for chunk_idx, (s, e) in enumerate(intervals):
        invok = {}
        slots = [t for t in range(s, e+1)]
        for i, ts in enumerate(slots):
            if str(ts) in traces.keys(): # spans exist in the i-th time slot
                spans = traces[str(ts)]
                tmp_node_lat = [[] for _ in range(node_num)]
                for k, lat_lst in spans.items():
                    if k not in invok: 
                        invok[k] = 0
                    invok[k] += len(lat_lst)
                    t_node = int(k.split('-')[-1])                  
                    tmp_node_lat[t_node].extend(lat_lst)
                    #print(tmp_node_lat)
                for t_node in range(node_num):
                    if len(tmp_node_lat[t_node]) > 0:
                        latency[chunk_idx][t_node][i][0] = np.mean(tmp_node_lat[t_node])
        invocations.append(invok)
    
    z_zero_scaler = lambda x: (x-np.mean(x)) / (np.std(x)+1e-8)
    for i in range(node_num):
        latency[:, i, :, 0] = z_zero_scaler(latency[:, i, :, 0])
    
    chunk_traces = {"invok": invocations, "latency": latency}
    with open(("traces.pkl"), "wb") as fw:
        pickle.dump(chunk_traces, fw)

    #print(chunk_traces)
    return chunk_traces



def trace_vectorization(input_filepath, output_filepath):
    print("*** trace vectorization...")
    traces = read_json(input_filepath)
    zeors_array = np.zeros((41, 41))
    fo = open(output_filepath, "w")
    for timestamp in traces.keys():
        var_line = str(timestamp) + ":"
        A = zeors_array
        for invok in traces[timestamp].keys():
            callee = int(invok.split('-')[-1])
            caller = int(invok.split('-')[-2])
            latency = sum(traces[timestamp][invok])/len(traces[timestamp][invok])
            #print(latency)
            A[caller][callee] = latency
        #flatten:对数组进行降维，返回折叠后的一维数组，原数组不变
        A = A.flatten()
        #print(A)
        for i in range(A.size):
            if A[i] == 0:
                var_line = var_line + str(int(A[i])) + ','
            else:
                var_line = var_line + str(float(A[i])) + ','
        var_line = var_line.strip(',')       
        #print(var_line)
        fo.writelines(var_line + "\n")
    fo.close()
    




if "__main__" == __name__:
    index = "1"
    raw_filepath = "spans-" + index + ".json"
    trace_info_path = "trace_info-" + index + ".json"
    trace_invok_lat_path = "trace_invok_lat-" + index +".json"
    trace_vectorization_path = "trace_vectorization-" + index + ".txt"
    #生成trace_info.json
    #trace_info_dict = trace_info_extract_func(raw_filepath, index)
    #生成trace_invok_lat.json
    #trace_invok_lat_dic = get_trace_invok_lat_func(trace_info_dict, index)

    #start,end = get_start_end_intervels(trace_invok_lat_path, 10)
    #chunk_traces = deal_traces(intervals, 41, 10)

    #生成向量化trace文件
    trace_vectorization(trace_invok_lat_path, trace_vectorization_path)





"""
#func1
#get a trace-level file
#input: span-level file
#output:trace-level trace(pkl, list)
'''
trace_latency_list:
[{"traceID": uuid,
  "start": timestamp,
  "end": timestamp,
  "latency": [t1, t2, ...]
  },
  {...}...]
'''
def trace_2_trace_latency_list(filepath):
    trace_json = read_json(filepath)
    trace_lat_dic = {}
    trace_lat_list = []
    trace_lantency = []
    for key in trace_json.keys():
        trace_lat_dic["traceID"] = key
        trace_lat_dic["start"] = trace_json[key][0]["sendingTime"]
        trace_lat_dic["end"] = trace_json[key][0]["sendingTime"]
        for span in trace_json[key]:
            trace_lantency.append(span["duration"])
            if span["sendingTime"] < trace_lat_dic["start"]:
                min_sendingTime = span["sendingTime"]
            if span["sendingTime"] >= trace_lat_dic["end"]:
                max_sendingTime = span["sendingTime"]
        trace_lat_dic["start"] = min_sendingTime
        trace_lat_dic["end"] = max_sendingTime
        trace_lat_dic["lantency"] = trace_lantency
        trace_lat_list.append(trace_lat_dic)
    print(trace_lat_list)    
    with open("trace_lat_list.pkl", "wb") as fw:
        pickle.dump(trace_lat_list, fw)
    return trace_lat_list

#func3
#
#input:
#output:
def trace_process_func(trace_latency_list):
    invocations = [] # the number of invocations
    latency = np.zeros((len(intervals), chunk_lenth, 2))
    intervals = [(s, s+chunk_lenth-1) 
                 for s in range(start, end-chunk_lenth+1)]
    for trace in trace_latency_list:
"""