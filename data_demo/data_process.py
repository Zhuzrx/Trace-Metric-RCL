
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
def span_process_func(filepath):
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
    print(service_dict)


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
            span_data["sendingTime"] = round(span["startTime"]/1000000)
            span_data["duration"] = (span["duration"]/1000000)
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
    #with open('trace_processed_new.json', 'w') as json_file:
    #    json.dump(trace_dict, json_file)
    #    print("json write done!")
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

def get_trace_data(trace_dict):
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
    with open('trace_new.json', 'w') as json_file:
        json.dump(trace_lat_dic, json_file)
        print("json write done!")
    return trace_lat_dic


#func2
#trace_new.json的时间来计算
def get_start_end_intervels(filepath, chunk_lenth):
    trace_json = read_json(filepath)
    sendingTime = []
    for key in trace_json.keys():
        sendingTime.append(int(key))
    print(len(sendingTime))
    start = min(sendingTime)
    end = max(sendingTime)

    intervals = [(s, s+chunk_lenth-1) 
             for s in range(start, end-chunk_lenth+1)]
    #print(intervals)
    print("start:",start, ",end:",end)
    return start, end, intervals



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
    traces = read_json("trace_new.json")
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
    '''
    for i in range(node_num):
        latency[:, i, :, 0] = z_zero_scaler(latency[:, i, :, 0])
    '''
    chunk_traces = {"invok": invocations, "latency": latency}
    with open(("traces.pkl"), "wb") as fw:
        pickle.dump(chunk_traces, fw)

    print(invocations)
    return chunk_traces




if "__main__" == __name__:
    row_filepath = "./TT.2022-04-19T001753D2022-04-19T020534/spans.json"
    trace_processed_path = "trace_processed_new.json"
    trace_new_path = "trace_new.json"
    #trace_dict = span_process_func(row_filepath) #生成trace_processed_new.json
    #trace_lat_dic = get_trace_data(trace_dict) #生成trace_new.json
    start,end,intervals = get_start_end_intervels(trace_new_path, 10)
    #chunk_traces = deal_traces(intervals, 17, 10)
    #trace_latency_list = trace_2_trace_latency_list()
    #trace_process_func(trace_latency_list)



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