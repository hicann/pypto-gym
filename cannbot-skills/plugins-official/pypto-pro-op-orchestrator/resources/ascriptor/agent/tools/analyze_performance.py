# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Audit explicitly scoped cost/interval inputs; never infer HBM traffic from GM."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


def number(value,name,positive=False):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError(f'{name} requires a finite number')
    if value < 0 or (positive and value <= 0):
        raise ValueError(f'{name} is outside its domain')
    return value


def union(intervals):
    out=[]
    for begin,end in sorted(intervals):
        number(begin,'interval start')
        number(end,'interval end')
        if end < begin:
            raise ValueError('Reversed interval')
        if end == begin:
            continue
        if out and begin <= out[-1][1]:
            out[-1][1]=max(end,out[-1][1])
        else:
            out.append([begin,end])
    return out


def intersect(a,b):
    out=[]
    i=j=0
    a=union(a)
    b=union(b)
    while i<len(a) and j<len(b):
        start=max(a[i][0],b[j][0])
        end=min(a[i][1],b[j][1])
        if end>start:
            out.append([start,end])
        if a[i][1]<=b[j][1]:
            i+=1
        else:
            j+=1
    return out


def overlap(records):
    groups=defaultdict(lambda:{'cube':[],'vector':[]})
    for row in records:
        if row.get('kind') in ('dma','sync'):
            continue
        if row.get('kind') not in ('cube','vector'):
            raise ValueError('Unclassified compute interval')
        if any(k not in row for k in ('core','stage','item','start','end')):
            raise ValueError('Compute interval needs core, stage, item and endpoints')
        union([(row['start'],row['end'])])
        groups[str(row['core'])][row['kind']].append(row)
    result=[]
    for core,parts in sorted(groups.items()):
        cube=[(r['start'],r['end']) for r in parts['cube']]
        vector=[(r['start'],r['end']) for r in parts['vector']]
        all_overlap=intersect(cube,vector)
        cross=[]
        pairs=defaultdict(list)
        for c in parts['cube']:
            for v in parts['vector']:
                if c['item']==v['item']:
                    continue
                ranges=intersect([(c['start'],c['end'])],[(v['start'],v['end'])])
                if ranges:
                    cross.extend(ranges)
                    pairs[(str(c['stage']),str(v['stage']),c['item'],v['item'])].extend(ranges)
        result.append({'core':core,'compute_overlap':sum(b-a for a,b in all_overlap),
                       'cross_item_overlap':sum(b-a for a,b in union(cross)),
                       'pairs':[{'cube_stage':k[0],'vector_stage':k[1],'cube_item':k[2],'vector_item':k[3],
                                 'overlap':sum(b-a for a,b in union(v))} for k,v in sorted(pairs.items())]})
    return result


def hardware_samples(rows,expected_compute=True):
    timing=[]
    counters=[]
    for row in rows:
        try:
            time=number(row.get('duration_us'),'duration_us',positive=True)
        except ValueError:
            continue
        timing.append(time)
        try:
            cycles=number(row.get('total_cycles'),'total_cycles',positive=expected_compute)
        except ValueError:
            continue
        ratios=row.get('pipe_ratios',{})
        if not isinstance(ratios,dict):
            continue
        try:
            for value in ratios.values():
                if number(value,'pipe ratio')>1:
                    raise ValueError('Pipe ratio exceeds one')
        except ValueError:
            continue
        counters.append({'duration_us':time,'total_cycles':cycles,'pipe_ratios':ratios})
    return {'timing_status':'VALID' if timing else 'UNKNOWN','timing_samples':len(timing),
            'counter_status':'VALID' if counters else 'UNKNOWN','counter_samples':len(counters),
            'minimum_us':min(timing) if timing else None,
            'note':'A valid latency can coexist with unknown counters. Individual zero pipe ratios can be valid.'}


def analyze(data):
    if data.get('schema')!='ascriptor.performance-input/1':
        raise ValueError('Unsupported analysis input schema')
    if data.get('time_domain') not in ('model_cycles','hardware_us'):
        raise ValueError('Declare one time domain')
    out={'schema':'ascriptor.performance-analysis/1','time_domain':data['time_domain'],
         'identity':data.get('identity',{}),'hbm_efficiency':'UNKNOWN'}
    if 'resource_work' in data:
        if data['time_domain']!='model_cycles':
            raise ValueError('Resource work requires model-cycle scope')
        total=number(data.get('elapsed'),'elapsed',positive=True)
        costs=defaultdict(float)
        for row in data['resource_work']:
            if 'resource' not in row:
                raise ValueError('Resource identity is required')
            costs[str(row['resource'])]+=number(row['cost'],'resource cost')
        bound=max(costs.values(),default=0)
        if bound>total:
            raise ValueError('Resource work exceeds elapsed time; check grouping and units')
        out.update(resource_bound=bound,resource_efficiency=bound/total,
                   rescheduling_only_speedup_bound=total/bound if bound else None)
    if 'compute_intervals' in data:
        out['overlap_by_core']=overlap(data['compute_intervals'])
    if 'samples' in data:
        if data['time_domain']!='hardware_us':
            raise ValueError('Hardware samples require hardware-us scope')
        out['hardware_samples']=hardware_samples(data['samples'],data.get('expected_compute',True))
    if 'hbm_bytes' in data:
        if data['time_domain']!='hardware_us':
            raise ValueError('HBM metrics require hardware-us scope')
        if data.get('hbm_traffic_source')!='measured':
            raise ValueError('HBM efficiency needs measured HBM traffic')
        count=number(data['hbm_bytes'],'hbm_bytes')
        duration=number(data.get('hardware_duration_us'),'hardware_duration_us',True)
        bw=number(data.get('sustained_hbm_bytes_per_second'),'sustained_hbm_bytes_per_second',True)
        if not data.get('hbm_bandwidth_source'):
            raise ValueError('Sustained bandwidth source is required')
        ratio=count/(duration*1e-6*bw)
        classification=data.get('hbm_bandwidth_classification','unknown')
        if classification=='measured_sustained':
            out['hbm_efficiency']=ratio
        else:
            out['assumed_bandwidth_scenario']={'ratio':ratio,'bandwidth_classification':classification,
                                              'scope':'Scenario using the supplied bandwidth; not measured HBM efficiency.'}
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    try:
        r=analyze(json.loads(a.input.read_text()))
    except (ValueError,KeyError,TypeError) as e:
        p.error(str(e))
    content=json.dumps(r,indent=2)+'\n'
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(content)
    print(content,end='')

if __name__=='__main__':
    main()
