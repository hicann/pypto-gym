# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Select a focused guide and owned example without preloading a full catalog."""
import argparse
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]


def owned_source(root, source):
    path=Path(source)
    if path.is_absolute() or '..' in path.parts or len(path.parts)<2 or path.parts[0] not in ('agent','kernels','library'):
        raise ValueError(f'Invalid navigation owner: {source}')
    return root.parent/path


def example_scope(row):
    """What a folder offers a reader: the shape it is, and how to make it answer.

    Both owners are four files with a `main.py`, so there is one answer for both. Each keeps its own
    word for the searchable text -- a demo computes a `formula`, an API example exercises a
    `surface` -- and a row carries whichever its owner generated. Neither keeps a contract or a
    receipt beside it, which is why nothing here resolves one."""
    return {**{key:row[key] for key in ('formula','surface') if row.get(key)},
            'topology':row.get('topology',''),
            'tags':row.get('tags',[]),'cases':row.get('cases'),
            **({'pypto_pro_note':'main.py carries a comment naming what the PyPTO-Pro backend refuses here'}
               if row.get('pypto_pro_note') else {}),
            **({'refusals':row['refusals'],
                'refusal_note':'main.py names the reason beside the code it refuses and skips the case with it printed'}
               if row.get('refusals') else {}),
            'run':'python main.py --list, then --launcher sim|pipesim|aclnn|board|pypto in the folder',
            'support_scope':'A folder records no backend result. Where it has run is what you run, on the machine you run it from.'}


def term_matches(text, term):
    """Keep English word boundaries and explicit Chinese phrases; do not infer synonyms."""
    term=term.casefold()
    text=text.casefold()
    if re.search(r'[^\x00-\x7f]',term):
        return term in text
    expression=r'[\s_-]+'.join(re.escape(part) for part in re.split(r'[\s_-]+',term))
    return re.search(r'(?<![a-z0-9])'+expression+r'(?![a-z0-9])',text) is not None


def navigation(root):
    patterns=json.loads((root/'index/patterns.json').read_text())
    path=root/'index/kernels.json'
    generated=json.loads(path.read_text()) if path.is_file() else {}
    by_source={row['source']:row for row in generated.get('candidates',[])}
    rows=[]
    for pattern in patterns['patterns']:
        row={**by_source.get(pattern['source'],{}),**pattern}
        rows.append(row)
    hints={}
    for row in patterns['patterns']:
        hints.setdefault(row['source'],{key:row[key] for key in ('guide','api_guide','note','terms') if key in row})
    rows.extend({**row,**hints.get(row['source'],{})} for row in generated.get('candidates',[]))
    return rows,generated.get('deferred_candidates',[]),patterns.get('query_rules',[])


def match(row, query, requested):
    source=row['source']
    identities={row['id'].casefold(),source.casefold()}
    if requested:
        return (10000,['id_or_path']) if requested in identities else (0,[])
    if query.casefold() in identities:
        return 10000,['id_or_path']
    terms=list(dict.fromkeys(term.casefold() for term in row.get('terms',[])))
    hits=[term for term in terms if term_matches(query,term)]
    if hits:
        return 1000+max(len(term) for term in hits)+len(hits),hits
    if 'terms' in row:
        return 0,[]
    # Identifiers, descriptions, formulas and tags provide discovery for every generated entry.
    words=set(re.findall(r'[a-z0-9]+',query.casefold()))-{'a','an','the','for','with','of','to','and','in','on'}
    haystack=' '.join((row['id'],source,row.get('description',''),row.get('formula',''),
                       row.get('surface',''),row.get('topology',''),' '.join(row.get('tags',[]))))
    hits=sorted(word for word in words if term_matches(haystack,word))
    return len(hits),hits


def fallback(root, language):
    targets=[('library/docs/api/README.md','API topics and signatures','API 专题与签名'),
             ('library/docs/api/manifest.json','Find a declared name and its usage','按名称查声明与用例'),
             ('library/examples/api/README.md','Runnable primitive examples','可运行 primitive 样例'),
             ('library/examples/api/index.json','Every API example, its surface and its cases','全部 API 样例、surface 与 case'),
             ('library/docs/status.md','Current validation scope and evidence','当前验证范围与证据'),
             ('kernels/index.json','Every kernel demo, its formula and its topology','全部 kernel 样例、公式与拓扑')]
    result=[]
    for path,en,cn in targets:
        target=owned_source(root,path)
        if not target.is_file():
            raise ValueError(f'Stale navigation target: {target}')
        result.append({'path':str(target),'purpose':cn if language=='zh-CN' else en})
    return result


def landing(source):
    """A folder's entry is its main.py; a pattern may also point at a plain page or profile file."""
    return source/'main.py' if source.is_dir() else source


def select(query='', pattern=None, language='en', device='a5', limit=3, root=ROOT):
    if language not in ('en','zh-CN') or not 1 <= limit <= 5:
        raise ValueError('Use en/zh-CN and limit between one and five')
    index=root/'index/kernels.json'
    generated=json.loads(index.read_text()) if index.is_file() else {}
    hardware=generated.get('release',{}).get('declared_hardware')
    if hardware and device not in hardware:
        return {'status':'deferred','device':device,'guide':str(root/generated['deferred']['handoff']),
                'reason':'This target is outside the snapshot gallery device scope; no cross-family support inferred.','selected':[]}
    requested=pattern.casefold() if pattern else None
    rows,deferred_rows,rules=navigation(root)
    relation=next((rule for rule in rules if any(term_matches(query,term) for term in rule['terms'])),None)
    expanded=' '.join(relation.get('expand_terms',[])) if relation else ''
    ranked=[]
    for outside,entries in ((False,rows),(True,deferred_rows)):
        for position,row in enumerate(entries):
            score,hits=match(row,query,requested)
            if not score and expanded and not requested:
                score,hits=match(row,expanded,None)
            if score:
                ranked.append((-score,outside,position,row,hits))
    if any(item[0]==-10000 for item in ranked):
        ranked=[item for item in ranked if item[0]==-10000]
    selected=[]
    related=[]
    deferred=[]
    rejected=[]
    gallery=[]
    seen=set()
    for _,outside,_,row,hits in sorted(ranked,key=lambda item:item[:3]):
        if row['source'] in seen:
            continue
        seen.add(row['source'])
        guide=root/language/row['guide']
        source=owned_source(root,row['source'])
        for path in (guide,landing(source)):
            if not path.is_file():
                raise ValueError(f'Stale navigation target: {path}')
        base={'pattern':row['id'],'guide':str(guide),'source':str(source),'matched':hits}
        if row.get('case'):
            base['case']=row['case']  # `python main.py --case <id>` in that folder
        declared=row.get('devices', [row.get('device', 'a5')])
        if device not in declared:
            if len(rejected)<limit:
                rejected.append({**base,'reason':'device_not_declared','declares':declared})
            continue
        if row.get('api_guide'):
            base['api_guide']=str(owned_source(root,row['api_guide']))
        if row.get('note'):
            base['semantic_note']=row['note']
        for key in ('study_for','do_not_copy_when'):
            if row.get(key):
                base[key]=row[key]
        if outside:
            if len(deferred)<limit:
                deferred.append({**base,'reason':'outside_snapshot_scope'})
            continue
        if relation and device in relation.get('related_devices', ['a2', 'a3', 'a5']):
            if len(related)<limit:
                related.append({**base,'reason':relation['reason'],
                                'scope':'Related reading only; requested filters are not established.'})
            continue
        if len(selected)>=limit:
            # A curated pattern scores in the thousands and a word found in a folder's formula,
            # surface or tags scores in the ones, so a folder nobody curated can never reach
            # `selected` while any pattern matches. That is right for ranking and wrong for
            # discovery: the only vector-only online-softmax demo was invisible to
            # `--query 'online softmax'` even though its tags said so. Name those separately --
            # owner, id, path and what matched.
            if 'terms' not in row:
                gallery.append({'id':row['id'],'owner':row.get('owner',''),'source':str(source),
                                'matched':hits,'topology':row.get('topology',''),
                                'title':row.get('description','')})
            continue
        selected.append({**base,**example_scope(row)})
    # Equal word hits leave index order deciding, which returns the same shape three times. One
    # per topology first says more in the same space: the vector-only statement of a pattern is
    # the one a reader cannot find any other way.
    first,rest={},[]
    for row in gallery:
        rest.append(row) if row['topology'] in first else first.setdefault(row['topology'],row)
    gallery=(list(first.values())+rest)[:limit]
    reason=('matched' if selected else 'outside_snapshot_scope' if deferred else
            'related_only' if related else rejected[0]['reason'] if rejected else 'no_candidates')
    nxt='Read the selected guide and the source itself; an absent match is not a capability gap. A demo declares no backend result — run it on the launcher you need the answer from.'
    if not selected and rejected:
        # `device_not_declared` read as "this demo has no device" to a cold reader holding an A2 task
        # under the a5 default. Say which family it does declare and what to pass.
        other=rejected[0]
        nxt=(f"Nothing matched on {device}; {other['pattern']} matched and declares {', '.join(other['declares'])}. "
             f"If that is the family you are writing for, rerun with --device {other['declares'][0]}. ")+nxt
    return {'status':'selected' if selected else 'no_match','device':device,
            'baseline':[str(root/language/'common-language.md'),str(root/language/'references/authoring-preflight.md')],
            'selected':selected,'reason':reason,'related':related,'deferred_material':deferred,'rejected':rejected,
            'also_matched':gallery,
            'fallback':fallback(root,language) if not selected else [],
            'next':nxt}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--query',default='')
    p.add_argument('--pattern')
    p.add_argument('--language',choices=('en','zh-CN'),default='en')
    p.add_argument('--device',choices=('a5','a2','a3'),default='a5')
    p.add_argument('--limit',type=int,default=3)
    a=p.parse_args()
    if not a.query and not a.pattern:
        p.error('Provide --query or --pattern')
    try:
        result=select(a.query,a.pattern,a.language,a.device,a.limit)
    except (ValueError,OSError,KeyError,TypeError) as e:
        p.error(str(e))
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
