#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Index reverse-import evidence against the generated forward support baseline.

Observed means a checked fixture emits an opcode, not that every operand form is
admitted. No evidence is a work item, not a guessed declaration of no converter.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from pypto_support import dispatched  # noqa: E402

from ascriptor.importers.pypto_pro import ProImportError, import_module, loads  # noqa: E402
from ascriptor.importers.pypto_pro.cache import TARGETS as CACHE  # noqa: E402
from ascriptor.importers.pypto_pro.cross import FLAG_TARGETS as FLAGS  # noqa: E402
from ascriptor.importers.pypto_pro.cross import TARGETS as CORE  # noqa: E402
from ascriptor.importers.pypto_pro.ctrl import TARGETS as CTRL  # noqa: E402
from ascriptor.importers.pypto_pro.debug import TARGETS as DEBUG  # noqa: E402
from ascriptor.importers.pypto_pro.mx import TARGETS as MX  # noqa: E402
from ascriptor.importers.pypto_pro.nz_parameters import TARGETS as NZ  # noqa: E402
from ascriptor.importers.pypto_pro.scalar_ops import TARGETS as SCALAR  # noqa: E402
from ascriptor.importers.pypto_pro.simt import TARGETS as SIMT  # noqa: E402
from ascriptor.importers.pypto_pro.simt_exact import TARGETS as SIMT_EXACT  # noqa: E402
from ascriptor.importers.pypto_pro.simt_math import TARGETS as SIMT_MATH  # noqa: E402
from ascriptor.importers.pypto_pro.slots import TARGETS as SLOTS  # noqa: E402
from ascriptor.importers.pypto_pro.sort import TARGETS as SORT  # noqa: E402
from ascriptor.importers.pypto_pro.vector import ARITHMETIC  # noqa: E402
from ascriptor.importers.pypto_pro.vector_fused import TARGETS as FUSED  # noqa: E402
from ascriptor.importers.pypto_pro.vector_fused_cast import TARGETS as FUSED_CAST  # noqa: E402
from ascriptor.importers.pypto_pro.vector_index import TARGETS as INDEX  # noqa: E402
from ascriptor.importers.pypto_pro.vector_indexed import TARGETS as INDEXED  # noqa: E402
from ascriptor.importers.pypto_pro.vector_integer import TARGETS as INTEGER  # noqa: E402
from ascriptor.importers.pypto_pro.vector_masks import TARGETS as MASKS  # noqa: E402
from ascriptor.importers.pypto_pro.vector_memory import TARGETS as MEMORY  # noqa: E402
from ascriptor.importers.pypto_pro.vector_predicates import TARGETS  # noqa: E402
from ascriptor.importers.pypto_pro.vector_rearrange import TARGETS as REARRANGE  # noqa: E402
from ascriptor.importers.pypto_pro.vector_spr import TARGETS as SPR  # noqa: E402
from ascriptor.importers.pypto_pro.views import TARGETS as VIEWS  # noqa: E402

DOC = ROOT / 'docs/pypto-pro-import-support.md'
TESTS = (('arithmetic', 'vf_math'), ('predicate/cast', 'mask_cast'), ('integer', 'integer'), ('FP16/BF16', 'float16'),
         ('register rearrangement', 'rearrange'), ('activation/reduction/index', 'misc'),
         ('register arithmetic/accumulator', 'arith'), ('scalar bit/shift/extremum', 'scalar'),
         ('Mat fill/literal spelling', 'fill'), ('fused cast/predicate spill', 'fused_mask'),
         ('launch identity/CTRL readback', 'identity'), ('cross-core collective', 'collective'), ('debug call', 'debug'),
         ('SIMT launch', 'simt'), ('SIMT math', 'simt_float'), ('SIMT atomic/exact math', 'simt_exact'),
         ('VF block copy/unaligned access', 'vf_memory'), ('VF block strides/cursors', 'vf_blocks'),
         ('indexed gather/scatter/compaction/pack/histogram', 'vf_gather'),
         ('SPR mask/datablock reduction/interleave', 'vf_spr'), ('GM view/tile alias', 'dma_views'), ('sort/merge record', 'dma_mx'),
         ('merge spelling', 'mrgsort'),
         ('DMA layout follow-up', 'dma_followup'), ('slot buffer', 'dma_slots'), ('microscaling product', 'mx'),
         ('NZ-packed GM load', 'dma_nz'), ('measured FP32 extremum/register gather wrap', 'semantics'),
         ('data cache clean', 'dcci'), ('measured DMA relaxation', 'dma_probes'), ('INT8 NZ block store', 'i039_nz_int8'))
# Audited targets without a Pro source form. Evidence of a producer must replace the entry.
NO_PRODUCER = {
    'scalar.ceil_div': 'Pro scalar expressions have no ceiling division',
    'list.count': 'Pro kernel parameters have no tensor-list kind',
    'list.item': 'Pro kernel parameters have no tensor-list kind',
    'list.item_dim': 'Pro kernel parameters have no tensor-list kind',
    'dma.l0c_to_gm.nz2dn': 'Pro stores require ascending order; PTO Acc stores have no DN arm',
    'dma.gm_to_ub.nd': 'Pro GM loads are 2-D TLOAD descriptors without NDDMA strides or pad values (A5-UP-028..030)',
    'dma.ub_to_ub': 'Pro Vec→Vec moves print TMovVecToVec, a VF register loop (pto a5 TMov.hpp:522-547), not a UB DMA',
    'atomic.begin': 'Pro atomic accumulation is a per-store none/add attribute, not a region',
    'atomic.end': 'Pro atomic accumulation is a per-store none/add attribute, not a region',
    'atomic.set_type': 'Pro atomic accumulation is a per-store none/add attribute, not a region',
    'debug.assert': 'Pro pto_assert prints and continues; the target assertion stops the model',
    'debug.print_reg': 'Pro has no register print; printf takes scalars',
    'sync.event': 'Pro has raw flags and cross-core calls, no event object',
    'sync.set': 'Pro has raw flags and cross-core calls, no event object',
    'sync.wait': 'Pro has raw flags and cross-core calls, no event object',
    'sync.set_all': 'Pro has raw flags and cross-core calls, no event object',
    'sync.release': 'Pro has raw flags and cross-core calls, no event object',
    'sync.mutex': 'Pro writes credit seeds/drains as explicit cross-core calls, not a declaration',
    'vec.set_mask_by_count': 'Pro has no count-to-mask SPR call',
    'vf.gathermask': 'Pro has no VF gathermask; its squeeze is a different source form',
    'simt.log2': 'Pro prints log2 as its A5 formula over log, which import expands into scalar operations',
    'vec.set_mask_count': 'Pro set_mask_count selects a counter mode no c310 vector instruction consults (D-217)',
}


def report():
    emitted = defaultdict(set)
    origins = defaultdict(set)
    accepted, refused = [], []
    cast_forms = defaultdict(set)
    for path in sorted((ROOT / 'tests/importers/fixtures').rglob('*.json')):
        source = loads(path.read_text())
        try:
            module = import_module(source)
        except ProImportError as error:
            refused.append({'fixture': str(path.relative_to(ROOT)), 'reason': str(error)})
            continue
        fixture = str(path.relative_to(ROOT / 'tests/importers/fixtures'))
        accepted.append(fixture)
        nodes = {n['id']: n for g in source.document['targets'] for n in g['nodes']}
        ledger = module.attrs['import_ledger']
        for op in module.walk():
            emitted[op.opcode].add(fixture)
            if op.opcode == 'vf.cast':
                dst, src = op.operands[:2]
                key = (src.type.dtype.name, dst.type.dtype.name, op.attrs['mask'].type.width,
                       op.attrs['layout'].name, op.attrs['round'].name, op.attrs['saturate'])
                cast_forms[key].add(fixture)
            candidates = [r for r in ledger if op.id in r['target_ids']]
            if candidates:
                # Parent statement records also contain descendants. Retain the
                # smallest directly accountable source record, never every parent.
                row = min(candidates, key=lambda r: (len(r['target_ids']), r['kind'] != 'Call'))
                node = nodes[row['source']]
                origins[op.opcode].add(node['fields'].get('name', node['kind']))
    forward, _ = dispatched()
    rules = defaultdict(list)
    for source, target in ARITHMETIC.items():
        rules[target].append(source)
    for source, targets in (TARGETS | INTEGER | REARRANGE | INDEX | INDEXED | FUSED | FUSED_CAST | MASKS | MEMORY | SPR | SCALAR | CORE
                            | CTRL | FLAGS | DEBUG | SIMT | SIMT_MATH | SIMT_EXACT | VIEWS | SORT | SLOTS | MX | NZ | CACHE).items():
        for target in targets:
            rules[target].append(source)
    rows = [{'target': target, 'rule': sorted(rules[target]), 'origins': sorted(origins[target]),
             'fixtures': sorted(emitted[target]), 'no_producer': NO_PRODUCER.get(target)} for target in sorted(forward)]
    stale = [r['target'] for r in rows if r['no_producer'] and (r['fixtures'] or r['rule'])]
    if stale or set(NO_PRODUCER) - forward:
        raise SystemExit(f'Audited no-producer entries contradict evidence or the baseline: {stale}')
    return {'forward_count': len(forward), 'rows': rows, 'accepted_fixtures': accepted,
            'refused_fixtures': refused,
            'cast_forms': [dict(zip(('source', 'target', 'mask_bits', 'layout', 'round', 'saturate'), key, strict=True),
                                fixtures=sorted(fixtures)) for key, fixtures in sorted(cast_forms.items())],
            'import_only_targets': sorted(set(emitted) - forward)}


def render(data):
    observed = sum(bool(r['fixtures']) for r in data['rows'])
    gaps = sum(bool(r['no_producer']) for r in data['rows'])
    tests = ', '.join(f'[{name}](../tests/importers/test_pypto_pro_{stem}.py)' for name, stem in TESTS)
    lines = ['# PyPTO Pro reverse-import evidence', '',
        'Generated by `tools/pypto_import_support.py`; do not edit by hand.', '',
        'Baseline: [forward backend support](pypto-pro-support.md). Its direction is IR → Pro;',
        'this index records Pro → IR. A printed opcode is not a reverse-conversion guarantee.', '',
        f'Forward baseline: **{data["forward_count"]}** opcodes; **{observed}** have checked import fixture evidence;',
        f'**{gaps}** are audited without a Pro source form; **{data["forward_count"] - observed - gaps}** remain unaudited.',
        f'Accepted fixtures: **{len(data["accepted_fixtures"])}**; refused fixtures: **{len(data["refused_fixtures"])}**.', '',
        '“Rule” identifies the dispatch tables used by the importer itself. “Observed” means a fixture',
        'produces that target opcode, possibly as part of an expansion or ABI declaration. “Unobserved”',
        'is a work item; it does not prove that no handler exists. The source column records the smallest',
        'ledger owner, not a one-to-one instruction promise. Export this index with `--json` for exact fixture paths.', '',
        'Rule scope: FP32 arithmetic, b16/b32 storage predicates, admitted numeric casts and INT32 bitwise/shifts; ZEROING.',
        'Register copies/bitcasts/permutations and cast/comparison overloads have explicit admission checks.',
        'Other operand/type/layout limits remain in the [import contract](rfc/0015-pypto-pro-import.md) and converters.',
        'Fixture import proves verifier/provenance coverage only. Models and native/CCE hardware results are',
        f'separate, revision-scoped gates; see the {tests} tests.', '',
        '## Forward baseline and reverse evidence', '',
        '| IR opcode | Reverse evidence | Source rule / observed owner | Fixtures |', '|---|---|---|---|']
    for row in data['rows']:
        state = ('rule + observed' if row['fixtures'] else 'rule') if row['rule'] else ('observed' if row['fixtures'] else 'unobserved')
        names = row['rule'] or row['origins']
        source = ', '.join(f'`{n}`' for n in names) or '—'
        if row['no_producer']:
            state, source = 'no Pro producer', row['no_producer']
        lines.append(f'| `{row["target"]}` | {state} | {source} | {len(row["fixtures"])} |')
    lines += ['', '## Observed cast forms', '',
        f'**{len(data["cast_forms"])}** checked type/mask/layout/round/saturation forms. These do not add opcode credit.', '',
        '| Source → target | Mask | Layout | Round | Saturate | Fixtures |', '|---|---|---|---|---|---|']
    for form in data['cast_forms']:
        lines.append(f'| {form["source"]} → {form["target"]} | b{form["mask_bits"]} | {form["layout"]} | '
                     f'{form["round"]} | {"ON" if form["saturate"] else "OFF"} | {len(form["fixtures"])} |')
    lines += ['', '## Semantic gates', '',
        '- Arange admits increasing FP32/INT32 ramps only; descending order is a separate semantic gate.',
        '- FP32 scalar min/max in vector and cube sections are IEEE maximum/minimum (NaN 0x7FFFFFFF, -0 < +0), as A5',
        '  measured; in VF bodies they are invalid A5 IR (its compiler rejects max()) and SIMT is unmeasured, so both',
        '  refuse. Integer scalar shifts and negation keep C++17 domains; the model rejects runtime violations.',
        '- FP32 literals must keep their bits through Pro\'s six-digit `std::to_string` spelling.',
        '- Mat expands fills the whole FP16 tile with an FP16-exact literal; valid extents must equal the shape.',
        '- muls_cast rounds the exact product once to FP16, ties away from zero; exp_sub is approximate',
        '  and tests FP16 predicates at source lanes. Only the default mem_bar mode is in the pinned profile.',
        '- AIV block counts are AIC counts in mixed launches; get_subblock_num is 1 except on mixed AIV.',
        '  CTRL reads must be whole assignments; integers derived from get_ctrl_spr (uint64 in Pro CCE)',
        '  must stay proven nonnegative, and reassigned variables lose that proof. Runtime CTRL writes keep',
        '  the low bit.',
        '- Cross-core mode 0 covers every core of its side; mode 1 only the AIVs of one core in mixed launches.',
        '- Debug calls are observation-only; GM dumps keep their PIPE_ALL barrier, and failed assertions continue.',
        '- dcci admits static INT32/FP32/FP16/BF16 GM parameters, proven offsets and every destination. A5 keeps',
        '  cross-core scalar stores into one 64-byte line only with a CACHELINE_OUT dcci after the store and a',
        '  publication that cannot run before it (I012). AIV mode 0/1 sets on PIPE_S refuse.',
        '- SIMT launches are one-dimensional and top-level on every AIV; syncthreads is top-level in its body.',
        '  UINT32 context queries keep C conversions and need no-wrap proofs; element indices must be proven.',
        '- GM views are root-sourced `mem.view`s declared at make_tensor with static strides proven inside the root;',
        '  a view source gives its origin, not its strides; offsets read reassigned scalars at the pointer (both A5).',
        '- UB loads ending inside a 32-byte block fill its rest with the burst\'s first element (A5, I038).',
        '- Tile groups with an advancing cursor or over 16 runtime-selected slots are one contiguous Vec/Mat slot buffer;',
        '  selections are proven in range and are only load, store, move and VF operands.',
        '- FP16/INT8 NZ-packed GM tensors are their ND storage; transfers copy whole column blocks inside it: Mat loads',
        '  narrowed on one axis, NZ Vec alias bursts and scaled INT8 Acc blocks, several only at [0, 0] from padded rows',
        '  (A5). FP32 and other uses refuse.',
        '- Tile aliases retype or reshape an allocation\'s first declaration and share its bytes; an identical Vec',
        '  declaration without valid metadata is that tile, and loop re-declarations bind once (A5). Identical Mat, L0',
        '  and non-transposing Mat aliases refuse. getval/setval index views flat; SIMT reads a view at its declared pitch.',
        '- ZN Mats are typed as their transpose; Vec-to-Mat moves copy bytes into NZ blocks or a flat row; NZ Vec',
        '  aliases are compact-fractal insert sources; one-row Mats are flat rows (all measured on A5).',
        '- sort32 writes descending FP32 (score, UINT32 identifier) records per 32-value group; row r restarts at',
        '  identifier row r or one shared row, over valid rows or valid groups. UINT32 storage is admitted only as',
        '  identifiers or SIMT launch storage. mrgsort block_len counts storage elements of the valid width; mrgsort2',
        '  binds (dst, src0, tmp, src1) with sources of their own lengths, tmp unspecified.',
        '- An FP8 move and its E8M0 plane move are one dma.l1_to_l0.mx when adjacent or in Pro\'s A5-measured order',
        '  (Left data, Right data, Left plane, Right plane); other orders refuse. Group exponents add before scaling,',
        '  subnormal partial sums flush to signed zero, and a ZZ scale load of fewer rows zeroes the rest of its box.',
        '- SIMT FP32 math is approximate (1e-6 + 1e-5·|f|); log/log2 expand into Pro\'s subnormal scaling and',
        '  division by log(2). Rounding makes zero results +0, as the A5 builtins do (measured).',
        '- SIMT atomics keep Pro\'s prior bits on INT32/UINT32/FP32 storage; FP32 max/min skip a NaN operand and',
        '  cas compares bits, as measured on A5. fma rounds once, fmod is exact and classification values are 0 or 1.',
        '- FP32 axpy/mul_add_dst/mul_dst_add round once, as measured on A5; separate multiply/add',
        '  rounding is a different result. In-place destinations need an earlier write in the VF body.',
        '- Register move is currently unmasked; masked Pro move merges destination lanes.',
        '- VF block copies take a literal block stride in [0, 32767] and follow the A5 rules of RFC-0001; stride-1',
        '  loads need whole-block create_mask predicates, and post-updating copies fold to static offsets outside VF',
        '  control flow. Unaligned loads/stores follow Pro cursors per printed address; chained stores need a',
        '  post_update attribute on store_unalign_post that the pinned profile does not declare.',
        '- Bitcast assignments are value snapshots: reinterpret plus copy, not a persistent alias.',
        '- UINT8/16/32 registers are VF-local INT32 views. Register gathers read FP32/INT32 data with INT32/UINT32 indices',
        '  and FP16/UINT16 modulo the source lanes (A5). Block gathers select whole blocks by index-lane predicate bits;',
        '  active scatter duplicates need identical payloads; inactive gather lanes, squeeze tails and unpacked halves are zero.',
        '- FP16/BF16 widening samples predicates at source positions: b32 + ONE yields zero;',
        '  use a b16 source predicate to activate odd positions. Narrowing currently admits b32 only.',
        '- SPR writes are top-level after set_mask_norm; a read needs a live write followed only by A5-measured',
        '  pset/plt/pand/movp/vcmp_lt, GM stores and descriptor updates. Counter mode is refused.',
        '- `vf.reduce_max/min` keep the native index lane (`index = true`); datablock groups write lane g.',
        '  Empty selections give dtype extremes. INTLV stores write every pair whatever their predicate.',
        '- Merging predicates, other dtype/cast layouts and VF control flow require their own mappings and',
        '  evidence. They are not inferred from matching names.',
        '- `cf.return` evidence includes generated epilogues; it does not admit source early returns.',
        '- Unobserved rows guide the next audit; all unknown source forms still fail with a location.', '']
    if data['import_only_targets']:
        lines += ['Import emits targets outside this forward baseline: ' + ', '.join(f'`{x}`' for x in data['import_only_targets']) + '.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    data = report()
    if args.json:
        print(json.dumps(data, indent=2))
        return
    text = render(data)
    if args.check:
        if not DOC.exists() or DOC.read_text() != text:
            raise SystemExit('Reverse import support index is stale; run tools/pypto_import_support.py')
    else:
        DOC.write_text(text)


if __name__ == '__main__':
    main()
