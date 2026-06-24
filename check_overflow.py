#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_overflow.py
从 PyPower / DCOPF 日志里定位“branch flow constraints”表，解析列并找出 |Pf| 或 |Pt| 超过 |Pmax| 的线路。
支持表头形如：
  #     Bus    Pf  mu     Pf      |Pmax|      Pt      Pt  mu   Bus
"""

import re
import argparse
import csv
from typing import List, Dict, Tuple, Optional

HEADER_REQUIRED_TOKENS = ["#", "Bus", "Pf", "|Pmax|", "Pt", "Bus"]

NUM_RE = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?$')

def smart_split_line(line: str) -> list:
    """常规按空白切，但不去掉中间的负号/科学计数法；返回 token 列表"""
    return line.strip().split()

def try_split_bus_pair(tok: str):
    """
    试图把像 '35250' 这样的粘连 bus 对拆成 (35, 250)。
    经验规则：
      1) 全为数字（允许前导0）
      2) 考虑所有切分点，优先右侧<=3位（常见 bus 号不超过3位）
      3) 两侧都能解析为 int
    """
    if not tok.isdigit() or len(tok) < 2:
        return None
    # 优先让右侧 1~3 位
    for r in range(1, min(3, len(tok)-1) + 1):
        a, b = tok[:-r], tok[-r:]
        try:
            return (int(a), int(b))
        except:
            continue
    # 兜底：遍历所有切分
    for i in range(1, len(tok)):
        a, b = tok[:i], tok[i:]
        try:
            return (int(a), int(b))
        except:
            continue
    return None

def parse_data_line_whitespace(names: list[str], line: str):
    """
    更鲁棒的数据行解析：
    1) 先按空白切
    2) 若列数比 header 少 1，且第二列疑似 Bus 粘连，则尝试拆分
    3) 列数仍不匹配则返回 None 交给其他策略（如切片法）或跳过
    """
    toks = smart_split_line(line)

    # 针对你这类表头: ['line_idx','Bus','Pf','mu','Pf','|Pmax|','Pt','Pt','mu','Bus']
    # 两个 Bus 紧邻并分别在索引 1 和最后一个位置
    want = len(names)

    if len(toks) == want - 1 and names[1] == 'Bus' and names[-1] == 'Bus':
        # line_idx 在 toks[0]，Bus 粘在 toks[1] 的常见情况
        pair = try_split_bus_pair(toks[1])
        if pair is not None:
            # 组装成期望列数
            toks = [toks[0], str(pair[0])] + toks[2:] + [str(pair[1])]

    if len(toks) != want:
        return None

    # 映射到 dict
    row = {}
    for k, v in zip(names, toks):
        vv = v.replace('D','E').replace('d','e')
        # 把 '-' 视为缺失
        if v == '-':
            row[k] = 0.0
            continue
        # 能转成数就转（float/int），否则保留原字符串
        try:
            if NUM_RE.match(vv):
                f = float(vv)
                # 对 Bus/line_idx 更适合 int
                if k.lower().startswith('bus') or 'line' in k.lower():
                    row[k] = int(round(f))
                else:
                    row[k] = f
            else:
                row[k] = v
        except:
            row[k] = v
    return row

def is_header_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    # 宽松匹配：同时包含 # 和 |Pmax|，并且包含 Pf 和 Pt
    return ("#" in s) and ("|Pmax|" in s) and ("Pf" in s) and ("Pt" in s)

def compute_columns_from_header(header: str) -> Tuple[List[str], List[Tuple[int,int]]]:
    """
    根据表头里每个 token 的起始位置确定列切片范围（左闭右开）。
    这样后续数据行用相同的切片就能稳定取到对应列。
    对重复列名自动加后缀：Pf -> Pf_1, Pf_2；Pt -> Pt_1, Pt_2。
    """
    tokens = []
    for m in re.finditer(r'\S+', header):
        tokens.append((m.group(), m.start()))
    # 排序（其实已按出现顺序）
    tokens.sort(key=lambda x: x[1])

    names = []
    starts = [pos for _, pos in tokens]
    ends = starts[1:] + [len(header)]

    # 去重命名
    seen = {}
    for name, _ in tokens:
        base = name
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            name = f"{base}_{seen[base]}"
        names.append(name)

    spans = list(zip(starts, ends))
    return names, spans

def parse_row_with_spans(line: str, names: List[str], spans: List[Tuple[int,int]]) -> Dict[str, str]:
    row = {}
    for name, (a, b) in zip(names, spans):
        row[name] = line[a:b].strip()
    return row

def to_int(x: str) -> Optional[int]:
    try:
        return int(x)
    except:
        return None

def to_float(x: str) -> Optional[float]:
    try:
        # 兼容科学计数法
        return float(x.replace('D','E').replace('d','e'))
    except:
        return None

def looks_like_data_line(line: str) -> bool:
    # 数据行通常以索引或空格开始，且含数字
    if not line.strip():
        return False
    return bool(re.search(r'[0-9]', line))

def find_section(lines: List[str]) -> Tuple[int, int, List[str], List[Tuple[int,int]]]:
    """
    找到目标表头及其列定义。
    返回：header行号、数据起始行号、列名、列切片
    """
    for i, line in enumerate(lines):
        if is_header_line(line):
            names, spans = compute_columns_from_header(line.rstrip('\n'))
            # 跳过下一行（通常是分隔线）
            data_start = i + 2 if i + 1 < len(lines) else i + 1
            return i, data_start, names, spans
    raise RuntimeError("未找到匹配的表头行（包含 # / |Pmax| / Pf / Pt）。")

def pick_pf_pt_pmax(row: Dict[str, str]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    取第一个 Pf、Pt 以及 |Pmax| 列（字符串 -> float）
    """
    # 可能存在 Pf, Pf_2；Pt, Pt_2
    pf_key = "Pf" if "Pf" in row else ("Pf_1" if "Pf_1" in row else None)
    pt_key = "Pt" if "Pt" in row else ("Pt_1" if "Pt_1" in row else None)

    # |Pmax| 有时包含竖线，列名就是 "|Pmax|"
    pmax_key = None
    for k in row.keys():
        if k.strip().upper().replace(" ", "") in {"|PMAX|", "│PMAX│", "|Pmax|"}:
            pmax_key = k
            break
    # 兜底：严格找名为 |Pmax|
    if pmax_key is None and "|Pmax|" in row:
        pmax_key = "|Pmax|"

    Pf = to_float(row[pf_key]) if pf_key else None
    Pt = to_float(row[pt_key]) if pt_key else None
    Pmax = to_float(row[pmax_key]) if pmax_key else None
    return Pf, Pt, Pmax

def parse_and_check(path: str) -> Tuple[List[Dict], List[Dict]]:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    header_idx, data_start, names, spans = find_section(lines)

    all_rows = []
    violations = []

    for j in range(data_start, len(lines)):
        line = lines[j].rstrip('\n')
        if not looks_like_data_line(line):
            # 到了空行或非数据行，结束
            break

        # row = parse_row_with_spans(line, names, spans)
        # row = parse_data_line_whitespace(names, line)
        cnt = 0
        row = {}
        splits = line.split(' ')
        for s in splits:
            if s != '':
                if s == '-':
                    row[names[cnt]] = 0.0
                else:
                    row[names[cnt]] = float(s)
                cnt += 1

        # 基本字段（尽量解析，失败就留 None）
        line_idx = to_int(row.get("#", ""))
        bus_from = to_int(row.get("Bus", ""))  # 第一列 Bus
        # 最后一列 Bus（可能叫 Bus_2 或 Bus_3，取名字里含 'Bus' 的最后一个）
        bus_to_key = None
        for k in row.keys():
            if k.startswith("Bus") and k != "Bus":
                bus_to_key = k
        # bus_to = to_int(row.get(bus_to_key, "")) if bus_to_key else None
        try:
            bus_to = int(row['Pt_2'])
        except:
            import ipdb
            ipdb.set_trace()

        # Pf, Pt, Pmax = pick_pf_pt_pmax(row)
        Pf = row['mu']
        Pt = row['|Pmax|']
        Pmax = row['Pf_2']

        record = dict(
            line_raw=line,
            line_idx=line_idx,
            bus_from=bus_from,
            bus_to=bus_to,
            Pf=Pf,
            Pt=Pt,
            Pmax=Pmax
        )
        all_rows.append(record)

        if Pmax is not None:
            over_pf = (Pf is not None) and (abs(Pf) >= abs(Pmax))
            over_pt = (Pt is not None) and (abs(Pt) >= abs(Pmax))
            if over_pf or over_pt:
                record['over_pf'] = bool(over_pf)
                record['over_pt'] = bool(over_pt)
                violations.append(record)

        # 若后面出现新表头/分隔标题，也可以提前结束
        if is_header_line(line):
            break

    return all_rows, violations

def save_csv(rows: List[Dict], out_path: str):
    if not rows:
        return
    # 统一字段顺序
    keys = ["#", "bus_from", "bus_to", "Pf", "Pt", "Pmax", "over_pf", "over_pt", "line_raw"]
    # 确保键存在
    normalized = []
    for r in rows:
        rr = {k: r.get(k) for k in keys}
        normalized.append(rr)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(normalized)

def main():
    ap = argparse.ArgumentParser(description="检查日志中的线路越限（|Pf| 或 |Pt| > |Pmax|）")
    ap.add_argument("logfile", help="日志文件路径")
    ap.add_argument("--out", default="", help="把越限结果导出为 CSV（可选）")
    args = ap.parse_args()

    all_rows, violations = parse_and_check(args.logfile)

    print(f"解析到 {len(all_rows)} 行数据；发现越限 {len(violations)} 条。")
    if violations:
        print("\n越限 Top-10（按出现顺序）：")
        for v in violations[:10]:
            print(f"- #={v.get('line_idx')}, "
                  f"{v.get('bus_from')}->{v.get('bus_to')}, "
                  f"Pf={v.get('Pf')}, Pt={v.get('Pt')}, |Pmax|={v.get('Pmax')}, "
                  f"over_pf={v.get('over_pf', False)}, over_pt={v.get('over_pt', False)}")
        print(f"所有越限线路: {[v.get('line_idx') for v in violations]}")
        print(f"建议容量: {[v.get('Pmax') * 1.5 for v in violations]}")

    if args.out:
        save_csv(violations, args.out)
        print(f"\n已导出越限明细到: {args.out}")

if __name__ == "__main__":
    main()
