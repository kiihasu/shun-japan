#!/usr/bin/env python3
"""Independent relative-strength report; requires the existing shun.py.
ETF results are uncorrected for distributions and are ALWAYS reference only.
"""
import argparse
import re
from urllib.request import Request, urlopen
from html import unescape
from pathlib import Path
import numpy as np
import pandas as pd
from shun import JQuants, config, live_data, atomic_text

LOOKBACK = 252

def nikkei_members(asof):
    url = 'https://indexes.nikkei.co.jp/nkave/index/component?idx=nk225'
    with urlopen(Request(url, headers={'User-Agent': 'ShunJapanDaily/1.0'}), timeout=25) as response:
        html = response.read().decode('utf-8')
    plain = unescape(re.sub(r'<[^>]+>', ' ', html))
    match = re.search(r'更新日付[\s：:]*([0-9]{4}[.][0-9]{2}[.][0-9]{2})', plain)
    if not match or match[1].replace('.', '-') != asof:
        raise ValueError('対象日の日経225構成を確認できません')
    cells = re.findall(r'<t[dh]\b[^>]*>(.*?)</t[dh]>', html, re.I | re.S)
    codes = {re.sub(r'<[^>]+>', '', cell).strip() for cell in cells}
    members = {code + '0' for code in codes if re.fullmatch(r'[0-9][0-9A-Z][0-9][0-9A-Z]', code)}
    if len(members) != 225:
        raise ValueError('日経225の225銘柄を確認できません')
    return members


def adjusted_close(frame, sessions):
    g = frame.copy()
    g['Date'] = g.Date.astype(str)
    g = g[g.Date.isin(sessions)].sort_values('Date')
    if g.Date.duplicated().any():
        raise ValueError('価格の日付が重複しています')
    g = g.set_index('Date').reindex(sessions)
    c = pd.to_numeric(g.C, errors='coerce')
    f = pd.to_numeric(g.AdjFactor, errors='coerce')
    # Do not invent missing observations or adjustment factors.
    good = f.notna() & np.isfinite(f) & (f > 0)
    adj = f.where(good).iloc[::-1].cumprod(skipna=False).iloc[::-1] / f
    return (c * adj).where(np.isfinite(c) & (c > 0))

def signal(close, benchmark):
    ratio = close / benchmark
    valid = ratio.notna() & np.isfinite(ratio) & (ratio > 0)
    # A full calendar window is required. Missing days are not dropped.
    if len(ratio) < LOOKBACK + 1 or not valid.iloc[-253:].all():
        return None
    previous = ratio.shift(1).rolling(LOOKBACK, min_periods=LOOKBACK).max()
    flags = (ratio > previous) & valid
    hit = bool(flags.iloc[-1])
    if not hit:
        state = '更新なし'
    elif len(ratio) < 254 or not valid.iloc[-254:].all():
        state = '更新・前日判定不可'
    elif bool(flags.iloc[-2]):
        state = '連日更新'
    else:
        state = '新規・再更新'
    return dict(hit=hit, state=state, ratio=float(ratio.iloc[-1]),
                prior_high=float(previous.iloc[-1]),
                excess_pct=float((ratio.iloc[-1]/previous.iloc[-1]-1)*100),
                stock_change_pct=float((close.iloc[-1]/close.iloc[-2]-1)*100))

def make_report(master, raw, benchmarks, sessions, asof, out, members=None):
    sessions = sorted(set(d for d in sessions if d <= asof))
    if len(sessions) < 253 or sessions[-1] != asof:
        raise ValueError('対象日まで最低253営業日が必要です')
    # Retain full window including missing calendar days.
    sessions = sessions[-280:]
    raw = raw.copy(); raw['Code'] = raw.Code.astype(str)
    groups = {code: g for code, g in raw.groupby('Code')}
    prices = {code: adjusted_close(g, sessions) for code,g in groups.items()
              if code in set(master.Code.astype(str))}
    lines = [f'# 相対強度52週高値レポート {asof}', '',
             '上昇率ランキングとは独立。日経225採用は1321、グロースは2516、その他はTOPIXで個別に判定。',
             '本日の終値比率 > 本日を除く直前252営業日の最高比率（52週の近似）。同値は更新なし。',
             '株価は分割等を調整。配当は未調整。欠損日・履歴不足は判定不可。',
             'ETF代用は分配金未補正・参考判定です。指数そのものの高値更新とは異なります。',
             '分配落ちによる見かけの更新、費用、指数との乖離が含まれます。自動売買・通知機能はありません。',
             '新規・再更新は「前日は更新なし、当日は更新」を意味し、過去初の更新ではありません。', '']
    records=[]
    if members is None:
        lines += ['日経225構成を確認できず、プライム市場の比較先は判定保留。', '']
    code = master.Code.astype(str)
    market = master.Mkt.astype(str)
    nikkei = code.isin(members or set())
    unknown = (market == '0111') & (members is None)
    if unknown.any():
        lines += [f'比較先未確定 {int(unknown.sum())}銘柄（日経225構成未確認）。', '']
    for label, bench in benchmarks.items():
        selected = nikkei if '1321' in label else ((market == '0113') & ~nikkei if '2516' in label else ~(nikkei | (market == '0113') | unknown))
        subset = master.loc[selected]
        counts = f'判定可能 0 / 対象 {len(subset)}銘柄。判定不能 {len(subset)}銘柄。'
        proxy = 'ETF' in label
        lines += [f'## {label}', '']
        if isinstance(bench, str):
            lines += [f'{counts} 取得失敗・判定不可: {bench}', '']; continue
        if bench.index.duplicated().any():
            lines += [counts + ' 指数日付が重複しています。判定不可。', '']; continue
        b = pd.to_numeric(bench.reindex(sessions), errors='coerce')
        b = b.where(np.isfinite(b) & (b > 0))
        if b.iloc[-253:].isna().any():
            lines += [counts + ' 比較対象の253営業日分に欠損があります。', '']; continue
        eligible=0; hits=[]
        for row in subset.itertuples(index=False):
            code=str(row.Code); close=prices.get(code)
            result = None if close is None else signal(close,b)
            rec = dict(Date=asof,Code=code[:-1] if len(code)==5 and code.endswith('0') else code,
                       Name=row.CoName,Benchmark=label,ReferenceOnly=proxy,
                       Status='履歴不足・欠損' if result is None else result['state'])
            if result is not None:
                eligible+=1; rec.update(result)
                if result['hit']: hits.append(rec)
            records.append(rec)
        lines += [f'判定可能 {eligible} / 対象 {len(subset)}銘柄。判定不能 {len(subset)-eligible}銘柄。更新 {len(hits)}銘柄。', '']
        if proxy: lines += ['**以下はETF分配金未補正の参考候補。実指数での更新を保証しません。**', '']
        if not hits:
            lines += ['判定可能な銘柄に更新なし。' if eligible else '判定可能な銘柄なし。', '']; continue
        hits.sort(key=lambda r:(-r['excess_pct'],r['Code']))
        lines += ['| コード | 銘柄 | 更新状況 | 旧高値超過率 | 株価前日比 |',
                  '|---|---|---|---:|---:|']
        for r in hits:
            name=str(r['Name']).replace('|','／').replace('\n',' ')
            lines += [f"| {r['Code']} | {name} | {r['Status']} | {r['excess_pct']:.3f}% | {r['stock_change_pct']:+.2f}% |"]
        lines += ['', 'ティッカー（市場名なし）', '```text', ','.join(r['Code'] for r in hits)+',', '```', '']
    dest=Path(out)/asof
    atomic_text(dest/'relative_strength_report.md','\n'.join(lines))
    atomic_text(dest/'relative_strength_results.csv',pd.DataFrame(records).to_csv(index=False))
    return dest

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--date');p.add_argument('--cache',default='cache')
    p.add_argument('--refresh',action='store_true')
    p.add_argument('--config',type=Path,default=Path(__file__).parent/'config.json')
    p.add_argument('--out',default='reports')
    args=p.parse_args();c=config(args.config);c['history_sessions']=max(280,c['history_sessions'])
    master,raw,asof,sessions=live_data(c,args)
    api=JQuants(c['request_interval']);benchmarks={}
    try:
        t=api.get('indices/bars/daily/topix',**{'from':sessions[0],'to':asof})
        benchmarks['TOPIX・実指数']=t.set_index('Date')['C']
    except (ValueError,RuntimeError,KeyError,OSError) as e:
        benchmarks['TOPIX・実指数']=str(e)
    for code,label in [('13210','日経225・ETF1321代用'),('25160','グロース250・ETF2516代用')]:
        try:
            # Fetch explicitly; some existing daily caches may exclude ETFs.
            t=api.get('equities/bars/daily',code=code,**{'from':sessions[0],'to':asof})
            if t.empty or set(t.Code.astype(str)) != {code}:
                raise ValueError('ETF価格なし、またはコード不一致')
            benchmarks[label]=adjusted_close(t,sessions)
        except (ValueError,RuntimeError,KeyError,OSError) as e:
            benchmarks[label]=str(e)
    try:
        members = nikkei_members(asof)
    except (OSError, ValueError) as error:
        print(f'日経225構成取得不可: {error}')
        members = None
    dest=make_report(master,raw,benchmarks,sessions,asof,args.out,members)
    print(f'相対強度レポート: {dest}/relative_strength_report.md')

if __name__=='__main__':
    main()
