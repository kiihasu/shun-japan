#!/usr/bin/env python3
"""旬理論・日次スクリーナー。発注機能なし。Python 3.11+。"""
import argparse
import json
import math
import os
import sys
import time
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DEFAULT = dict(min_value=200_000_000, min_history=210, min_group=3,
               min_coverage=0.70, group_threshold=60, stock_threshold=65,
               top_n=10, per_group=3, stop_pct=0.05, capital=1_000_000,
               risk_pct=0.005, max_position_pct=0.10, max_basket_pct=0.30,
               lot=100, request_interval=1.1, history_sessions=280)


def atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(content, encoding='utf-8')
    tmp.replace(path)


def config(path):
    c = DEFAULT.copy()
    if path.exists():
        supplied = json.loads(path.read_text(encoding='utf-8'))
        if set(supplied) - set(c):
            raise ValueError('未知の設定キー: ' + str(set(supplied) - set(c)))
        c.update(supplied)
    if not (0 < c['stop_pct'] < 1 and 0 < c['risk_pct'] < 1 and
            0 < c['max_position_pct'] <= c['max_basket_pct'] <= 1 and
            0 < c['min_coverage'] <= 1):
        raise ValueError('割合設定が範囲外です')
    for k in ['capital', 'lot', 'top_n', 'per_group', 'min_group', 'min_value']:
        if c[k] <= 0:
            raise ValueError(f'{k} は正数が必要です')
    if c['min_history'] < 210 or c['history_sessions'] < c['min_history'] + 5:
        raise ValueError('min_history>=210、history_sessions>=min_history+5 が必要です')
    return c


class JQuants:
    def __init__(self, interval):
        self.key = os.environ.get('JQUANTS_API_KEY')
        if not self.key:
            raise ValueError('環境変数 JQUANTS_API_KEY を設定してください。デモは --demo。')
        self.interval = interval

    def get(self, endpoint, **params):
        rows, seen = [], set()
        while True:
            url = 'https://api.jquants.com/v2/' + endpoint + '?' + urllib.parse.urlencode(params)
            for attempt in range(6):
                time.sleep(self.interval)
                try:
                    req = urllib.request.Request(url, headers={'x-api-key': self.key})
                    with urllib.request.urlopen(req, timeout=60) as response:
                        payload = json.load(response)
                    break
                except urllib.error.HTTPError as e:
                    if e.code not in (429, 500, 502, 503, 504) or attempt == 5:
                        raise RuntimeError(f'API {endpoint}: HTTP {e.code}。契約・利用範囲を確認してください。') from None
                    time.sleep(min(2 ** attempt * 2, 45))
                except (urllib.error.URLError, TimeoutError):
                    if attempt == 5:
                        raise RuntimeError(f'API {endpoint}: 通信失敗') from None
                    time.sleep(min(2 ** attempt, 30))
            if not isinstance(payload.get('data'), list):
                raise ValueError('APIレスポンスに data 配列がありません')
            rows.extend(payload['data'])
            token = payload.get('pagination_key')
            if not token:
                return pd.DataFrame(rows)
            if token in seen:
                raise ValueError('ページングの循環を検出しました')
            seen.add(token)
            params['pagination_key'] = token


def target_date(api, now, explicit=None):
    # 朝は前営業日。20時以前は当日を使わず、未確定データを避ける。
    upper = pd.Timestamp(explicit).date() if explicit else now.date() - timedelta(days=int(now.hour < 20))
    if upper > now.date() or (upper == now.date() and now.hour < 20):
        raise ValueError('当日20時前または未来日付の終値は使用できません')
    cal = api.get('markets/calendar', **{'from': str(upper - timedelta(days=550)), 'to': str(upper)})
    if cal.empty:
        raise ValueError('取引カレンダーを取得できません')
    dates = sorted(cal.loc[cal.HolDiv.astype(str).isin(['1', '2']), 'Date'].unique())
    if not dates:
        raise ValueError('対象の営業日がありません')
    return dates[-1], dates


def live_data(c, args):
    api = JQuants(c['request_interval'])
    asof, sessions = target_date(api, datetime.now(ZoneInfo('Asia/Tokyo')), args.date)
    sessions = sessions[-c['history_sessions']:]
    master = api.get('equities/master', date=asof)
    required = {'Code', 'CoName', 'S33Nm', 'Mkt'}
    if not required.issubset(master):
        raise ValueError('銘柄マスターの必要項目が不足しています')
    master['Code'] = master.Code.astype(str)
    master = master[master.Mkt.astype(str).isin(['0111','0112','0113']) & master.Code.str.endswith('0')].copy()
    if 'ProdCat' in master:
        master = master[master.ProdCat.astype(str) == '011']
    master = master[~master.Code.isin(['83010','84210'])]
    if master.empty or master.Code.duplicated().any():
        raise ValueError('銘柄マスターが空、または重複しています')
    cache = Path(args.cache)
    pieces = []
    for i, d in enumerate(sessions):
        p = cache / (d + '.json')
        # 最新5営業日は訂正を再取得。全期間の訂正は --refresh。
        if p.exists() and not args.refresh and d not in sessions[-5:]:
            frame = pd.DataFrame(json.loads(p.read_text()))
        else:
            frame = api.get('equities/bars/daily', date=d)
            if frame.empty:
                raise ValueError(f'{d} の株価が未提供です。古い日付に置換せず停止しました')
            atomic_text(p, frame.to_json(orient='records', force_ascii=False))
        if set(frame.Date.astype(str)) != {d}:
            raise ValueError(f'{d}: 日付の不一致')
        pieces.append(frame)
        if (i+1) % 40 == 0:
            print(f'取得 {i+1}/{len(sessions)} 営業日', flush=True)
    raw = pd.concat(pieces, ignore_index=True)
    return master, raw, asof, sessions


def validate(raw):
    req = ['Date','Code','O','H','L','C','Va','AdjFactor']
    if not set(req).issubset(raw):
        raise ValueError('株価CSVの必要列: ' + ','.join(req))
    raw = raw.copy()
    raw['Date'] = pd.to_datetime(raw.Date).dt.strftime('%Y-%m-%d')
    raw['Code'] = raw.Code.astype(str)
    if raw.duplicated(['Code','Date']).any():
        raise ValueError('株価に日付・コードの重複があります')
    for col in req[2:]:
        raw[col] = pd.to_numeric(raw[col], errors='raise')
    if not np.isfinite(raw.AdjFactor).all() or (raw.AdjFactor <= 0).any():
        raise ValueError('調整係数が欠落または不正です')
    # nullは取引なしとして保持し、前日値で埋めない。
    price = raw[['O','H','L','C','Va']]
    if (price.notna().any(axis=1) & price.isna().any(axis=1)).any():
        raise ValueError('OHLC・売買代金の一部だけが欠損しています')
    good = raw.dropna(subset=['C'])
    if not np.isfinite(good[['O','H','L','C','Va','AdjFactor']].to_numpy()).all():
        raise ValueError('無限値が含まれています')
    if ((good[['O','H','L','C']] <= 0).any(axis=1) | (good.Va < 0) |
        (good.H < good[['O','L','C']].max(axis=1)) |
        (good.L > good[['O','H','C']].min(axis=1))).any():
        raise ValueError('株価・売買代金が不正です')
    return raw.sort_values(['Code','Date'])


def features(raw, master, asof, sessions, c):
    raw = validate(raw)
    raw = raw[raw.Date <= asof]  # 将来データを先に除外。調整係数も同様。
    sessions = [d for d in sessions if d <= asof]
    if len(sessions) < c['min_history']:
        raise ValueError('営業日数が不足しています')
    records = []
    universe = set(master.Code)
    for code, g in raw.groupby('Code', sort=False):
        if code not in universe:
            continue
        g = g.set_index('Date').reindex(sessions)
        # 未取得日がある銘柄は除外。分割係数の欠落を1と推測しない。
        if g.tail(c['min_history']).AdjFactor.isna().any():
            continue
        factors = g.AdjFactor.fillna(1)
        cum = factors.iloc[::-1].cumprod().iloc[::-1] / factors
        close, high, low = g.C * cum, g.H * cum, g.L * cum
        if close.tail(c['min_history']).isna().any() or g.Va.iloc[-1] <= 0:
            continue
        ma = {n: close.rolling(n).mean() for n in [5,20,50,100,200]}
        vbase = g.Va.shift(1).rolling(20).mean().iloc[-1]
        if vbase <= 0:
            continue
        tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        closepos = (close.iloc[-1]-low.iloc[-1]) / max(high.iloc[-1]-low.iloc[-1], 1e-8)
        signed = (np.sign(close.diff()) * g.Va).tail(5).sum() / max(g.Va.tail(5).sum(),1)
        rec = dict(Code=code, close=close.iloc[-1], r1=close.iloc[-1]/close.iloc[-2]-1,
                   r5=close.iloc[-1]/close.iloc[-6]-1, r20=close.iloc[-1]/close.iloc[-21]-1,
                   value=g.Va.iloc[-1], liquidity=vbase, value_ratio=g.Va.iloc[-1]/vbase,
                   flow5=g.Va.tail(5).mean()/g.Va.iloc[-25:-5].mean(), signed5=signed,
                   breakout=bool(close.iloc[-1] > high.shift(1).rolling(20).max().iloc[-1]),
                   high20=high.shift(1).rolling(20).max().iloc[-1], atr_pct=atr/close.iloc[-1],
                   closepos=closepos, bias20=close.iloc[-1]/ma[20].iloc[-1]-1,
                   slope20=ma[20].iloc[-1]/ma[20].iloc[-6]-1,
                   prevclose=close.iloc[-2], upper_limit=str(g.get('UL',pd.Series('0',index=g.index)).iloc[-1]) == '1')
        rec.update({f'ma{n}': s.iloc[-1] for n,s in ma.items()})
        records.append(rec)
    if not records:
        raise ValueError('有効な銘柄がありません。履歴・欠損・対象日を確認してください')
    f = pd.DataFrame(records).merge(master[['Code','CoName','S33Nm']], on='Code', validate='one_to_one')
    coverage = len(f) / len(master)
    if coverage < c['min_coverage']:
        raise ValueError(f'全体データ充足率 {coverage:.1%}。基準未達のため順位を出しません')
    # 基準は有効ユニバースの中央値。TOPIXとは異なる。
    f['rs5'] = f.r5 - f.r5.median()
    f['rs20'] = f.r20 - f.r20.median()
    f['above20'] = f.close > f.ma20
    f['above200'] = f.close > f.ma200
    for col in ['rs5','rs20','flow5','signed5']:
        f['p_'+col] = f[col].rank(pct=True)
    f['score'] = (25*f.p_rs5 + 20*f.p_rs20 + 20*f.p_flow5 + 10*f.p_signed5 +
                  10*f.above20 + 5*f.above200 + 10*f.breakout)
    f['eligible'] = (f.liquidity >= c['min_value']) & (f.value >= c['min_value'])
    return f, coverage


def group_table(f, master, themes, c):
    rows = []
    groups = [('業種',name,set(g.Code)) for name,g in master.groupby('S33Nm')]
    if not themes.empty:
        groups += [('テーマ',name,set(g.Code)) for name,g in themes.groupby('Theme')]
    for kind, name, codes in groups:
        # 現在上場マスターにある構成銘柄のみ。テーマは任意の固定分類表。
        codes &= set(master.Code)
        z = f[f.Code.isin(codes)]
        if not codes:
            continue
        coverage = len(z)/len(codes)
        if len(z) < c['min_group'] or coverage < c['min_coverage']:
            rows.append(dict(kind=kind, group=name, n=len(z), coverage=coverage, score=np.nan, stage='判定不足'))
            continue
        score = (0.45*z.score.median() + 25*z.above20.mean() +
                 15*z.rs5.gt(0).mean() + 15*np.clip((z.flow5.median()-1)/0.5,0,1))
        rows.append(dict(kind=kind, group=name, n=len(z), coverage=coverage, score=score,
                         rs5=z.rs5.median(), rs20=z.rs20.median(), r5=z.r5.median(),
                         breadth=z.above20.mean(), flow5=z.flow5.median(),
                         hot=(z.bias20 > .20).mean(), stage='監視'))
    return pd.DataFrame(rows)


def stages(groups, previous, c):
    p = previous.set_index(['kind','group']) if not previous.empty else pd.DataFrame()
    groups = groups.copy()
    for i,r in groups.iterrows():
        if pd.isna(r.score):
            continue
        key=(r.kind,r.group)
        old=p.loc[key,'score'] if not p.empty and key in p.index else np.nan
        delta=r.score-old
        groups.loc[i,'delta5']=delta
        if r.hot >= .4:
            stage='過熱'
        elif (r.rs5 < 0 and r.flow5 < 1) or (pd.notna(delta) and delta <= -12):
            stage='失速'
        elif r.score >= c['group_threshold'] and r.rs5 > 0 and r.flow5 >= 1.1 and r.breadth >= .55:
            stage='初動' if pd.notna(old) and old < c['group_threshold'] else '継続'
        else:
            stage='監視'
        groups.loc[i,'stage']=stage
    return groups.sort_values('score', ascending=False, na_position='last')


def select(f, groups, themes, c):
    f=f.copy()
    active=groups[groups.stage.isin(['初動','継続'])].sort_values('score',ascending=False)
    memberships={code:[] for code in f.Code}
    for _,r in active.iterrows():
        codes=set(f.loc[f.S33Nm.eq(r.group),'Code']) if r.kind=='業種' else set(themes.loc[themes.Theme.eq(r.group),'Code'])
        for code in codes & set(memberships):
            memberships[code].append(f'{r.kind}:{r.group}')
    states=[]
    for _,r in f.iterrows():
        if not r.eligible:
            s='流動性不足'
        elif r.bias20 > .20 or r.r5 > .25 or r.upper_limit:
            s='過熱・追随回避'
        elif r.close < r.ma20 and (r.rs5 < 0 or r.flow5 < 1):
            s='失速・撤退点検'
        elif not memberships[r.Code] or r.score < c['stock_threshold'] or r.close <= r.ma200:
            s='監視'
        elif r.breakout and r.value_ratio >= 1.5 and r.closepos >= .65:
            s='高値更新候補'
        elif -.02 <= r.bias20 <= .05 and r.slope20 > 0 and r.r1 > 0 and r.flow5 >= 1.1:
            s='押し目反発候補'
        elif r.close > r.ma20 and r.rs5 > 0 and r.flow5 >= 1.1:
            s='上昇継続監視'
        else:
            s='監視'
        states.append(s)
    f['signal']=states
    f['groups']=f.Code.map(lambda code:' / '.join(memberships[code]))
    f['reason']=f.apply(lambda r:f'5日相対強度 {r.rs5:+.1%} / 5日売買代金倍率 {r.flow5:.2f} / 20日線乖離 {r.bias20:+.1%}',axis=1)
    f['reference_stop']=f.close*(1-c['stop_pct'])
    f['atr_warning']=f.atr_pct*2 > c['stop_pct']
    f['lot_estimate']=[math.floor(min(c['capital']*c['risk_pct']/(p*c['stop_pct']),
                                      c['capital']*c['max_position_pct']/p)/c['lot'])*c['lot'] for p in f.close]
    chosen=[]; counts={}; used=0
    candidates=f[f.signal.isin(['高値更新候補','押し目反発候補'])].sort_values('score',ascending=False)
    for i,r in candidates.iterrows():
        # 重複テーマ・同じ業種への集中を抑える。サイズは終値仮定の参考値。
        exposure=set(memberships[r.Code]) | {'業種:'+r.S33Nm}
        if any(counts.get(g,0)>=c['per_group'] for g in exposure):
            continue
        shares=min(r.lot_estimate, math.floor(max(c['capital']*c['max_basket_pct']-used,0)/r.close/c['lot'])*c['lot'])
        if shares<=0:
            continue
        f.loc[i,'lot_estimate']=shares
        used+=shares*r.close
        chosen.append(i)
        for g in exposure:
            counts[g]=counts.get(g,0)+1
        if len(chosen)>=c['top_n']:
            break
    return f.sort_values('score',ascending=False), f.loc[chosen].copy()


def md_table(df, columns):
    if df.empty:
        return '該当なし。条件を緩めて無理に選定しません。\n'
    names=list(columns)
    def cell(v):
        if isinstance(v,(float,np.floating)):
            return '—' if not np.isfinite(v) else f'{v:.3f}'
        return str(v).replace('|','／').replace('\n',' ')
    return '| '+' | '.join(columns.values())+' |\n|'+ '|'.join(['---']*len(names))+'|\n'+'\n'.join('| '+' | '.join(cell(r.get(k,'')) for k in names)+' |' for _,r in df.iterrows())+'\n'


def report(master, raw, asof, sessions, themes, c, out, demo=False):
    f, coverage=features(raw,master,asof,sessions,c)
    old_date=sessions[-6]
    prev,_=features(raw,master,old_date,sessions,c)
    groups=stages(group_table(f,master,themes,c),group_table(prev,master,themes,c),c)
    stocks,picks=select(f,groups,themes,c)
    breadth=f.above20.mean()
    regime='新規エントリー抑制' if breadth < .40 else '通常監視'
    if breadth < .40:
        picks=picks.iloc[:0]
    out=Path(out)/asof
    out.mkdir(parents=True,exist_ok=True)
    for name,frame in [('groups',groups),('stocks',stocks),('picks',picks)]:
        atomic_text(out/(name+'.csv'),frame.to_csv(index=False))
    title='【架空データ・動作例】' if demo else ''
    body=f'''# {title}旬レポート {asof}

対象：東証プライム・スタンダード・グロースの普通株式。分析基準は当該日の終値。

市場環境：**{regime}**。20日線上比率 {breadth:.1%}。データ充足率 {coverage:.1%}（{len(f)}/{len(master)}銘柄）。
相対強度は分析可能銘柄のリターン中央値との差。TOPIX比較ではありません。
売買代金・価格による資金集中の代理指標であり、純流入額や心理を直接観測するものではありません。

## 注目業種・テーマ

'''+md_table(groups.head(20),{'kind':'区分','group':'分類','n':'銘柄数','score':'旬スコア','delta5':'5営業日前比','breadth':'20日線上比率','flow5':'代金倍率','stage':'局面'})+'''
## エントリー検討候補

'''+md_table(picks,{'Code':'コード','CoName':'銘柄','signal':'判定','score':'点数','close':'終値','reference_stop':'参考損切り','lot_estimate':'参考株数','atr_warning':'値動き注意','reason':'根拠'})+'''
## 監視上位（購入候補とは別）

'''+md_table(stocks.head(20),{'Code':'コード','CoName':'銘柄','signal':'判定','score':'点数','groups':'旬の分類','reason':'根拠'})+f'''
## 規律

- 参考損切りは終値の-{c['stop_pct']:.0%}。実際は約定価格を基準に設定。窓開け・値幅制限ではその価格で売れず損失が超過します。
- 参考株数は仮の運用資金 {c['capital']:,.0f}円、1銘柄許容損失 {c['risk_pct']:.1%}、1銘柄配分上限 {c['max_position_pct']:.0%}、候補バスケット上限 {c['max_basket_pct']:.0%}、{c['lot']}株単位で算出。既存保有は加味していません。
- ATR注意=True は直近14日平均値幅の2倍が損切り幅を超える状態。参考株数は発注数量ではありません。
- 過熱は追随回避・保有時の一部利確検討、失速は撤退点検。利益率だけを理由に全売却しません。
- 決算・増資・信用規制・材料の事実確認は別途必要。ニュース/SNS解析と保有建玉監視は未実装です。
- 閾値・点数は仮説であり、統計的優位性・収益性は未検証。バックテスト結果ではありません。
'''
    atomic_text(out/'report.md',body)
    metadata=dict(asof=asof, demo=demo, coverage=coverage, config=c, stocks=len(f), picks=len(picks),
                  created_at=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat())
    atomic_text(out/'metadata.json',json.dumps(metadata,ensure_ascii=False,indent=2))
    print(f'出力: {out}/report.md / 候補 {len(picks)}銘柄')
    return groups,stocks,picks


def demo_data():
    rng=np.random.default_rng(42)
    dates=pd.bdate_range('2025-01-06',periods=280).strftime('%Y-%m-%d').tolist()
    masters=[]; rows=[]
    for k in range(60):
        code=f'D{k:03d}0'; sec=['デモ電力設備','デモ半導体','デモ内需'][k//20]
        masters.append(dict(Code=code,CoName=f'架空企業{k+1:02}',S33Nm=sec))
        price=700+20*k
        for i,d in enumerate(dates):
            ret=rng.normal(.0007,.009)
            if i>=255 and k<20:
                ret+=.004
            if i>=270 and k>=40:
                ret-=.008
            if i==279 and k<5:
                ret=.025
            opening=price; price*=math.exp(ret)
            val=400_000_000*rng.uniform(.8,1.2)*(2.0 if i>=275 and k<20 else 1)
            rows.append(dict(Date=d,Code=code,O=opening,H=max(price,opening)*1.003,
                             L=min(price,opening)*.997,C=price,Va=val,AdjFactor=1.0))
    return pd.DataFrame(masters),pd.DataFrame(rows),dates[-1],dates


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--demo',action='store_true',help='ネット不要の架空データ')
    p.add_argument('--date',help='分析上限日 YYYY-MM-DD。未指定は日本時間で最新の確定営業日')
    p.add_argument('--csv-dir',type=Path,help='master.csv / prices.csv / calendar.csv を読み込む')
    p.add_argument('--config',type=Path,default=BASE/'config.json')
    p.add_argument('--themes',type=Path,default=BASE/'themes.csv')
    p.add_argument('--cache',default=str(BASE/'cache'))
    p.add_argument('--out',default=None)
    p.add_argument('--refresh',action='store_true',help='全日付のキャッシュを更新')
    args=p.parse_args()
    c=config(args.config)
    if args.demo:
        master,raw,asof,sessions=demo_data()
        themes=pd.DataFrame(columns=['Code','Theme','Subtheme'])
    else:
        if args.csv_dir:
            master=pd.read_csv(args.csv_dir/'master.csv',dtype={'Code':str})
            raw=pd.read_csv(args.csv_dir/'prices.csv',dtype={'Code':str})
            cal=pd.read_csv(args.csv_dir/'calendar.csv',dtype=str)
            sessions=sorted(cal.loc[cal.HolDiv.isin(['1','2']),'Date'].unique())
            upper=args.date or raw.Date.max()
            sessions=[d for d in sessions if d<=upper]
            if not sessions:
                raise ValueError('対象営業日がありません')
            asof=sessions[-1]
            if asof != raw.loc[raw.Date<=asof,'Date'].max():
                raise ValueError('対象営業日の株価がありません')
            if master.Code.duplicated().any():
                raise ValueError('マスターの銘柄が重複しています')
        else:
            master,raw,asof,sessions=live_data(c,args)
        themes=pd.read_csv(args.themes,dtype=str).fillna('') if args.themes.exists() else pd.DataFrame(columns=['Code','Theme','Subtheme'])
        if not {'Code','Theme'}.issubset(themes):
            raise ValueError('themes.csv は Code,Theme 列が必要です')
        if not themes.empty:
            themes['Code']=themes.Code.str.strip().str.upper().map(lambda s:s+'0' if len(s)==4 else s)
            unknown=set(themes.Code)-set(master.Code)
            if unknown:
                print('注意: マスターにないテーマ構成銘柄を除外: '+','.join(sorted(unknown)))
            themes=themes.drop_duplicates(['Code','Theme'])
    report(master,raw,asof,sessions,themes,c,args.out or BASE/('demo_reports' if args.demo else 'reports'),args.demo)


if __name__=='__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError, KeyError) as e:
        print('停止: '+str(e),file=sys.stderr)
        sys.exit(1)
