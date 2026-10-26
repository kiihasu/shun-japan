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
               top_n=10, per_group=3, stop_pct=0.05, capital=10_000_000,
               risk_pct=0.005, max_position_pct=0.10, max_basket_pct=0.30,
               lot=100, request_interval=1.1, history_sessions=280,
               ranking_top_n=500, ranking_history_days=26, ranking_min_value=0,
               rotation_min_change=3, rotation_rate_change=0.05)


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
    for k in ['ranking_top_n', 'ranking_history_days', 'rotation_min_change']:
        if not isinstance(c[k], int) or c[k] <= 0:
            raise ValueError(f'{k} は正の整数が必要です')
    if c['ranking_history_days'] < 6 or c['ranking_min_value'] < 0 or not 0 < c['rotation_rate_change'] <= 1:
        raise ValueError('ランキング設定が範囲外です')
    if c['history_sessions'] < c['ranking_history_days'] + 20:
        raise ValueError('ランキング比較に必要な取得日数が不足しています')
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


# 公式の製品・事業紹介を参考にした初期の監視バスケット。売上構成比による純粋事業分類ではない。
BUSINESS_SEED = '''Code,Parent,Business,SourceURL,Note
80350,半導体,前工程製造装置,https://www.tel.com/product/,複数の前工程装置
77350,半導体,前工程製造装置,https://www.screen.co.jp/spe/products,洗浄装置を含む
65250,半導体,前工程製造装置,https://www.kokusai-electric.com/company/strength,成膜・膜質改善
80350,半導体,成膜装置,https://www.tel.com/product/,他事業も含む
65250,半導体,成膜装置,https://www.kokusai-electric.com/company/strength,初期2銘柄で業種転換判定は不足
68570,半導体,検査・計測,https://www.advantest.com/en/products/semiconductor-test-system/,電気的テスト
77290,半導体,検査・計測,https://www.accretech.com/jp/company/,半導体製造装置・精密測定
69200,半導体,検査・計測,https://www.lasertec.co.jp/products/semiconductor/,マスク・ウェーハの検査計測
58010,電力インフラ,電力ケーブル,https://www.furukawaelectric.com/product/energy/power_cable/,他事業も含む
58020,電力インフラ,電力ケーブル,https://sumitomoelectric.com/jp/cables,他事業も含む
58050,電力インフラ,電力ケーブル,https://www.swcc.co.jp/jpn/products/energy.html,他事業も含む
65040,電力インフラ,変電・配電機器,https://www.fujielectric.co.jp/products/power_distribution_systems/,他事業も含む
65080,電力インフラ,変電・配電機器,https://www.meidensha.co.jp/products/energy/prod_01/prod_01_05/,変圧器
66220,電力インフラ,変電・配電機器,https://www.daihen.co.jp/products/electric/trans/,変圧器
19420,電力インフラ,電気設備工事,https://www.kandenko.co.jp/business/,他事業も含む
19440,電力インフラ,電気設備工事,https://www.kinden.co.jp/business/denki/,他事業も含む
19590,電力インフラ,電気設備工事,https://www.kraftia.co.jp/service/electricity/,他事業も含む
70110,防衛,航空機・エンジン,https://www.mhi.com/jp/company/business-area,複合企業で他事業も含む
70120,防衛,航空機・エンジン,https://www.khi.co.jp/corporate/division/,防衛航空機
70130,防衛,航空機・エンジン,https://www.ihi.co.jp/company/business/aeroengine_space_defense/,防衛航空エンジン
'''


def load_businesses(path=None):
    import io
    if path is not None and Path(path).exists():
        frame=pd.read_csv(path,dtype=str).fillna('')
    else:
        frame=pd.read_csv(io.StringIO(BUSINESS_SEED),dtype=str).fillna('')
    if not {'Code','Parent','Business'}.issubset(frame):
        raise ValueError('businesses.csvにはCode,Parent,Business列が必要です')
    frame=frame.copy()
    for col in ['Code','Parent','Business']:
        frame[col]=frame[col].str.strip()
    if frame[['Code','Parent','Business']].eq('').any().any():
        raise ValueError('事業分類のコード・親分類・事業名は空にできません')
    frame['Code']=frame.Code.str.upper().map(lambda s:s+'0' if len(s)==4 else s)
    frame['Label']=frame.Parent+' / '+frame.Business
    return frame.drop_duplicates(['Code','Label'])


def classification_groups(master, themes=None, businesses=None):
    universe=set(master.Code)
    groups=[('業種',name,set(g.Code)) for name,g in master.groupby('S33Nm')]
    if themes is not None and not themes.empty:
        groups += [('テーマ',name,set(g.Code)&universe) for name,g in themes.groupby('Theme')]
    if businesses is not None and not businesses.empty:
        groups += [('事業',name,set(g.Code)&universe) for name,g in businesses.groupby('Label')]
    return groups


def group_table(f, master, themes, c, businesses=None):
    rows = []
    groups = classification_groups(master,themes,businesses)
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


def select(f, groups, themes, c, businesses=None):
    f=f.copy()
    active=groups[groups.stage.isin(['初動','継続'])].sort_values('score',ascending=False)
    memberships={code:[] for code in f.Code}
    for _,r in active.iterrows():
        if r.kind=='業種':
            codes=set(f.loc[f.S33Nm.eq(r.group),'Code'])
        elif r.kind=='事業':
            codes=set(businesses.loc[businesses.Label.eq(r.group),'Code'])
        else:
            codes=set(themes.loc[themes.Theme.eq(r.group),'Code'])
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


def load_positions(path, asof, sessions):
    """全保有を1銘柄1行で申告。空ファイルは保有なしとして扱う。"""
    cols=['Code','EntryDate','AvgCost','Shares','Stage','StopPrice']
    if path is None:
        return pd.DataFrame(columns=cols)
    p=pd.read_csv(path,dtype={'Code':str,'EntryDate':str,'Stage':str})
    if not set(cols).issubset(p):
        raise ValueError('positions.csv 必須列: '+','.join(cols))
    p=p[cols].copy()
    p['Code']=p.Code.str.strip().str.upper().map(lambda s:s+'0' if len(s)==4 else s)
    for col in ['AvgCost','Shares','StopPrice']:
        p[col]=pd.to_numeric(p[col],errors='raise')
        if not np.isfinite(p[col]).all() or (p[col]<=0).any():
            raise ValueError('保有数量・価格は有限の正数が必要です')
    if p.Code.duplicated().any() or (p.Shares%1!=0).any():
        raise ValueError('保有は1銘柄1行・整数株数で登録してください')
    if not p.Stage.isin(['trial','confirmed']).all():
        raise ValueError('Stageはtrialまたはconfirmedです')
    if not p.EntryDate.isin(sessions).all() or (p.EntryDate>asof).any():
        raise ValueError('取得日は取得済み価格履歴内の営業日・分析日以前が必要です')
    return p


def early_plan(stocks, master, history, rotation, positions, sessions, asof, c, themes=None, businesses=None):
    """未来を参照せず予兆を判定。約定・保有状態の自動変更はしない。"""
    s=stocks.copy(); days=sorted(history.Date.unique())
    leaders=set(history.loc[history.Date.eq(asof)&history.period.isin([1,5]),'Code'])
    group_rows=[]; memberships={code:[] for code in s.Code}; confirmed=set()
    for kind,name,registered in classification_groups(master,themes,businesses):
        g=rotation[(rotation.kind==kind)&(rotation.sector==name)&(rotation.period==1)]
        if g.empty or g.iloc[0].signal=='判定不足':
            continue
        r=g.iloc[0]; key=f'{kind}:{name}'
        h=history[history.period.eq(1)]
        new=set(h.loc[h.Date.isin(days[-3:]) & h.entry.eq('新規'),'Code']) & registered
        today=set(h.loc[h.Date.eq(asof),'Code']) & registered
        liquid=set(s.loc[s.eligible,'Code'])
        recent=new & today & liquid
        old=set(h.loc[h.Date.eq(days[-4]),'Code']) & registered
        # 3日前と比べた絶対数・率、全市場に対する集中を確認する。
        delta=len(today)-len(old)
        early=(len(recent)>=2 and delta>=2 and delta/r.universe>=.02 and r.concentration>=1.2)
        confirmed_rows=rotation[(rotation.kind==kind)&(rotation.sector==name)&rotation.period.isin([1,5])]
        ok=confirmed_rows.signal.eq('転換確認候補').any()
        if ok: confirmed.add(key)
        group_rows.append(dict(kind=kind,group=name,new_liquid=len(recent),delta3=delta,
                               rate_delta3=delta/r.universe,concentration=r.concentration,
                               signal='広がり確認' if ok else ('打診の予兆' if early else '監視')))
        if early or ok:
            for code in registered & set(memberships): memberships[code].append(key)
    evidence=pd.DataFrame(group_rows,columns=['kind','group','new_liquid','delta3','rate_delta3','concentration','signal'])
    held=positions.set_index('Code'); indexed=s.set_index('Code'); reviews=[]
    unknown=set(held.index)-set(indexed.index)
    total=sum(float(indexed.loc[code,'close'])*row.Shares for code,row in held.iterrows() if code in indexed.index)
    trial_total=sum(float(indexed.loc[code,'close'])*row.Shares for code,row in held.iterrows() if code in indexed.index and row.Stage=='trial')
    for code,p in held.iterrows():
        if code not in indexed.index:
            reviews.append(dict(Code=code,action='データ不足・手動確認',reason='評価できない保有があるため全買付候補を停止',close=np.nan,stop=p.StopPrice))
            continue
        r=indexed.loc[code]; stop=max(p.StopPrice,p.AvgCost*(1-c['stop_pct']))
        age=sum(p.EntryDate<d<=asof for d in sessions)
        strong=bool(set(memberships[code]) & confirmed)
        if r.close<=stop: action='損切り条件到達'; reason='終値が取得価格基準または登録済み撤退価格以下'
        elif p.Stage=='trial' and age>=5 and not strong: action='期限撤退点検'; reason='取得後5営業日以上、広がり未確認'
        elif r.signal=='失速・撤退点検': action='失速・撤退点検'; reason='20日線割れと相対強度または代金倍率の悪化'
        elif r.signal=='過熱・追随回避': action='一部利確点検'; reason='過熱条件到達。利益率だけで全売却しない'
        else: action='継続監視'; reason='撤退条件未到達'
        reviews.append(dict(Code=code,action=action,reason=reason,close=r.close,stop=stop,return_pct=(r.close/p.AvgCost-1)*100,held_sessions=age))
    review=pd.DataFrame(reviews,columns=['Code','action','reason','close','stop','return_pct','held_sessions'])
    blocked=set(review.loc[review.action.ne('継続監視'),'Code'])
    counts={}; held_codes=set(held.index)
    def exposures(code):
        result={'業種:'+indexed.loc[code,'S33Nm']}
        for kind,name,members in classification_groups(master,themes,businesses):
            if code in members: result.add(f'{kind}:{name}')
        return result
    for code in held_codes & set(indexed.index):
        for key in exposures(code): counts[key]=counts.get(key,0)+1
    rows=[]; market_ok=s.above20.mean()>=.40 and not unknown
    for _,r in s.sort_values(['score','Code'],ascending=[False,True]).iterrows():
        keys=memberships[r.Code]; is_held=r.Code in held_codes
        strong=bool(set(keys)&confirmed)
        base=(r.Code in leaders and r.eligible and r.close>r.ma20 and r.rs5>0 and r.r1>0 and
              r.bias20<=.20 and r.r5<=.25 and not r.upper_limit)
        if not market_ok or not base or not keys or r.Code in blocked: continue
        if is_held:
            p=held.loc[r.Code]
            if not strong or r.close<=p.AvgCost or r.close<=r.ma200 or r.score<c['stock_threshold'] or r.flow5<1.1: continue
            action='追加候補'; existing=p.Shares*r.close
            # 一度に全枠を使わず資金の2%刻みで積み増す。
            allocation=min(c['capital']*.02,c['capital']*c['max_position_pct']-existing)
            stop=max(p.StopPrice,p.AvgCost*(1-c['stop_pct']))
            remaining_risk=c['capital']*c['risk_pct']-max(p.AvgCost-stop,0)*p.Shares
            allocation=min(allocation,max(remaining_risk,0)/max(r.close-stop,1e-9)*r.close)
            # trialからconfirmedへの変更は利用者が約定後に行う。変更前は打診枠も守る。
            trial=p.Stage=='trial'
        else:
            if r.value_ratio<1.5 or r.closepos<.65: continue
            action='打診候補'; existing=0; trial=True; stop=r.close*(1-c['stop_pct'])
            allocation=min(c['capital']*.02,c['capital']*c['max_position_pct'],c['capital']*c['risk_pct']/c['stop_pct'])
            if len(held_codes)>=c['top_n'] or any(counts.get(k,0)>=c['per_group'] for k in exposures(r.Code)): continue
        allocation=min(allocation,max(c['capital']*c['max_basket_pct']-total,0))
        if trial: allocation=min(allocation,max(c['capital']*.10-trial_total,0))
        shares=math.floor(allocation/r.close/c['lot'])*c['lot']
        if shares<=0: continue
        item=r.to_dict();item.update(signal=action,lot_estimate=shares,reference_stop=stop,groups=' / '.join(keys),
                                   reason=' / '.join(keys)+('：広がり確認・含み益あり' if is_held else '：複数銘柄の予兆・代金増加'),reference_amount=shares*r.close)
        rows.append(item);total+=shares*r.close
        if trial: trial_total+=shares*r.close
        if not is_held:
            held_codes.add(r.Code)
            for key in exposures(r.Code): counts[key]=counts.get(key,0)+1
    picks=pd.DataFrame(rows) if rows else s.iloc[:0].assign(reference_amount=pd.Series(dtype=float))
    return picks,review,evidence


def md_table(df, columns):
    if df.empty:
        return '該当なし。条件を緩めて無理に選定しません。\n'
    names=list(columns)
    def cell(v):
        if isinstance(v,(float,np.floating)):
            return '—' if not np.isfinite(v) else f'{v:.3f}'
        return str(v).replace('|','／').replace('\n',' ')
    return '| '+' | '.join(columns.values())+' |\n|'+ '|'.join(['---']*len(names))+'|\n'+'\n'.join('| '+' | '.join(cell(r.get(k,'')) for k in names)+' |' for _,r in df.iterrows())+'\n'


def ranking_rotation(raw, master, sessions, asof, c, themes=None, businesses=None):
    """直近26営業日の1/5/20日ランキング。固定の比較母集団で構成変化を測る。"""
    days = [d for d in sessions if d <= asof]
    lookback = c['ranking_history_days']
    required = lookback + 20
    if len(days) < required:
        raise ValueError(f'ランキングには最低{required}営業日のデータが必要です')
    days = days[-required:]
    data = validate(raw)
    data = data[data.Date.isin(days) & data.Code.isin(master.Code)]
    def panel(col):
        return data.pivot(index='Date', columns='Code', values=col).reindex(days)
    close, factor, value = panel('C'), panel('AdjFactor'), panel('Va')
    # 比較期間の全日で観測できる共通母集団。IPO/売買停止/未取得による見かけの交代を回避。
    good = (close.notna().all() & factor.notna().all() & value.gt(0).all())
    codes = list(good[good].index)
    coverage = len(codes) / len(master)
    if coverage < c['min_coverage']:
        raise ValueError(f'ランキング比較母集団の充足率 {coverage:.1%}。基準未達です')
    close, factor, value = close[codes], factor[codes], value[codes]
    cumulative = factor.iloc[::-1].cumprod().iloc[::-1] / factor
    close = close * cumulative
    liquidity = value.shift(1).rolling(20).mean()
    flow = value.rolling(5).mean() / value.shift(5).rolling(20).mean()
    means20 = close.rolling(20).mean()
    dates = days[-lookback:]
    mapping = master.set_index('Code')
    histories, summaries, latest = [], [], {}
    class_groups = classification_groups(master,themes,businesses)
    business_tags={}
    if businesses is not None and not businesses.empty:
        business_tags=businesses.groupby('Code').Label.agg(lambda labels:' / '.join(sorted(set(labels)))).to_dict()
    for window in [1,5,20]:
        returns = close.pct_change(window, fill_method=None)
        memberships, frames = {}, {}
        for d in dates:
            r = returns.loc[d]
            allowed = r.gt(0) & liquidity.loc[d].ge(c['ranking_min_value'])
            ranked = pd.DataFrame({'Code': r.index[allowed], 'return': r[allowed].values})
            ranked = ranked.sort_values(['return','Code'], ascending=[False,True]).head(c['ranking_top_n']).reset_index(drop=True)
            ranked['rank'] = np.arange(1,len(ranked)+1)
            ranked['Date'],ranked['period'] = d,window
            ranked['CoName'] = ranked.Code.map(mapping.CoName)
            ranked['S33Nm'] = ranked.Code.map(mapping.S33Nm)
            ranked['BusinessTags'] = ranked.Code.map(business_tags).fillna('未分類')
            memberships[d] = set(ranked.Code)
            frames[d] = ranked
        # 連続登場は保存前歴がなくても価格履歴から再構成。最大lookback日で打ち切り。
        for i,d in enumerate(dates):
            frame = frames[d]
            before = memberships[dates[i-1]] if i else None
            frame['entry'] = frame.Code.map(lambda code: '比較履歴なし' if before is None else ('新規' if code not in before else '継続'))
            streaks = {}
            for code in memberships[d]:
                n=0
                for prior in reversed(dates[:i+1]):
                    if code not in memberships[prior]:
                        break
                    n+=1
                streaks[code]=n
            frame['streak'] = frame.Code.map(streaks)
            frame['streak_capped'] = frame.streak.eq(i+1)
            histories.append(frame)
        latest[window] = frames[asof].copy()
        snapshot=latest[window]
        for p in [1,5,20]:
            snapshot[f'r{p}_pct']=snapshot.Code.map(close.pct_change(p,fill_method=None).loc[asof])*100
        snapshot['flow5']=snapshot.Code.map(flow.loc[asof])
        snapshot['value_oku']=snapshot.Code.map(value.loc[asof])/1e8
        snapshot['liquidity_oku']=snapshot.Code.map(liquidity.loc[asof])/1e8
        today, yesterday, before5 = [memberships[d] for d in [dates[-1],dates[-2],dates[-6]]]
        n_universe = len(codes)
        for kind, sector, registered_members in class_groups:
            members=registered_members & set(codes)
            class_coverage=len(members)/len(registered_members) if registered_members else 0.0
            n=len(members)
            if n==0:
                continue
            now = members & today
            prev = members & yesterday
            old = members & before5
            rate=len(now)/n
            old_rate=len(old)/n
            market_rate=len(today)/n_universe
            concentration=rate/market_rate if market_rate>0 else 0.0
            concentration5=(len(old)/n)/(len(before5)/n_universe) if before5 else 0.0
            entrants=now-prev
            exits=prev-now
            valid_liquid=[code for code in members if liquidity.loc[asof,code]>=c['min_value']]
            # 薄商いの急騰だけでは確認済み転換にしない。
            sector_flow=float(flow.loc[asof,valid_liquid].median()) if valid_liquid else np.nan
            sector_ret=float(returns.loc[asof,list(members)].median())
            b20=float((close.loc[asof,list(members)]>means20.loc[asof,list(members)]).mean())
            liquid_leaders=len(set(valid_liquid)&now)
            recent_rates=[len(members & memberships[d])/n for d in dates[-3:]]
            sustained=all(rate_ > old_rate for rate_ in recent_rates)
            streak3=sum(int(frames[asof].set_index('Code').loc[code,'streak'])>=3 for code in now)
            delta=len(now)-len(old)
            delta_rate=rate-old_rate
            # セクター規模、相対集中、3日持続、代金の裏付けを同時確認。
            if n < c['min_group'] or class_coverage < c['min_coverage']:
                signal='判定不足'
            elif delta<=-c['rotation_min_change'] and delta_rate<=-c['rotation_rate_change']:
                signal='広がり縮小'
            elif delta>=c['rotation_min_change'] and delta_rate>=c['rotation_rate_change'] and concentration>=1.2:
                signal='転換確認候補' if sustained and sector_flow>=1.1 and liquid_leaders>=3 and sector_ret>0 else '転換の兆し'
            elif concentration>=1.2 and streak3>=3 and sector_ret>0:
                signal='上位定着'
            else:
                signal='監視'
            summaries.append(dict(kind=kind, period=window, sector=sector, universe=n, registered=len(registered_members), classification_coverage=class_coverage, top_count=len(now),
                                  count_prev=len(prev),count_5ago=len(old),delta1=len(now)-len(prev),delta5=delta,
                                  rate=rate,rate_delta5=delta_rate,concentration=concentration,
                                  concentration_delta5=concentration-concentration5,
                                  entrants=len(entrants),exits=len(exits),streak3=streak3,
                                  liquid_leaders=liquid_leaders,flow5=sector_flow,sector_return=sector_ret,
                                  breadth20=b20,sustained3=sustained,signal=signal,
                                  leaders=' / '.join(frames[asof].loc[frames[asof].Code.isin(members),'Code'].head(5)),
                                  entrants_codes=' / '.join(sorted(entrants)), exits_codes=' / '.join(sorted(exits))))
    sectors=pd.DataFrame(summaries)
    # 過熱度は転換の監視とは別軸。売買候補の判定を上書きしない。
    sectors=sectors.sort_values(['period','concentration','delta5'],ascending=[True,False,False])
    combined=[]
    for (kind,sector),g in sectors.groupby(['kind','sector']):
        signals=dict(zip(g.period,g.signal))
        turn=[p for p in [1,5,20] if signals.get(p)=='転換確認候補']
        signs=[p for p in [1,5,20] if signals.get(p)=='転換の兆し']
        declines=[p for p in [1,5,20] if signals.get(p)=='広がり縮小']
        if all(s=='判定不足' for s in signals.values()):
            verdict='判定不足'
        elif len(turn)>=2:
            verdict='複数期間で転換確認候補'
        elif turn:
            verdict='転換確認候補'
        elif signs:
            verdict='転換の兆し'
        elif declines:
            verdict='広がり縮小'
        elif any(s=='上位定着' for s in signals.values()):
            verdict='上位定着'
        else:
            verdict='監視'
        if (turn or signs) and declines:
            verdict='期間で方向不一致'
        combined.append(dict(kind=kind, sector=sector, signal=verdict, **{f'signal_{p}d':signals.get(p,'判定不足') for p in [1,5,20]}))
    overview=pd.DataFrame(combined)
    priority={'複数期間で転換確認候補':0,'転換確認候補':1,'転換の兆し':2,'期間で方向不一致':3,'上位定着':4,'広がり縮小':5,'監視':6,'判定不足':7}
    overview['_order']=overview.signal.map(priority)
    overview=overview.sort_values(['_order','kind','sector']).drop(columns='_order')
    return latest, pd.concat(histories,ignore_index=True), sectors, overview, coverage


def ranking_report(latest, sectors, overview, c, coverage):
    body=f'''\n## 上昇率上位{c['ranking_top_n']}銘柄から読む業種転換\n
1日・5日・20日の上昇率をそれぞれ順位付け。プラス騰落銘柄のみを最大{c['ranking_top_n']}件抽出します。
比較母集団の充足率 {coverage:.1%}。直近{c['ranking_history_days']+20}営業日を連続観測できた銘柄を共通母集団にし、IPO・欠損による見かけの増減を抑えています。
業種は分析日現在、テーマ・事業は登録表の分類で過去分も再集計します。履歴上の当時の構成・分類を復元したバックテストではありません。

'''
    body+=md_table(overview,{'kind':'区分','sector':'業種・テーマ・事業','signal':'総合監視判定','signal_1d':'1日上昇率','signal_5d':'5日上昇率','signal_20d':'20日上昇率'})
    for period in [1,5,20]:
        body+=f'\n### {period}日上昇率ランキングの業種別内訳\n\n'
        table=sectors[sectors.period.eq(period)].copy()
        table['rate_pct']=table.rate*100
        table['change_pp']=table.rate_delta5*100
        body+=md_table(table,{'kind':'区分','sector':'業種・テーマ・事業','universe':'母数','top_count':'ランク内','delta1':'前日比','delta5':'5日前比','rate_pct':'登場率%','change_pp':'登場率差pt','concentration':'集中倍率','entrants':'新規','exits':'退出','streak3':'3日連続','flow5':'代金倍率','signal':'判定','leaders':'上位コード'})
    body+='''
- 登場率＝業種内ランク入り銘柄数÷業種の共通母集団。集中倍率＝業種登場率÷市場全体登場率。1倍超は相対的な偏りを表します。
- 「転換の兆し」は5日前比の登場銘柄数と登場率の増加、集中倍率1.2倍以上を条件とします。「転換確認候補」はさらに3日間の広がり、流動性のある主導株3銘柄以上、代金倍率1.1以上、業種騰落率中央値プラスを確認します。
- 日次ランキングに新規登場しても「上場後初」ではありません。連続登場日数は再構成期間内の下限値です。streak_capped=Trueなら、それ以前から続いている可能性があります。
- 強い業種からの押し出しでも順位は下がるため、「広がり縮小」は資金流出や株価下落の断定ではありません。既存の旬スコア・チャート・材料と併用してください。
- 転換確認候補は追加判定にも使用します。早期打診は別の3営業日条件を使用。過熱回避・損切りの規律を維持します。
'''
    return body


def analysis_input_report(asof, groups, latest, rank_history, rotation, coverage, rank_coverage, demo=False):
    """分類を固定せず、ChatGPTが横断的な共通点を検討するための観測データ。"""
    industry_groups=groups[groups.kind.eq('業種')].copy()
    industry_rotation=rotation[rotation.kind.eq('業種')].copy()
    # 上位500×3期間の和集合。テーマ・事業のタグを根拠として持ち込まない。
    frames=[]
    for period,frame in latest.items():
        f=frame.copy()
        f=f.rename(columns={'rank':f'rank_{period}d','entry':f'entry_{period}d','streak':f'streak_{period}d'})
        frames.append(f)
    common=['Code','CoName','S33Nm','r1_pct','r5_pct','r20_pct','flow5','value_oku','liquidity_oku']
    stocks=pd.concat([f.reindex(columns=common) for f in frames],ignore_index=True).drop_duplicates('Code')
    for period,f in zip([1,5,20],frames):
        fields=['Code',f'rank_{period}d',f'entry_{period}d',f'streak_{period}d']
        stocks=stocks.merge(f.reindex(columns=fields),on='Code',how='left',validate='one_to_one')
    stocks['_priority']=stocks[[f'rank_{p}d' for p in [1,5,20]]].min(axis=1)
    stocks=stocks.sort_values(['_priority','Code']).drop(columns='_priority')
    for col in stocks.select_dtypes(include='number'):
        stocks[col]=stocks[col].round(3)
    dates=sorted(rank_history.Date.unique())[-6:]
    recent=rank_history[rank_history.Date.isin(dates)]
    counts=recent.groupby(['period','S33Nm','Date']).Code.nunique()
    trend=[]
    for _,r in industry_rotation.iterrows():
        trend.append(dict(period=r.period,industry=r.sector,
                          daily_counts=' → '.join(str(counts.get((r.period,r.sector,d),0)) for d in dates)))
    banner='【架空データ・動作例】' if demo else ''
    text=f'''# {banner}ChatGPT分析用・33業種と主導銘柄 {asof}

分析対象日は{asof}、騰落率は配当を含まない終値基準。株式分割等の調整を反映。
長期指標のデータ充足率 {coverage:.1%}、上昇率ランキングの共通母集団充足率 {rank_coverage:.1%}。
固定のテーマ分類・事業分類は、この資料の分析根拠に使用していません。

## この資料を読むときの依頼

1. 33業種の強弱と直近の変化を整理してください。
2. ランク入り銘柄を業種横断で見て、共通する製品・事業・顧客・設備投資需要などからテーマの仮説を挙げてください。
3. 各仮説には根拠の銘柄名と観測値を添え、単一銘柄だけの動き、複数銘柄への広がり、一時的な急騰を区別してください。
4. 価格から観測できる事実、事業内容からの推論、ニュースで確認した材料を分けてください。事業・ニュースの事実確認が必要な場合は企業の公式資料等を確認し、未確認の因果関係を断定しないでください。
5. 初動らしいテーマ、継続、過熱、失速、反証条件を示してください。新しいテーマを無理に作らず、根拠不足ならその旨を述べてください。
6. 過去に共有されたレポートが参照できる場合は比較してください。参照できない過去レポートの内容を推測しないでください。

## 読み方と限界

業種は東証33業種。データ不足の業種も一覧に残します。ランキングは1日・5日・20日上昇率それぞれプラス銘柄の最大500件。
直近26営業日の比較は直近46営業日を連続観測できた共通母集団内で再構成。IPO・売買停止等の一部は対象外です。
業種は分析日現在の分類を過去分にも適用しています。売買代金や順位は純資金流入を直接観測した値ではありません。
旬スコア・転換ラベルはルールによる補助指標です。テーマ名や値動きの原因をプログラムが確定したものではありません。
コードはJ-Quants形式で原則5文字、末尾0は普通株識別用。r1/r5/r20_pctの単位は%、value_okuとliquidity_okuは億円。
flow5は直近5日平均代金÷その前20日平均代金。rankの空欄は該当期間のプラス銘柄上位500に入っていないことを示します。
連続登場日数は再構成期間内の下限（最大26日）。本資料は現在のランク入り銘柄を収録し、全銘柄の事業内容やニュースは含みません。

## 33業種の全体一覧

'''
    text+=md_table(industry_groups,{'group':'業種','n':'有効銘柄','coverage':'充足率','score':'旬スコア','delta5':'5日前差','rs5':'5日相対強度','rs20':'20日相対強度','breadth':'20日線上比率','flow5':'代金倍率','stage':'局面'})
    text+='\n## 業種別ランキング変化\n\n'
    for period in [1,5,20]:
        text+=f'\n### {period}日上昇率の上位500から集計\n\n'
        text+=md_table(industry_rotation[industry_rotation.period.eq(period)],{'sector':'業種','universe':'母数','top_count':'ランク内','delta1':'前日比','delta5':'5日前比','rate':'登場率','concentration':'集中倍率','entrants':'新規','exits':'退出','streak3':'3日連続','flow5':'代金倍率','signal':'監視ラベル','leaders':'上位コード','entrants_codes':'新規コード','exits_codes':'退出コード'})
    text+='\n## 直近6営業日の日別ランク入り数\n\n日付：'+' → '.join(dates)+'\n\n'
    text+=md_table(pd.DataFrame(trend),{'period':'上昇率期間','industry':'業種','daily_counts':'日別件数'})
    text+='\n## 3期間のランク入り銘柄の和集合（CSV）\n\n同じ銘柄は1行です。テーマの共通点は企業の事業を確認して検討してください。\n\n```csv\n'
    text+=stocks.to_csv(index=False)
    text+='```\n'
    return text, stocks


def report(master, raw, asof, sessions, themes, c, out, demo=False, businesses=None, positions=None):
    f, coverage=features(raw,master,asof,sessions,c)
    sessions=[d for d in sessions if d<=asof]
    latest, rank_history, rotation, rotation_overview, rank_coverage = ranking_rotation(raw,master,sessions,asof,c,themes,businesses)
    old_date=sessions[-6]
    prev,_=features(raw,master,old_date,sessions,c)
    groups=stages(group_table(f,master,themes,c,businesses),group_table(prev,master,themes,c,businesses),c)
    stocks,picks=select(f,groups,themes,c,businesses)
    if positions is None: positions=load_positions(None,asof,sessions)
    legacy_picks=picks.copy()
    picks,position_review,early_groups=early_plan(stocks,master,rank_history,rotation,positions,sessions,asof,c,themes,businesses)
    breadth=f.above20.mean()
    regime='新規エントリー抑制' if breadth < .40 else '通常監視'
    if breadth < .40:
        picks=picks.iloc[:0]
    out=Path(out)/asof
    out.mkdir(parents=True,exist_ok=True)
    for name,frame in [('groups',groups),('stocks',stocks),('picks',picks),('confirmed_watch',legacy_picks),('position_review',position_review),('early_groups',early_groups)]:
        atomic_text(out/(name+'.csv'),frame.to_csv(index=False))
    for period,frame in latest.items():
        atomic_text(out/f'ranking_{period}d.csv',frame.to_csv(index=False))
    for name,frame in [('ranking_history',rank_history),('sector_rotation',rotation),('rotation_overview',rotation_overview)]:
        atomic_text(out/(name+'.csv'),frame.to_csv(index=False))
    if businesses is not None:
        audit=businesses.copy()
        audit['in_master']=audit.Code.isin(master.Code)
        atomic_text(out/'business_classification.csv',audit.to_csv(index=False))
        atomic_text(out/'business_rotation.csv',rotation[rotation.kind.eq('事業')].to_csv(index=False))
    title='【架空データ・動作例】' if demo else ''
    body=f'''# {title}旬レポート {asof}

対象：東証プライム・スタンダード・グロースの普通株式。分析基準は当該日の終値。

市場環境：**{regime}**。20日線上比率 {breadth:.1%}。データ充足率 {coverage:.1%}（{len(f)}/{len(master)}銘柄）。
相対強度は分析可能銘柄のリターン中央値との差。TOPIX比較ではありません。
売買代金・価格による資金集中の代理指標であり、純流入額や心理を直接観測するものではありません。

## 分類別一覧（標準は33業種）

'''+md_table(groups,{'kind':'区分','group':'分類','n':'銘柄数','score':'旬スコア','delta5':'5営業日前比','breadth':'20日線上比率','flow5':'代金倍率','stage':'局面'})+'''
## 打診・追加候補（保有登録反映）

'''+md_table(picks,{'Code':'コード','CoName':'銘柄','signal':'判定','score':'点数','close':'終値','reference_stop':'参考損切り','lot_estimate':'参考株数','atr_warning':'値動き注意','reason':'根拠'})+'''
## 監視上位（購入候補とは別）

'''+md_table(stocks.head(20),{'Code':'コード','CoName':'銘柄','signal':'判定','score':'点数','groups':'旬の分類','reason':'根拠'})+f'''
## 規律

- 新規打診の参考損切りは終値の-{c['stop_pct']:.0%}。実際は約定価格を基準に設定。保有は登録StopPriceと平均取得価格基準の高い方で点検。窓開け・値幅制限ではその価格で売れず損失が超過します。
- 参考株数は仮の運用資金 {c['capital']:,.0f}円、1銘柄許容損失 {c['risk_pct']:.1%}、1銘柄配分上限 {c['max_position_pct']:.0%}、候補バスケット上限 {c['max_basket_pct']:.0%}、{c['lot']}株単位で算出。positions.csvの登録保有を時価評価して枠を控除。未登録の保有は反映されません。打診20万円・全打診100万円（資金比2%・10%）、追加は資金比2%刻み。
- ATR注意=True は直近14日平均値幅の2倍が損切り幅を超える状態。参考株数は発注数量ではありません。
- 過熱は追随回避・保有時の一部利確検討、失速は撤退点検。利益率だけを理由に全売却しません。
- 決算・増資・信用規制・材料の事実確認は別途必要。ニュース/SNS解析と発注は未実装。保有は手動登録し、終値で撤退点検します。
- 閾値・点数は仮説であり、統計的優位性・収益性は未検証。バックテスト結果ではありません。
'''
    if businesses is not None:
        matched=businesses[businesses.Code.isin(master.Code)]
        body+=f'\n## 事業分類の対象範囲\n\n登録 {businesses.Label.nunique()}区分・{businesses.Code.nunique()}銘柄、当日マスターとの一致 {matched.Code.nunique()}銘柄。未登録銘柄は未分類です。公式の業種分類ではなく、事業内容を参考にした監視バスケットです。複合企業の株価は他事業でも動き、登録事業の売上比率は重み付けしていません。細分した分類は3銘柄未満またはデータ充足率不足なら判定不足とします。\n'
        body+=md_table(rotation_overview[rotation_overview.kind.eq('事業')],{'sector':'事業分類','signal':'監視判定','signal_1d':'1日','signal_5d':'5日','signal_20d':'20日'})
    body+='\n## 保有の撤退点検\n\n'+md_table(position_review,{'Code':'コード','action':'判定','stop':'撤退価格','held_sessions':'保有営業日','reason':'理由'})
    body+='\n## 早期の広がり\n\n'+md_table(early_groups,{'kind':'区分','group':'分類','new_liquid':'新規・流動性銘柄数','delta3':'登場数3日前比','signal':'判定'})
    body+=ranking_report(latest,rotation,rotation_overview,c,rank_coverage)
    analysis_text,analysis_stocks=analysis_input_report(asof,groups,latest,rank_history,rotation,coverage,rank_coverage,demo)
    atomic_text(out/'analysis_input.md',analysis_text+'\n## 早期の広がり\n\n'+md_table(early_groups,{'kind':'区分','group':'分類','new_liquid':'新規銘柄数','delta3':'3日前比','signal':'判定'}))
    atomic_text(out/'analysis_stocks.csv',analysis_stocks.to_csv(index=False))
    if businesses is None:
        for filename in ['business_rotation.csv','business_classification.csv']:
            (out/filename).unlink(missing_ok=True)
    atomic_text(out/'report.md',body)
    metadata=dict(version='2.3', asof=asof, demo=demo, coverage=coverage, ranking_coverage=rank_coverage, config=c, stocks=len(f), picks=len(picks),
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
    p.add_argument('--positions',type=Path,default=None,help='保有CSV。未指定は同フォルダのpositions.csvがあれば使用')
    p.add_argument('--classified',action='store_true',help='任意: 固定のテーマ・事業分類を使う。標準は33業種のみ')
    p.add_argument('--demo',action='store_true',help='ネット不要の架空データ')
    p.add_argument('--date',help='分析上限日 YYYY-MM-DD。未指定は日本時間で最新の確定営業日')
    p.add_argument('--csv-dir',type=Path,help='master.csv / prices.csv / calendar.csv を読み込む')
    p.add_argument('--config',type=Path,default=BASE/'config.json')
    p.add_argument('--themes',type=Path,default=BASE/'themes.csv')
    p.add_argument('--businesses',type=Path,default=BASE/'businesses.csv',help='事業分類表。未配置なら内蔵初期分類を使用')
    p.add_argument('--cache',default=str(BASE/'cache'))
    p.add_argument('--out',default=None)
    p.add_argument('--refresh',action='store_true',help='全日付のキャッシュを更新')
    args=p.parse_args()
    c=config(args.config)
    if args.demo:
        master,raw,asof,sessions=demo_data()
        themes=pd.DataFrame(columns=['Code','Theme','Subtheme'])
        businesses=pd.DataFrame({'Code':master.Code[:6], 'Parent':['デモ半導体']*6,'Business':['成膜装置']*3+['検査装置']*3})
        businesses['Label']=businesses.Parent+' / '+businesses.Business
        if not args.classified:
            businesses=None
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
        themes=pd.read_csv(args.themes,dtype=str).fillna('') if args.classified and args.themes.exists() else pd.DataFrame(columns=['Code','Theme','Subtheme'])
        businesses=load_businesses(args.businesses) if args.classified else None
        if not {'Code','Theme'}.issubset(themes):
            raise ValueError('themes.csv は Code,Theme 列が必要です')
        if not themes.empty:
            themes['Code']=themes.Code.str.strip().str.upper().map(lambda s:s+'0' if len(s)==4 else s)
            unknown=set(themes.Code)-set(master.Code)
            if unknown:
                print('注意: マスターにないテーマ構成銘柄を除外: '+','.join(sorted(unknown)))
            themes=themes.drop_duplicates(['Code','Theme'])
    position_path=args.positions or (BASE/'positions.csv' if (BASE/'positions.csv').exists() and not args.demo else None)
    positions=load_positions(position_path,asof,sessions)
    report(master,raw,asof,sessions,themes,c,args.out or BASE/('demo_reports' if args.demo else 'reports'),args.demo,businesses,positions)


if __name__=='__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError, KeyError) as e:
        print('停止: '+str(e),file=sys.stderr)
        sys.exit(1)
