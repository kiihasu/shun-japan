"""v2.3の逐次バックテスト。現行マスター固定の参考検証。発注なし。"""
import argparse
import json
import math
from pathlib import Path
import pandas as pd
import numpy as np
import shun


def sell_quantity(shares, half, lot=100):
    return math.floor(shares/2/lot)*lot if half else shares


def executable(bar, side):
    # OHLCだけでは板の約定可否を復元できない。値幅制限接触は保守的に見送る。
    flag='UL' if side=='BUY' else 'LL'
    return pd.notna(bar.O) and bar.O>0 and bar.Va>0 and str(bar.get(flag,'0')) not in ['1','1.0']


def simulate(master,raw,sessions,c,test_days=60,fee=.0005,slip=.001):
    raw=shun.validate(raw); sessions=sorted(sessions)
    start=max(c['min_history']+5,len(sessions)-test_days)
    if start>=len(sessions)-1:raise ValueError('準備期間後に最低2営業日必要です')
    cash=float(c['capital']); positions={}; pending=[]; trades=[]; curve=[]; diagnostics=[]; last_prices={}
    def sell(code,quantity,price,date,reason):
        nonlocal cash
        p=positions[code];quantity=min(quantity,p['Shares'])
        if quantity<=0:return
        fill=price*(1-slip);cost=fill*quantity*fee
        pnl=(fill-p['AvgCost'])*quantity-cost-p['buy_fee']*quantity/p['Shares']
        p['buy_fee']*=1-quantity/p['Shares'];cash+=fill*quantity-cost
        trades.append(dict(Date=date,Code=code,Side='SELL',Shares=quantity,Price=fill,Fee=cost,PnL=pnl,Reason=reason))
        p['Shares']-=quantity
        if p['Shares']<=0:del positions[code]
    for i in range(start,len(sessions)):
        date=sessions[i]; day=raw[raw.Date.eq(date)].set_index('Code'); retry=[]
        # 当日の分割を保有に反映。端株の現金処理を仮定せず停止する。
        for code,p in list(positions.items()):
            if code not in day.index:raise ValueError(f'{date} {code}: 保有株価欠損。成績計算停止')
            factor=float(day.loc[code,'AdjFactor'])
            if factor!=1:
                shares=p['Shares']/factor
                if not np.isclose(shares,round(shares)):raise ValueError('分割・併合の端株処理が必要です')
                p['Shares']=int(round(shares));p['AvgCost']*=factor;p['StopPrice']*=factor
        for order in pending:
            if order['Code'] in day.index:
                factor=float(day.loc[order['Code'],'AdjFactor'])
                order['shares']=math.floor(order['shares']/factor/c['lot'])*c['lot']
                if 'signal_close' in order:order['signal_close']*=factor
        # 前日までに確定した売却を先に実行。失敗した売却は翌日へ継続。
        for order in [o for o in pending if o['side']=='SELL']:
            code=order['Code']
            if code not in positions:continue
            if code not in day.index or not executable(day.loc[code],'SELL'):
                retry.append(order);continue
            sell(code,order['shares'],float(day.loc[code,'O']),date,order['reason'])
        # 買付は翌日始値＋スリッページ。約定価格で資金・個別・打診上限を再確認。
        for order in [o for o in pending if o['side']=='BUY']:
            code=order['Code']
            if code not in day.index or not executable(day.loc[code],'BUY'):continue
            b=day.loc[code];price=float(b.O)*(1+slip)
            if code in positions and price<=positions[code]['AvgCost']:continue
            if price>order['signal_close']*1.05:continue
            existing=positions.get(code)
            mark=lambda k:float(day.loc[k,'O']) if k in day.index and pd.notna(day.loc[k,'O']) else last_prices[k]
            total=sum(p['Shares']*mark(k) for k,p in positions.items())
            trials=sum(p['Shares']*mark(k) for k,p in positions.items() if p['Stage']=='trial')
            own=existing['Shares']*price if existing else 0
            budget=min(cash/(1+fee),c['capital']*.02,c['capital']*c['max_position_pct']-own,c['capital']*c['max_basket_pct']-total)
            if existing is None:budget=min(budget,c['capital']*.10-trials)
            stop=max(existing['StopPrice'],existing['AvgCost']*(1-c['stop_pct'])) if existing else price*(1-c['stop_pct'])
            risk_used=max(existing['AvgCost']-stop,0)*existing['Shares'] if existing else 0
            budget=min(budget,max(c['capital']*c['risk_pct']-risk_used,0)/max(price-stop,1e-9)*price)
            shares=min(order['shares'],math.floor(max(budget,0)/price/c['lot'])*c['lot'])
            if shares<=0:continue
            cost=price*shares*fee;cash-=price*shares+cost
            if existing:
                n=existing['Shares'];existing['AvgCost']=(existing['AvgCost']*n+price*shares)/(n+shares)
                existing['Shares']+=shares;existing['Stage']='confirmed';existing['buy_fee']+=cost
                existing['StopPrice']=max(stop,existing['AvgCost']*(1-c['stop_pct']))
            else:
                positions[code]=dict(Code=code,EntryDate=date,AvgCost=price,Shares=shares,Stage='trial',StopPrice=stop,buy_fee=cost,heat_latched=False)
            trades.append(dict(Date=date,Code=code,Side='BUY',Shares=shares,Price=price,Fee=cost,PnL=np.nan,Reason=order['reason']))
        # 常設の損切りを模擬。窓開けなら始値、日中到達なら撤退価格から滑らせる。
        for code,p in list(positions.items()):
            b=day.loc[code];stop=max(p['StopPrice'],p['AvgCost']*(1-c['stop_pct']))
            if b.L<=stop:
                if executable(b,'SELL'):sell(code,p['Shares'],min(float(b.O),stop),date,'逆指値損切り')
                else:retry.append(dict(Code=code,side='SELL',shares=p['Shares'],reason='未約定損切り'))
        # 当日以降のデータをシグナルに渡さない。
        hist=raw[raw.Date.le(date)];ds=sessions[:i+1]
        f,_=shun.features(hist,master,date,ds,c)
        prev,_=shun.features(hist,master,ds[-6],ds,c)
        themes=pd.DataFrame(columns=['Code','Theme'])
        groups=shun.stages(shun.group_table(f,master,themes,c),shun.group_table(prev,master,themes,c),c)
        stocks,_=shun.select(f,groups,themes,c)
        _,rh,rot,_,_=shun.ranking_rotation(hist,master,ds,date,c)
        pframe=pd.DataFrame(list(positions.values())) if positions else shun.load_positions(None,date,ds)
        picks,review,_=shun.early_plan(stocks,master,rh,rot,pframe,ds,date,c)
        pending=retry
        # 同銘柄の未約定売却があれば新規注文を重ねない。
        blocked={o['Code'] for o in pending}
        for _,r in review.iterrows():
            code=r.Code;p=positions[code]
            heat=r.action=='一部利確点検'
            if not heat:p['heat_latched']=False
            if code in blocked:continue
            full=r.action in ['損切り条件到達','期限撤退点検','失速・撤退点検']
            if full or (heat and not p['heat_latched']):
                quantity=sell_quantity(p['Shares'],not full,c['lot'])
                if heat:p['heat_latched']=True
                if quantity:
                    pending.append(dict(Code=code,side='SELL',shares=quantity,reason=r.action));blocked.add(code)
        for _,r in picks.iterrows():
            if r.Code not in blocked:pending.append(dict(Code=r.Code,side='BUY',shares=int(r.lot_estimate),reason=r.signal,signal_close=r.close))
        for code,p in positions.items():
            if code not in day.index or pd.isna(day.loc[code,'C']):raise ValueError('保有の終値欠損')
            last_prices[code]=float(day.loc[code,'C'])
        market_value=sum(p['Shares']*last_prices[k] for k,p in positions.items())
        curve.append(dict(Date=date,Cash=cash,MarketValue=market_value,Equity=cash+market_value,Positions=len(positions)))
        diagnostics.append(dict(Date=date,Candidates=len(picks),Pending=len(pending)))
        if len(curve)%10==0:print(f'検証 {len(curve)}/{len(sessions)-start} 営業日',flush=True)
    eq=pd.DataFrame(curve);t=pd.DataFrame(trades,columns=['Date','Code','Side','Shares','Price','Fee','PnL','Reason'])
    eq['Drawdown']=eq.Equity/eq.Equity.cummax().clip(lower=c['capital'])-1
    sells=t[t.Side.eq('SELL')];profits=sells.PnL
    result=dict(initial_capital=c['capital'],final_equity=float(eq.Equity.iloc[-1]),return_pct=float((eq.Equity.iloc[-1]/c['capital']-1)*100),max_drawdown_pct=float(eq.Drawdown.min()*100),sell_fills=len(sells),sell_fill_win_rate_pct=float(profits.gt(0).mean()*100) if len(sells) else None,fees=float(t.Fee.sum()),open_positions=len(positions),from_date=eq.Date.iloc[0],to_date=eq.Date.iloc[-1],fee_rate=fee,slippage_rate=slip)
    return result,eq,t,pd.DataFrame(diagnostics),pd.DataFrame(list(positions.values()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--demo',action='store_true');p.add_argument('--csv-dir',type=Path)
    p.add_argument('--test-days',type=int,default=60);p.add_argument('--date');p.add_argument('--cache',default='cache');p.add_argument('--refresh',action='store_true')
    p.add_argument('--fee',type=float,default=.0005);p.add_argument('--slippage',type=float,default=.001)
    p.add_argument('--out',type=Path,default=Path('backtest_results'));args=p.parse_args()
    if not 0<=args.fee<.1 or not 0<=args.slippage<.1 or args.test_days<2:raise ValueError('引数範囲が不正です')
    c=shun.config(shun.BASE/'config.json')
    if args.demo:master,raw,_,sessions=shun.demo_data()
    elif args.csv_dir:
        master=pd.read_csv(args.csv_dir/'master.csv',dtype={'Code':str});raw=pd.read_csv(args.csv_dir/'prices.csv',dtype={'Code':str})
        cal=pd.read_csv(args.csv_dir/'calendar.csv',dtype=str);sessions=sorted(cal.loc[cal.HolDiv.isin(['1','2']),'Date'])
        upper=args.date or raw.Date.max();sessions=[d for d in sessions if d<=upper];raw=raw[raw.Date<=upper]
    else:
        c['history_sessions']=max(280,args.test_days+c['min_history']+5)
        master,raw,_,sessions=shun.live_data(c,args)
        if len(sessions)<c['history_sessions']:raise ValueError('取得可能なカレンダー長が不足。CSV入力か短い検証期間を指定してください')
    result,eq,trades,diag,holdings=simulate(master,raw,sessions,c,args.test_days,args.fee,args.slippage)
    result['demo']=args.demo;args.out.mkdir(parents=True,exist_ok=True)
    shun.atomic_text(args.out/'summary.json',json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    for name,df in [('equity',eq),('trades',trades),('diagnostics',diag),('holdings',holdings)]:shun.atomic_text(args.out/(name+'.csv'),df.to_csv(index=False))
    title='架空データ・動作確認のみ' if args.demo else '実データ・現行マスター固定の参考検証'
    body=f'# バックテスト：{title}\n\n'+shun.md_table(pd.DataFrame([result]),{'from_date':'開始','to_date':'終了','initial_capital':'初期資金','final_equity':'最終評価額','return_pct':'騰落率%','max_drawdown_pct':'最大DD%','sell_fills':'売却約定数','fees':'手数料'})
    body+='\n期限撤退・失速は翌営業日始値で全売却。過熱は翌営業日始値で半数（100株単位切下げ）、過熱継続中は再実行しない。100株保有なら半数売却なし。逆指値は別途日中安値で判定し、窓開けを反映。\n\n現行マスターを全期間に固定しており、生存者バイアスがあります。配当・税金・板の再現なし。値幅制限接触日の該当方向の約定は保守的に見送る。最終日は保有を時価評価し強制売却しない。売却勝率は部分売却を含む約定単位で、往復取引単位ではありません。テーマは使わず33業種のみ。未検証の過去へのテーマ当てはめは行いません。\n'
    shun.atomic_text(args.out/'backtest_report.md',body);print(body)

if __name__=='__main__':main()
