import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# 設定
CSV_FILE   = 'data/SPY_intraday_data.csv'
START_DATE = '2025-01-02'
END_DATE   = '2025-07-31'


def prepare_15min_data(csv_file: str) -> pd.DataFrame:
    # 1) 讀取 1 分鐘資料
    df1 = (
        pd.read_csv(csv_file, parse_dates=['caldt'])
          .rename(columns={
              'caldt':  'Date',
              'open':   'Open',
              'high':   'High',
              'low':    'Low',
              'close':  'Close',
              'volume': 'Volume'
          })
          .set_index('Date')
          .sort_index()
    )
    # 2) 重採樣到 15 分鐘
    df15 = df1.resample('15min').agg({
        'Open':   'first',
        'High':   'max',
        'Low':    'min',
        'Close':  'last',
        'Volume': 'sum'
    })
    df15.dropna(subset=['Open','High','Low','Close'], inplace=True)

    # 3) 計算當日累積 VWAP（使用典型價）
    
    df15['Typical'] = (df15['High'] + df15['Low'] + df15['Close']) / 3
    df15['PV']      = df15['Typical'] * df15['Volume']
    df15['CumPV']   = df15.groupby(df15.index.date)['PV'].cumsum()
    df15['CumVol']  = df15.groupby(df15.index.date)['Volume'].cumsum()
    df15['VWAP']    = df15['CumPV'] / df15['CumVol']
    df15.drop(columns=['Typical','PV','CumPV','CumVol'], inplace=True)
    

    # 4) 計算當日累積 VWAC
    '''
    df15['PV']    = df15['Close'] * df15['Volume']
    df15['CumPV'] = df15.groupby(df15.index.date)['PV'].cumsum()
    df15['CumVol'] = df15.groupby(df15.index.date)['Volume'].cumsum()
    df15['VWAP']  = df15['CumPV'] / df15['CumVol']
    df15.drop(columns=['PV','CumPV','CumVol'], inplace=True)
    '''


    return df15


class VWAPShortStrategy(Strategy):
    def init(self):
        # 註冊 VWAP 指標
        self.vwap = self.I(lambda x: x, self.data.VWAP, name='VWAP')

    def next(self):
        price = self.data.Close[-1]

        # 若 VWAP 尚未計算完成，跳過
        if pd.isna(self.vwap[-1]):
            return

        # 空單出場：已有空單且價格突破 VWAP
        if self.position.is_short and crossover(self.data.Close, self.vwap):
            self.position.close()
            return

        # 做空進場：無持倉且價格跌破 VWAP
        if not self.position and crossover(self.vwap, self.data.Close):
            self.sell()


if __name__ == '__main__':
    # 資料準備與切片
    df15 = prepare_15min_data(CSV_FILE).loc[START_DATE:END_DATE]
    data = df15[['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']]

    # 回測設定：100k 初始資金、0 手續費、單一持倉、自動平倉計算
    bt = Backtest(data,
                  VWAPShortStrategy,
                  cash=100_000,
                  commission=0.0,
                  exclusive_orders=True,
                  finalize_trades=True)

    # 執行回測並輸出統計
    stats = bt.run()
    print(stats)

    # 繪製績效圖，顯示 VWAP 線
    bt.plot()
