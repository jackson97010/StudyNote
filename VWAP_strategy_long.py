import pandas as pd
from datetime import timedelta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# 設定
CSV_FILE   = 'data/TSM_intraday_data.csv'
START_DATE = '2025-01-02'
END_DATE   = '2025-07-31'
ATR_PERIOD = 14  # ATR 計算視窗


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

    # 3) 計算當日累積 VWAP

    df15['PV']    = df15['Close'] * df15['Volume']
    df15['CumPV'] = df15.groupby(df15.index.date)['PV'].cumsum()
    df15['CumVol'] = df15.groupby(df15.index.date)['Volume'].cumsum()
    df15['VWAP']  = df15['CumPV'] / df15['CumVol']

    # 4) 計算 True Range 和 ATR (每日獨立計算)
    df15['PrevClose'] = df15['Close'].shift(1)
    tr1 = df15['High'] - df15['Low']
    tr2 = (df15['High'] - df15['PrevClose']).abs()
    tr3 = (df15['Low'] - df15['PrevClose']).abs()
    df15['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df15['ATR'] = df15.groupby(df15.index.date)['TR']\
                         .transform(lambda x: x.rolling(window=ATR_PERIOD, min_periods=1).mean())
    df15.drop(columns=['PV','CumPV','CumVol','PrevClose','TR'], inplace=True)

    return df15


class VWAPStrategy(Strategy):
    def init(self):
        # 註冊 VWAP 與 ATR 指標
        self.vwap = self.I(lambda x: x, self.data.VWAP, name='VWAP')
        self.atr  = self.I(lambda x: x, self.data.ATR,  name='ATR')
        # 初始化停損與停利值
        self.stop_loss   = None
        self.take_profit = None

    def next(self):
        price = self.data.Close[-1]
        atr   = self.atr[-1]

        # 無效 ATR 或 VWAP 時跳過
        if pd.isna(atr) or pd.isna(self.vwap[-1]):
            return

        # 停損或停利檢查：若命中則平倉
        if self.position:
            if price <= self.stop_loss or price >= self.take_profit:
                self.position.close()
                return
            # VWAP 跌破時做為次要出場
            if crossover(self.vwap, self.data.Close):
                self.position.close()
                return

        # 多單進場：無持倉且價格突破 VWAP
        if not self.position and crossover(self.data.Close, self.vwap):
            # 計算停損與停利價格水平
            self.stop_loss   = price - atr
            self.take_profit = price + 1.5 * atr
            self.buy()


if __name__ == '__main__':
    # 資料準備與切片
    df15 = prepare_15min_data(CSV_FILE).loc[START_DATE:END_DATE]
    data = df15[['Open','High','Low','Close','Volume','VWAP','ATR']]

    # 回測設定：100k 初始資金、0 手續費、單一持倉、自動平倉計算
    bt = Backtest(data,
                  VWAPStrategy,
                  cash=100_000,
                  commission=0.0,
                  exclusive_orders=True,
                  finalize_trades=True)

    # 執行回測並輸出統計
    stats = bt.run()
    print(stats)

    # 繪製績效圖，顯示 VWAP 與 ATR 線
    bt.plot()
