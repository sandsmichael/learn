import math
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels import regression
import pandas as pd
import numpy as np


class Equity:

    def __init__(self, SECURITY = None, BENCHMARK = None, START_DATE= None, END_DATE = None, RETURN_COL = None) -> None:
        
        self.SECURITY = SECURITY
        self.BENCHMARK = BENCHMARK
        self.START_DATE = START_DATE
        self.END_DATE = END_DATE
        self.RETURN_COL = RETURN_COL

        self.get_prices()


    def get_prices(self):
        self.prices =  yf.download([self.SECURITY, self.BENCHMARK], start=self.START_DATE, end=self.END_DATE, progress = False)
        return self.prices


    def get_returns(self):
        self.returns =  self.get_prices()[self.RETURN_COL].pct_change().dropna(how = 'all', axis=0)
        return self.returns


    def get_cumulative_returns(self):
        self.cum_ret = self.get_returns().cumsum()        
        return self.cum_ret


    def get_volitility(self): # FIXME
        daily = self.get_returns().std()
        monthly = math.sqrt(21) * daily # Assume 21 trading days in a month
        annual = math.sqrt(252) * daily # Assume 252 trading days in a year
        return daily, monthly, annual
    

    def get_rolling_volitlity(self, window = 5):
        return self.prices[self.RETURN_COL].rolling(window).std()


    def get_cumulative_volitlity(self, window = 5):
        return self.get_rolling_volitlity(window = window).cumsum()


    def get_beta(self):
        returns = self.get_returns()

        X = returns[self.BENCHMARK].values
        Y = returns[self.SECURITY].values

        def linreg(x,y):
            x = sm.add_constant(x)
            model = regression.linear_model.OLS(y,x).fit()
            x = x[:, 1]
            return model.params[0], model.params[1]

        alpha, beta = linreg(X,Y)
        return alpha, beta, X, Y


    def get_rolling_beta(self, window=5):
        returns = self.get_returns()
        return (returns.rolling(window).cov().unstack()[self.BENCHMARK][self.SECURITY] / returns[self.BENCHMARK].rolling(window).var()).to_frame().reset_index().rename(columns = {0:'Rolling Beta'})


    def get_sharpe_ratio(self, y, window, risk_free_rate):
        mean_daily_return = sum(y) / len(y)
        s = y.std()
        daily_sharpe_ratio = (mean_daily_return - risk_free_rate) / s
        sharpe_ratio = 252**(window/252) * daily_sharpe_ratio     # annualized   
        return sharpe_ratio


    def get_rolling_sharpe_ratio(self, window, risk_free_rate):
        returns = self.get_returns()
        return returns.rolling(window).apply( lambda y : self.get_sharpe_ratio(y, window = window, risk_free_rate =risk_free_rate ))


    def famma_french():
        pass


    def get_drawdowns(self):
        prices = self.get_prices()[self.RETURN_COL].reset_index()
        xs = prices[self.SECURITY]
        i = np.argsort(np.maximum.accumulate(xs) - xs).iloc[-1]
        print(prices['Date'].iloc[i])
        j = np.argsort(xs[:i]).iloc[-1] 
        print(prices['Date'].iloc[j])
        plt.plot(xs)
        plt.plot([i, j], [xs[i], xs[j]], 'o', color='Red', markersize=10)
        plt.show()





class TearSheet:

    def __init__(self) -> None:
        pass

    @classmethod
    def cumulative_returns_chart(self, eqobj:Equity):
        df = eqobj.get_cumulative_returns().reset_index()
        melt = df.melt(id_vars = ['Date'], var_name = 'Ticker', value_name = 'Return')
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.lineplot(data = melt, x = 'Date', y = 'Return', hue = 'Ticker', ax = ax)
        ax.set_title('Cumulative Returns')
        ax.axhline(0, color = 'black')
        plt.show()


    @classmethod
    def daily_returns_chart(self, eqobj:Equity):
        df = eqobj.get_returns().reset_index()
        melt = df.melt(id_vars = ['Date'], var_name = 'Ticker', value_name = 'Return')
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.barplot(data = melt, x = 'Date', y = 'Return', hue = 'Ticker', ax = ax)
        ax.set_title('Cumulative Returns')
        ax.axhline(0, color = 'black')
        plt.show()


    @classmethod
    def rolling_std_dev_chart(self, eqobj:Equity):
        rolling_std = eqobj.get_rolling_volitlity(window = 20).reset_index()
        melt = rolling_std.melt(id_vars = ['Date'], var_name = 'Ticker', value_name = 'Rolling Std Dev')
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.lineplot(data = melt, x = 'Date', y = 'Rolling Std Dev', hue = 'Ticker', ax = ax)
        ax.set_title('Rolling Standard Deviation')
        plt.show()


    @classmethod
    def cumulative_std_dev_chart(self, eqobj:Equity):
        rolling_std = eqobj.get_cumulative_volitlity(window = 20).reset_index()
        melt = rolling_std.melt(id_vars = ['Date'], var_name = 'Ticker', value_name = 'Cumulative Std Dev')
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.lineplot(data = melt, x = 'Date', y = 'Cumulative Std Dev', hue = 'Ticker', ax = ax)
        ax.set_title('Cumulative Standard Deviation')
        plt.show()


    @classmethod
    def beta_chart(self, eqobj):
        alpha, beta, X, Y = eqobj.get_beta()
        X2 = np.linspace(X.min(), X.max(), 100)
        Y_hat = X2 * beta + alpha
        fig, ax = plt.subplots(figsize=(15, 10))
        plt.scatter(X, Y, alpha=0.3) # Plot the raw data
        plt.xlabel(f"{eqobj.BENCHMARK} Daily Return")
        plt.ylabel(f"{eqobj.SECURITY} Daily Return")
        plt.plot(X2, Y_hat, 'r', alpha=0.9)
        plt.show()
        print(alpha, beta)
        # show alpha, beta, and line slope formula on chart TODO


    @classmethod
    def rolling_beta_chart(self, eqobj):
        dataVeryShort = eqobj.get_rolling_beta(window = 20)
        dataShort = eqobj.get_rolling_beta(window = 60)
        dataLong = eqobj.get_rolling_beta(window = 120)
        print(dataShort)

        fig, ax = plt.subplots(figsize=(15,10))
        sns.lineplot(data = dataVeryShort, x = 'Date', y = 'Rolling Beta',  label = f'Beta 20 days', ax = ax)
        sns.lineplot(data = dataShort, x = 'Date', y = 'Rolling Beta',  label = f'Beta 60 days', ax = ax)
        sns.lineplot(data = dataLong, x = 'Date', y = 'Rolling Beta',  label = f'Beta 120 days', ax = ax)
        ax.axhline(0, color = 'black')
        plt.legend()
        plt.show()

    @classmethod
    def rolling_sharpe_ratio_chart(self, eqobj):
        window = 126
        data = eqobj.get_rolling_sharpe_ratio(window = window, risk_free_rate =  0.0).reset_index() # # 21 days per month X 6 months = 126
        melt = data.melt(id_vars = ['Date'], var_name = 'Ticker', value_name = 'Rolling Sharpe')
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.lineplot(data = melt, x = 'Date', y = 'Rolling Sharpe', hue = 'Ticker', ax = ax)
        ax.set_title(f'Rolling Sharpe {window} days')
        plt.show()


    def drawdowns_chart(self, eqobj):
        eqobj.get_drawdowns()


    @classmethod
    def monthly_return_heatmap(self, eqobj):
        returns = eqobj.get_returns().reset_index()
        returns['Date'] = pd.to_datetime(returns['Date'], format = "%Y-%m-%d")
        returns.set_index('Date', inplace = True)
        grouped_returns = returns.groupby(pd.Grouper(freq='M')).sum()[eqobj.SECURITY].to_frame()
        grouped_returns['Year'] = grouped_returns.index.strftime('%Y')
        grouped_returns['Month'] = grouped_returns.index.strftime('%b')
        grouped_returns = grouped_returns.pivot('Year', 'Month', eqobj.SECURITY).fillna(0)
        grouped_returns = grouped_returns[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]
        grouped_returns *= 100                       
        print(grouped_returns)
        fig, ax = plt.subplots(figsize=(15,10))
        ax = sns.heatmap(grouped_returns, ax=ax, annot=True, center=0,
                        fmt="0.2f", linewidths=0.5, cmap = 'RdYlGn' )
        plt.show()



eqobj = Equity(SECURITY = 'UNH', BENCHMARK = 'SPY', START_DATE = "2021-12-31", END_DATE = "2022-12-30", RETURN_COL = 'Adj Close')
ts = TearSheet()
# ts.daily_returns_chart(eqobj)
# ts.cumulative_returns_chart(eqobj)
# ts.rolling_std_dev_chart(eqobj)
# ts.cumulative_std_dev_chart(eqobj)
# ts.beta_chart(eqobj)
# ts.rolling_beta_chart(eqobj)
# ts.rolling_sharpe_ratio_chart(eqobj)
# ts.drawdowns_chart(eqobj)
ts.monthly_return_heatmap(Equity(SECURITY = 'UNH', BENCHMARK = 'SPY', START_DATE = "2018-12-31", END_DATE = "2022-12-30", RETURN_COL = 'Adj Close'))
