""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ 
                                                                                                                │
  │ # A monte carlo simulation on daily equity returns used to predict future price level
                           │
  │ # https://www.interviewqs.com/blog/intro-monte-carlo                                                             │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import seaborn as sns

class NormalDistribution:

    def __init__(self, data:pd.DataFrame(), col:str = None) -> None:
        
        self.data = data
        self.col = col
        self.number_of_trading_days = 252 # number of trading days in a year

        self.get_periods()
        self.get_growth()
        self.get_std_dev()
    
    def __str__(self):
        return self.serialize()

    def get_periods(self):
        self.n_days = (self.data.index[-1] - self.data.index[0]).days
        self.n_years = self.n_days / 365.0

    def get_growth(self):
        total_growth = (self.data[self.col][-1] / self.data[self.col][1])
        self.cagr = total_growth ** (1/self.n_years) - 1 # mean annual growth rate; raise the total growth to the inverse of the # of years (e.g. ~1/10) to annualize our growth rate

    def get_std_dev(self):
        # calculate the standard deviation of the daily price changes
        self.std_dev = self.data[self.col].pct_change().std()
        self.std_dev = self.std_dev * math.sqrt(self.number_of_trading_days) # scale std_dev by an annualization factor reference: https://www.fool.com/knowledge-center/how-to-calculate-annualized-volatility.aspx

    def serialize(self):
        return 'NormalDistributionObject: ' + str({
            'cagr':self.cagr,
            'std_dev':self.std_dev,
            'days':self.n_days,
        })


    @staticmethod
    def distribution_over_time(ax):
        data1 =  yf.download("AMZN", start="2020-01-01", end="2020-12-31")
        data2 =  yf.download("AMZN", start="2021-01-01", end="2021-12-31")
        data3 =  yf.download("AMZN", start="2022-01-01", end="2022-12-25")
        x = data1['Adj Close'].to_frame().pct_change().rename(columns = {'Adj Close':'2020'}).merge(
            data2['Adj Close'].to_frame().pct_change().rename(columns = {'Adj Close':'2021'}), how = 'outer', left_index = True, right_index = True
        ).merge(
            data3['Adj Close'].to_frame().pct_change().rename(columns = {'Adj Close':'2022'}), how = 'outer', left_index = True, right_index = True
        )
        print(x)
        for c in x.columns:
            sns.distplot(x[c], hist = False, kde = True, 
                            kde_kws = {'linewidth': 3},
                            label = c, ax = ax)
        # Plot formatting
        ax.legend(prop={'size': 16}, title = 'Year')
        plt.title('Density Plot')
        ax.set_xlabel('Percent Change')
        ax.set_ylabel('Density')
        return ax

    @staticmethod
    def normal_distribution_plot(axs):
        # for understanding indput data being provided
        mu, sigma = dist.cagr/dist.number_of_trading_days, dist.std_dev/math.sqrt(dist.number_of_trading_days)
        s = np.random.normal(mu, sigma, len(dist.data[dist.col]))
        counts, bins, ignored = axs[0, 0].hist(dist.data[dist.col].pct_change(), density = True, bins = 20) # norm=True thus returns a histogram which can be interpreted as a probability distribution.
        axs[0, 0].plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ),       linewidth=3, color='y')
        axs[0, 0].set_title(f'Normal Distribution', fontsize='15')
        axs[0, 0].set_xlabel('Daily Percent Change', fontsize='15')
        axs[0, 0].set_ylabel('Probability Density', fontsize='15')
        axs[0, 0].axvline(0, c='black')
        axs[0, 0].axvline(dist.cagr/dist.number_of_trading_days, c='orange')
        # Label the raw counts and the percentages below the x-axis
        # bin_centers = 0.5 * np.diff(bins) + bins[:-1]
        # for count, x in zip(counts, bin_centers):
        #     # Label the raw counts
        #     axs[0, 0].annotate('{:.0f}'.format(count), xy=(x, 0), xycoords=('data', 'axes fraction'),
        #         xytext=(0, 18), textcoords='offset points', va='top', ha='center')
        axs[0, 0].annotate(f'Mean: {round(mu,4)}', xy=(0.15, 0.99), xycoords=('data', 'axes fraction'),xytext=(0, 18), textcoords='offset points', va='top', ha='center')
        axs[0, 0].annotate(f'Std Dev: {round(dist.std_dev,4)}', xy=(0.15, 0.96), xycoords=('data', 'axes fraction'),xytext=(0, 18), textcoords='offset points', va='top', ha='center')

        # seaborn density plot
        # Density Plot and Histogram of all arrival delays
        # sns.distplot(dist.data[dist.col].pct_change(), hist=True, kde=True, 
        #             bins=20, color = 'darkblue', 
        #             hist_kws={'edgecolor':'black'},
        #             kde_kws={'linewidth': 4}, ax = axs[1, 0])


class MonteCarlo:

    def __init__(self) -> None:
        pass

    @classmethod
    def random_walk(self, dist:NormalDistribution, number_of_trials=100):
        #Generate random values for 1 year's worth of trading (252 days), using numpy and assuming a normal distribution with mean of dialy growth rate and standard dev of daily standard deviation of percent change in price
        #Now that we have created a random series of future daily return %s, we can simply apply these forward-looking to our last stock price in the window, effectively carrying forward  a price prediction for the next year
        closing_prices = []
        simulation_prices = []
        for i in range(number_of_trials):

            daily_return_percentages = np.random.normal(dist.cagr/dist.number_of_trading_days, dist.std_dev/math.sqrt(dist.number_of_trading_days), dist.number_of_trading_days) + 1 
            
            price_series = [dist.data[dist.col][-1]] # the last actual value provided in our input data
            
            for j in daily_return_percentages:
                price_series.append(price_series[-1] * j)
            
            simulation_prices.append(price_series)
        
            closing_prices.append(price_series[-1])

        self.closing_prices = closing_prices
        self.simulation_prices =  pd.DataFrame(simulation_prices).transpose()
        self.mean_end_price = round(np.mean(self.closing_prices),2)
        print("Expected price: ", str(self.mean_end_price))


    @classmethod
    def plot(self, dist:NormalDistribution):
        
        fig, axs = plt.subplots(3,2, figsize=(15, 10))
        fig.suptitle('Title')

        axs[0,0] = dist.normal_distribution_plot(axs)

        import pandas_market_calendars as mcal
        import datetime
        nyse = mcal.get_calendar('NYSE')

        # get historical prices (actuals)
        history = dist.data[[dist.col]].reset_index()
        history['Date'] = pd.to_datetime(history['Date']).dt.date
        history.set_index('Date', inplace = True)
        print(history)

        # construct a range of future dates to set as index for the simulated prices
        future = self.simulation_prices
        date_range = nyse.valid_days(start_date=dist.data.reset_index()['Date'].iloc[-1], end_date= '2030-01-01')[1:len(self.simulation_prices)+1]
        future['Date'] = date_range
        future['Date'] = future['Date'].dt.date
        print(future)
        future.set_index('Date', inplace = True)

        self.simulation_prices = history.merge(future, how = 'outer', left_index = True, right_index = True)

        print(self.simulation_prices) # market days from last date in 
        axs[0, 1].plot(self.simulation_prices)


        # #from here, we can check the mean of all ending prices allowing us to arrive at the most probable ending point
        axs[1, 1].axvline(self.mean_end_price,color='black', linewidth=2)
        #lastly, we can split the distribution into percentiles to help us gauge risk vs. reward
        #Pull top 10% of possible outcomes
        top_ten = np.percentile(self.closing_prices,100-10)
        #Pull bottom 10% of possible outcomes
        bottom_ten = np.percentile(self.closing_prices,10)
        #create histogram again
        axs[1, 1].hist(self.closing_prices,bins=40)
        #append w/ top 10% line
        axs[1, 1].axvline(top_ten,color='r',linestyle='dashed',linewidth=2)
        #append w/ bottom 10% line
        axs[1, 1].axvline(bottom_ten,color='r',linestyle='dashed',linewidth=2)
        #append with current price
        axs[1, 1].axvline(dist.data[dist.col][-1],color='g', linestyle='dashed',linewidth=2)


        axs[2,1] = dist.distribution_over_time(axs[2, 1])

        plt.show()






""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Run                                                                                                              │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
data =  yf.download("AMZN", start="2019-01-01", end="2022-12-25")
dist = NormalDistribution(data = data, col = 'Adj Close')
print(dist)

mc = MonteCarlo()
mc.random_walk(dist = dist)
mc.plot(dist = dist)

# print(pd.DataFrame(mc.closing_prices))
# print(mc.simulation_prices)
# simulation_description = mc.simulation_prices.describe()
# print(simulation_description)
# results_description = simulation_description.T[['mean', '50%']].describe()
# print(results_description)


""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Etc.                                                                                                             │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""

def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax.get_figure(), ax


# print data examples
# tmp_data = dist.data[dist.col].to_frame().reset_index().tail()
# tmp_data.Date = tmp_data.Date.apply(lambda x : x.strftime('%Y-%m-%d'))
# tmp_data['Adj Close'] = tmp_data['Adj Close'].round(2)
# fig,ax = render_mpl_table(tmp_data, header_columns=0, col_width=2.0)
# fig.savefig("table_mpl.png")


# daily_return_percentages = np.random.normal(dist.cagr/dist.number_of_trading_days, dist.std_dev/math.sqrt(dist.number_of_trading_days), dist.number_of_trading_days)+1
# print(daily_return_percentages)
# print(dist.cagr/dist.number_of_trading_days + 1)
# print(daily_return_percentages.mean())
# print(dist.std_dev/math.sqrt(dist.number_of_trading_days))
# print(daily_return_percentages.std())
# print(pd.DataFrame(mc.simulation_prices[-1]))




