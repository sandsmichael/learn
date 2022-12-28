# https://www.interviewqs.com/blog/intro-monte-carlo
import math
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from scipy.stats import norm

data =  yf.download("JNJ", start="2021-01-01", end="2022-10-30")

data.head()

time_elapsed = (data.index[-1] - data.index[0]).days
print('days: ', time_elapsed)

number_of_years = time_elapsed / 365.0 # time in years
print('years: ', number_of_years)

# total growth over period
total_growth = (data['Adj Close'][-1] / data['Adj Close'][1])
print('total growth: ', total_growth)

#we can raise the total growth to the inverse of the # of years (e.g. ~1/10 at time of writing) to annualize our growth rate
cagr = total_growth ** (1/number_of_years) - 1    #mean annual growth rate
print('cagr: ', cagr)

#calculate the standard deviation of the aily price changes
std_dev = data['Adj Close'].pct_change().std()
number_of_trading_days = 252 #because there are roughy ~252 trading days in a year, we'll need to scale this by an annualization factor reference: https://www.fool.com/knowledge-center/how-to-calculate-annualized-volatility.aspx
std_dev = std_dev * math.sqrt(number_of_trading_days)
print('std_dev: ', std_dev)

#From here, we have our two inputs needed to generate random values in our simulation
print ("cagr (mean returns) : ", str(round(cagr,4)))
print ("std_dev (standard deviation of return : )", str(round(std_dev,4)))


# for understanding indput data being provided
print( data['Adj Close'].pct_change())
normal_distribution = norm(cagr, std_dev)
x = np.linspace(start =-1, stop = 1, num = 1000)
fig, ax = plt.subplots(1, 1)
ax.plot(x, normal_distribution.pdf(x), 'k-', lw=2, label='frozen pdf')
plt.show()
print(norm.ppf(0.95, loc=0, scale=1))


def simulate(data, col=None):

    number_of_trials = 100

    closing_prices = []
    for i in range(number_of_trials):
        #Generate random values for 1 year's worth of trading (252 days), using numpy and assuming a normal distribution with mean of dialy growth rate and standard dev of daily standard deviation of percent change in price
        daily_return_percentages = np.random.normal(cagr/number_of_trading_days, std_dev/math.sqrt(number_of_trading_days), number_of_trading_days)+1
        # print(daily_return_percentages)
        #Now that we have created a random series of future daily return %s, we can simply apply these forward-looking to our last stock price in the window, effectively carrying forward  a price prediction for the next year
        #This distribution is known as a 'random walk'
        price_series = [data[col][-1]] # the last actual value provided in our input data
        for j in daily_return_percentages:
            price_series.append(price_series[-1] * j)

        closing_prices.append(price_series[-1])


    plt.plot(price_series)
    plt.show()


    #plot histogram
    plt.hist(closing_prices,bins=40)
    plt.show()


    #from here, we can check the mean of all ending prices allowing us to arrive at the most probable ending point
    mean_end_price = round(np.mean(closing_prices),2)
    print("Expected price: ", str(mean_end_price))

    #lastly, we can split the distribution into percentiles to help us gauge risk vs. reward
    #Pull top 10% of possible outcomes
    top_ten = np.percentile(closing_prices,100-10)
    #Pull bottom 10% of possible outcomes
    bottom_ten = np.percentile(closing_prices,10);
    #create histogram again
    plt.hist(closing_prices,bins=40)
    #append w/ top 10% line
    plt.axvline(top_ten,color='r',linestyle='dashed',linewidth=2)
    #append w/ bottom 10% line
    plt.axvline(bottom_ten,color='r',linestyle='dashed',linewidth=2)
    #append with current price
    plt.axvline(data['Adj Close'][-1],color='g', linestyle='dashed',linewidth=2)
    plt.show()



simulate(data = data, col = 'Adj Close')




#TODO make matplotlib panel figure with histogram and simulation line plots