import matplotlib.pyplot as plt
import pandas as pd

stki = pd.read_csv('stks/ZSAN.csv',index_col='Date')
        # Create the plot

fig = plt.figure(figsize=(13, 10))

# Labels for plot
ax1 = fig.add_subplot(111, ylabel='Price in $')

# Plot stock price over time
stki['Close'].plot(ax=ax1, color='black', lw=1.)

# Plot the the short and long moving averages
stki[['short_avg', 'long_avg']].plot(ax=ax1, lw=1.)

# Show the plot
plt.show()
