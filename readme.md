StkRadar
Based on popularity of certain stock in a subreddit, predict the profitability of the stock based on price history through machine learning.

Wildly covered news of Gamestop(GME)'s stock "skyrocketting", jumping from mere 7 to 8 dollars a share to $450 at its peak, people have gathered to find out what the next Gamestop will be. Even though it would be naive to think that Gamestop's case will repeat in the near future as the hedge funds will not be as bold like they were revealing what their next move is and it is more than easy to monitor the reddit enthusiasts and their investment. But I believe that there is a niche market where profit could be made, and they are pennystocks. With people doing their DD(Due Diligence) on the stocks they believe will soon rise in price and fellow traders jumping on board afterwhich hedge fund's hand participating to drop the price down to where it was in the beginning, pennystocks go through the so called Pump and Dump. The biggest caveat of investing in pennystocks after the stock going bankrupt, is getting on board when it is too late.

The aim of my program is to find these pennystocks at the right time, purchasing before its rise and then sell before it hits its peak. I will achieve this by doing the following:

Scraping the right subreddit(/r/pennystocks) for the next pennystock on the rise.
Building machine learning model that categorizes stocks based on the price history through dynamic time warping clustering.
Based on the train data, the stock's price history will be used to predict it's profitability.
Automate trading through stock trading API.
Currently, I am collecting the data necessary to train the machien learning model.

is it working?
