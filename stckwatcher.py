from collections import OrderedDict
from pandas_datareader import data
from datetime import datetime, timedelta
from tslearn.clustering import TimeSeriesKMeans as tsm
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import praw
import pandas as pd
import numpy as np
import urllib.request as rq
import csv
import io
import matplotlib.pyplot as plt


class StkHelper():

    wrdCounter = {} # list of tickers found in scrape
    commonCounter = {} # list of common words that happen to be also tickers
    stklist = [] # list of stocks already in data
    postnum = 50 # number of top posts to scrape
    ranknum = 20 # number of stks to rank
    time = 'day' # time_filter for praw top posts('all','hour','day','week','month','year')
    now = datetime.today()-timedelta(hours=14) # time based on UTC

    def __init__(self):
        print(f"app started, right now it is {self.now}")
        self.create_csv()
        with open('repos.csv', newline='') as f:
            reader = csv.reader(f)
            stktemp = list(reader)
            for num in range(len(stktemp)):
                self.stklist.append(stktemp[num][0])
        response = rq.urlopen("https://dumbstockapi.com/stock?format=csv&countries=US")
        datareader = csv.reader(io.TextIOWrapper(response))
        wrdlist1 = []
        for row in datareader:
            wrdlist1.append(row[0])
        # Tickers that happen to be commonly used words
        commonList = {"TOO","PE","S","DO","IMO","HOPE","A", "IS", "FOR", "ARE", "ALL", "IT", "ON", "AT", "CAN", "BE", "GO", "OR", "AM", 
            "AN", "SO", "NEXT", "HE", "LOVE", "ONE", "OUT", "BIG", "NOW", "HAS", "E", "BY", "OPEN", "VERY", 
            "MAN", "TV", "SEE", "CEO", "U", "NEW", "ANY", "F", "UK", "D", "O", "R", "FREE", "LIFE", "ASS",
            "DD","RH","I","AI","USA","ticker"}
        self.wrdCounter = dict((stk,[0,0,0]) for stk in wrdlist1)
        self.commonCounter = dict((stk,[0,0,0]) for stk in commonList)
        self.scrape()
        print("finished collecting...")
        sort_dict = OrderedDict(sorted(self.wrdCounter.items(),key=lambda x: x[1][0]+x[1][1], reverse=True))
        sort_dict1 = OrderedDict(sorted(self.commonCounter.items(),key=lambda x: x[1][0]+x[1][1], reverse=True))
        penny = list(sort_dict.keys())
        self.update(penny, sort_dict, self.stklist)

    def scrape(self, subreddits = ['pennystocks']): # Scraping through word found in reddit
        reddit = praw.Reddit(client_id='z1eHTSoHlmqgkw', \
                     client_secret='Y3g69V7w6_drdPkfiFenPWb6azh2tQ', \
                     user_agent='stkRadar', \
                     username='Great-Practice3637', \
                     password='satrhdqn19')
        for sub in subreddits:
            subreddit = reddit.subreddit(sub)
            topPosts = subreddit.top(time_filter=self.time,limit=self.postnum)
            for post in topPosts:
                print("working on post:",post.title)
                for word in post.title.split():
                    self.cIncrement(word, 0)
                for word in post.selftext.split():
                    self.cIncrement(word, 0)
                print("working on comments")
                post.comments.replace_more(limit=None)
                comment_queue = post.comments[:]
                while comment_queue:
                    comment = comment_queue.pop(0)
                    for word in comment.body.split():
                        self.cIncrement(word, 1)
                    comment_queue.extend(comment.replies)

    def cIncrement(self, word, itype): # Increments hit count
        if word in self.commonCounter.keys():
            self.commonCounter[word][itype] += 1
        elif word in self.wrdCounter.keys():
            self.wrdCounter[word][itype] += 1
            print(word, " has been found ", self.wrdCounter[word])
    
    def update(self, penny, sort_dict, stklist): # Prints Ranked Stock Info        
        with open('repos.csv', mode='a+', newline='') as f:
            writer = csv.writer(f)
            rank = 0
            for stk in penny:
                print("currently",stk)
                try:
                    stki = data.DataReader(stk, 
                        start= datetime.today()-timedelta(days=21,hours=14),
                        end= self.now, 
                        data_source='yahoo')
                except:
                    stki = data.DataReader(stk, 
                        start= datetime.today()-timedelta(days=21,hours=14),
                        end= self.now, 
                        data_source='yahoo')
                                            
                if stki['Adj Close'].iloc[-1] < 1 :
                    rank +=1
                    self.update_stk(stk, stklist, writer, rank)
                    self.assess(stki)
                    print("______________________")
                    print(f"{stk}:{sort_dict[stk]}\t real penny")
                    # print(stki)
                elif stki['Adj Close'].iloc[-1] < 5 :
                    rank +=1
                    self.update_stk(stk, stklist, writer, rank)
                    self.assess(stki)
                    print("______________________")
                    print(f"{stk}:{sort_dict[stk]}\t lincoln")   
                    # print(stki)
                else :
                    print("______________________")
                    print(stk,"is over $5:",sort_dict[stk])
                if rank == self.ranknum:
                    break
                
    def create_csv(self): # create ticker list if it doesn't exist
        
        csv_file = "repos.csv"
        csv_columns = ['Ticker']
        try:
            csvfile = open(csv_file, 'r+')
        except:
            with open(csv_file, 'w+') as csvfile:
                writer = csv.DictWriter(csvfile,fieldnames=csv_columns)

    def update_stk(self, stk, stklist, writer, rank=0): # update stock's csv
        
        if stk not in stklist: # stk's missing csv made 
            print("++++++++")
            writer.writerow([stk])
            stki = data.DataReader(stk, 
                start= datetime.today()-timedelta(days=90,hours=14),
                end= self.now, 
                data_source='yahoo')
            stki['short_avg'] = stki['Adj Close'].rolling(window=3, min_periods=1, center=False).mean()
            stki['long_avg'] = stki['Adj Close'].rolling(window=10, min_periods=1, center=False).mean()
            stki['rank'] = 0
            stki['rank'].iloc[-1] = rank
            open('stks/'+stk+'.csv', mode='w+')
            stki.to_csv('stks/'+stk+'.csv')

        elif datetime.today().weekday() < 5: # weekday update(updates the tail only)
            stki = pd.read_csv('stks/'+stk+'.csv', index_col='Date')
            stkt = data.DataReader(stk,
                start= datetime.today()-timedelta(days=22),
                end= self.now,
                data_source='yahoo')
            sa = stkt['Adj Close'].rolling(window=5, min_periods=1, center=False).mean()[-1]
            la = stkt['Adj Close'].rolling(window=20, min_periods=1, center=False).mean()[-1]
            ac = stkt.iloc[[-1]]
            ac.index.array[-1] = pd.Timestamp(self.now.date())
            ac['short_avg'] = sa
            ac['long_avg'] = la
            ac['rank'] = rank
            print("last row /n",ac)
            if stki.index.array[-1] == str(self.now.date()):
                print("---------")
                stki.iloc[[-1]] = ac.iloc[[-1]]
            else:
                print("==========")
                stki = stki.append(ac, ignore_index=False)
                stki.index.array[-1] = str(self.now.date())
                stki.to_csv('stks/'+stk+'.csv',date_format='%Y-%m-%d')

        else: # weekend's rank update only. Friday's rank is updated by averaging ranks of weekend and friday's rank with heavier weight on later days.
            stki = pd.read_csv('stks/'+stk+'.csv', index_col='Date')
            stki.iloc[-1, stki.columns.get_loc('rank')] = (stki.iloc[-1, stki.columns.get_loc('rank')]+rank*1.5)/2.5
            stki.to_csv('stks/'+stk+'.csv')

    def graphit(self, stk, stki):
        stki = pd.read_csv('stks/'+stk+'.csv',index_col='Date')
        # Create the plot
        fig = plt.figure(figsize=(13, 10))

        # Labels for plot
        ax1 = fig.add_subplot(111, ylabel='Price in $')

        # Plot stock price over time
        stki['Close'].plot(ax=ax1, color='black', lw=2.)

        # Plot the the short and long moving averages
        stki[['short_avg', 'long_avg']].plot(ax=ax1, lw=2.)

        # Show the plot
        plt.show()

    def assess(self, stki):
        type = ['km.json','dba_km.json','sdtw_km.json']
        asmnt = ['before the pump.','near the dump.','after the dump.']
        gbw = list('gbw')
        rating = [0,0,0]
        srating = [0,0,0]
        data = stki['Close']
        data = data.to_numpy()
        data = data.reshape(-1,1)
        data = data[np.newaxis,...]
        for num in range(3): # no scaler data processing
            for num1 in range(3):
                model = tsm.from_json('MLModels/'+gbw[num]+type[num1])
                bits = model.transform(data)
                bits = np.sum(bits)
                rating[num] = rating[num] + bits
        data = TimeSeriesScalerMeanVariance(1,.5).fit_transform(data)
        for num in range(3): # scaler data processing
            for num1 in range(3):
                model = tsm.from_json('MLModels/s'+gbw[num]+type[num1] )
                bits = model.transform(data)
                bits = np.sum(bits)
                srating[num] = srating[num] + bits
        print("This stock is possibly "+asmnt[srating.index(min(srating))])

if __name__ == '__main__':
    bang = StkHelper()