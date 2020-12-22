import time
import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
import re
import string
import nest_asyncio
import twint
import os
nest_asyncio.apply()

#scraping twitter
def scraping_twitter():
    print('This job id,{}'.format(time.ctime()))
    c = twint.Config()
    c.Search ="عقار OR استثمار OR رسول OR منتجات OR مقاطعة OR اللقاح OR كورونا"
    #c.Format = "Tweet id: {id} | Tweet: {tweet}"   # what do we need as information? tweets and ?  
    c.Lang = "ar"
    c.Store_json= True
    c.Output = "active_words_ar.json"
    output = twint.run.Search(c)
    data = pd.read_json("active_words_ar.json",lines = True,encoding ='utf8')
    return data

if __name__ == '__main__':
    scheduler = BlockingScheduler()
    #intervalTrigger=IntervalTrigger(hours = 10,start_date ="2020-12-22" ,end_date ="2021-01-02") 
    intervalTrigger=IntervalTrigger(hours = 1)
    scheduler.add_job(scraping_twitter, intervalTrigger, id='This_job_id')
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass
