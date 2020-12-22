# -*- coding: utf-8 -*-
import time
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
import nest_asyncio
import twint
import os
nest_asyncio.apply()


list_active_words1 = "عقار OR استثمار OR رسول OR منتجات OR مقاطعة OR اللقاح OR كورونا OR تطعيم"
list_active_words2 = "اضحكتني OR غاضب OR سعيد OR فرحان OR يضحك OR ابكي"



#scraping twitter
def scraping_twitter(list_active_words):
    print('This job id,{}'.format(time.ctime()))
    c = twint.Config()
    c.Search =list_active_words
    #c.Format = "Tweet id: {id} | Tweet: {tweet}  "   # what do we need as information? tweets and #?  
    c.Lang = "ar"
    c.Store_json= True
    c.Output = "active_words_ar.json"
    output = twint.run.Search(c)
    print("=== END ===")
    #data = pd.read_json("active_words_ar.json",lines = True,encoding ='utf8')
    #return data





# scheduling scraping 
if __name__ == '__main__':
    scheduler = BlockingScheduler()
    intervalTrigger=IntervalTrigger(start_date ="2020-12-22" ,end_date ="2020-12-23") # hours seconds ....
    scheduler.add_job(scraping_twitter,intervalTrigger, id='This_job_id',args=(list_active_words1,))
    scheduler.start()
    intervalTrigger=IntervalTrigger(start_date ="2020-12-23" ,end_date ="2020-12-24") # hours seconds ....
    scheduler.add_job(scraping_twitter,intervalTrigger, id='This_job_id',args=(list_active_words1,))
    scheduler.start()
    print('Press Ctrl+{0} to exit'.format('Break' if os.name == 'nt' else 'C'))
   
    try:
        while True:
            time.sleep(5)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()