#!/usr/bin/env python
# coding: utf-8

# In[448]:


import os
import re
import sys
import requests
from bs4 import BeautifulSoup
import urllib
from lxml import html, etree
from tqdm.notebook import tqdm
import csv
from itertools import cycle
import logging
import random


# In[482]:


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


# In[486]:


# scraping tools

user_agents = [
   #Chrome
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
    'Mozilla/5.0 (Windows NT 5.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
    #Firefox
    'Mozilla/4.0 (compatible; MSIE 9.0; Windows NT 6.1)',
    'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0)',
    'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (Windows NT 6.2; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.0; Trident/5.0)',
    'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0)',
    'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)',
    'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 5.1; Trident/4.0; .NET CLR 2.0.50727; .NET CLR 3.0.4506.2152; .NET CLR 3.5.30729)']

def get_proxies():
    """
    get proxies from proxy_url
    """
    PROXY_URL = 'https://free-proxy-list.net/'
    page = requests.get(PROXY_URL)
    tree = html.fromstring(page.content)
    proxies = set()
    for tr in tree.xpath("//table[@class='table table-striped table-bordered']/tbody/tr"):
        ip_add = tr.xpath('td/text()')[0].strip()
        port_num = tr.xpath('td/text()')[1].strip()
        if ip_add and port_num:
            proxy = ip_add + ':' + port_num
            proxies.add(proxy)
    return proxies


# In[495]:


# functions

def parse_rating(elem):
    """
    ratings are encoded as a span elements, e.g. <span class="ui_bubble_rating bubble_50"></span>
    where the last two digits represent the colour fill of the bubbles shown on TripAdvisor
    """
    rating = int(list(elem.classes)[-1][-2:]) # as integer [0-50]
    return rating // 10

def clean_text_block(text_block):
    """
    normalises whitespace by joining a list of strings separated by potentially broken line breaks 
    """
    # NOTE: line breaks are typically not well-formed (e.g. <br> instead of <br/>),
    # so get text of all descendendents as a list and then join as a string, normalising whitespace
    text = ' '.join(map(str.strip, text_block))
    text = re.sub('[\s\t]+', ' ', text)
    return text
    
def clean_date(string):
    """
    strips excess information from TripAdvisor date string e.g. `Reviewed 14 February 2017` --> `14 February 2017`
    """
    string = re.sub('(Reviewed|Responded)', '', string)
    return string.strip()

def parse_review_response_item(target):
    """
    target element of TripAdvisor review HTML page.
    e.g. div[@id='review_459559133'] from https://www.tripadvisor.com.au/ShowUserReviews-g2103653-d10607217-r459559133-Meriton_Suites_North_Sydney-North_Sydney_Greater_Sydney_New_South_Wales.html
    """
        
    # //*[@id="review_459559133"]/div/div[2]/span[2]
    try:
        review_date = clean_date(target.xpath("./div/div[2]/span[2]/descendant-or-self::text()")[0])
    except:
        review_date = 'N/A'
        
    # //*[@id="review_459559133"]/div/div[2]/div[10]/div[1]/div/span
    try:
        response_date = clean_date(target.xpath("./div/div[2]/div[10]/div[1]/div/span/descendant-or-self::text()")[0])
    except:
        response_date = 'N/A'
    
    # target elem xpath: //*[@id="HEADING"]
    title = target.xpath("//*[@id='HEADING'][@class='title']/descendant-or-self::text()")
    title = clean_text_block(title)

    # target elem xpath: //*[@id="review_459559133"]/div/div[2]/div[3]/div/p/span[1]
    review_msg = target.xpath("//div[@class='prw_rup prw_reviews_resp_sur_review_text']//span[@class='fullText ']/descendant-or-self::text()")
    review_msg = clean_text_block(review_msg)

    # target elem xpath: //*[@id="review_459559133"]/div/div[2]/div[10]/div[2]/div/p
    response_msg = target.xpath(".//div[@class='prw_rup prw_reviews_text_summary_hsx']/div/p/descendant-or-self::text()")
    response_msg = clean_text_block(response_msg)

    # target elem xpath: //*[@id="review_459559133"]/div/div[2]/span[1]
    rating = parse_rating(target.xpath("./div/div[2]/span[1]")[0])

    return (review_msg, response_msg, rating, review_date, response_date)

def scrape(url, proxies=None, user_agents=None, max_attempts=10, timeout=5):

    if proxies:
        proxy_pool = cycle(proxies)

    attempts = 0 
    
    while attempts < max_attempts:
        proxy = {"http": next(proxy_pool)} if proxies else None
        agent = random.choice(user_agents) if user_agents else None 
        headers = {'User-Agent': agent} if agent else None 
        
        try:
            logging.info('Using proxy {} and user agent {}'.format(proxy, agent))
            response = requests.get(url, headers=headers, proxies=proxy, timeout=timeout)
            return response
        except Exception as e:
            logging.info(f'[!] request failed at attempt {attempts} / {max_attempts}: {url} ({e})')
            response = None
        
        attempts += 1
    
    return response
    

def main():
    """
    process a single url to extract review response data content
    """
    url_file = sys.argv[1] 
    out_file = sys.argv[2]
    

    proxies = get_proxies()
    logging.info(f'Collected {len(proxies)} proxies')
    
    field_names = ['review', 'response', 'rating', 'review_date', 'response_date']
    
    c = 0
    with open(url_file, 'r', encoding='utf8') as url_f:
        with open(out_file, 'w', encoding='utf8') as csv_f:
            writer = csv.DictWriter(csv_f, delimiter='\t', fieldnames=field_names)
            writer.writeheader()
            
            for i, line in tqdm(enumerate(url_f)):
                url = line.strip()
                review_id = re.search('-r(\d+)-', url).group(1)
                page = scrape(url, proxies, user_agents)
                if page:
                    tree = html.fromstring(page.content)
                    try:
                        target_content = tree.xpath("//div[@id='review_{}']".format(review_id))[0]
                        items = parse_review_response_item(target_content)
                        writer.writerow(dict(zip(field_names, items)))
                        c += 1
                    except IndexError:
                        logging.info(f'[!] failed to find target content in html document: {url}. Is this a valid review-response page?')
                else:
                    logging.info(f'[!] failed to scrape {url}')
                    
    logging.info(f'Pages collected: {c}')
    return


# In[496]:


main()


# In[475]:


# print(len(proxies))
# url = 'https://www.tripadvisor.com.au/ShowUserReviews-g255340-d8450248-r470987141-Quest_Toowoomba-Toowoomba_Queensland.html'
# page = scrape(url, proxies=proxies, user_agents=user_agent_list, max_attempts=10, timeout=3)
# tree = html.fromstring(page.content)
# target_content = tree.xpath("//div[@id='review_{}']".format('470987141'))[0]
# items = parse_review_response_item(target_content)
# print(items)


# In[ ]:





# In[ ]:




