# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class NewCrawlerItem(scrapy.Item):
    Tender_Notice_Type = scrapy.Field()
    Tender_Category = scrapy.Field()
    Description = scrapy.Field()
    Action_Deadline = scrapy.Field()