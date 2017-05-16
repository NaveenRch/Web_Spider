# This package will contain the spiders of your Scrapy project
#
# Please refer to the documentation for information on how to create and manage
# your spiders.
import scrapy
from scrapy import Spider
from scrapy.selector import Selector

from new_crawler.items import NewCrawlerItem

class newSpider(Spider):
	name = 'new_spide'
	allowed_domains = ['medicaltenders.com']

	start_urls = []
	x = 0
	while x < 121:
		start_urls.append('http://www.medicaltenders.com/search.php?total=123&off=' + str(x) + '&inc=y&global=1&region_name[]=EG&notice_type_new[]=1,2,3,7,10,11,16,9,4,8,5&sector=18&deadline=')
		x = x + 10

	def parse(self, response):
		items = []
		tenders = Selector(response).xpath('//div[@class="cent"]')[7]
		table = tenders.xpath('.//table')
		innertable = table.xpath('.//table')
		data = []
		for values in innertable.xpath('.//table'):
			data.append(values.xpath('normalize-space(.//td/text())').extract())

		i = 0
		while (i < len(data)-4):
			item = NewCrawlerItem()
			item['Tender_Notice_Type'] = data[i][0]
			i = i + 1
			item['Tender_Category'] = data[i][0]
			i = i + 1
			item['Description'] = data[i][0]
			i = i + 1
			item['Action_Deadline'] = data[i][0]
			i = i + 2
			items.append(item)

		return items