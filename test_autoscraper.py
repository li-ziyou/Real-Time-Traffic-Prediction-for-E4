from autoscraper import AutoScraper
url = 'https://www.tomtom.com/traffic-index/stockholm-traffic/'
wanted_list = ['Wednesday, 28 Dec 2022', '8:00 PM', '12%']
scraper = AutoScraper()
res = scraper.build(url, wanted_list)
print(res)
