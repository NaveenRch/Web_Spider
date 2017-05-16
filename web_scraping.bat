echo Running web crawler
cd new_crawler 
del /f items.json
scrapy crawl new_spide  -o items.json -t json
copy "items.json" "..\Classification\items.json"

cd ..
cd Classification
python classifiers.py
pause Enter a key to continue