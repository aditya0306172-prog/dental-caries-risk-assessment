import urllib.request
import re
req = urllib.request.Request('https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Questionnaire&CycleBeginYear=2017')
html = urllib.request.urlopen(req).read().decode('utf-8')
links = re.findall(r'href=[\'"]([^\'"]*\.XPT)[\'"]', html, re.IGNORECASE)
for link in links:
    print(link)
