from bs4 import BeautifulSoup
import urllib.request as req

url = "https://news.naver.com/main/read.nhn?mode=LS2D&mid=shm&sid1=101&sid2=259&oid=119&aid=0002356448"

html = req.urlopen(url)
source = html.read()
html.close()

bs = BeautifulSoup(source, "html.parser")
head = bs.select_one("head")
print(head)
[s.extract() for s in head.find_all("script")]
[s.extract() for s in head.find_all("script")]