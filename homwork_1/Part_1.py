from urllib.parse import urlencode
from urllib.request import urlopen
from datetime import datetime, timezone, timedelta
from bs4 import BeautifulSoup
import json
import re
import os

log = open("./part1_log.txt", 'w')

def print_log(message):
    log_time = datetime.strftime(datetime.now(tz=timezone(timedelta(hours=8))), "%Y-%m-%d %H:%M:%S.%f")
    log_message = "[{}] {}".format(log_time, message)
    print(message)
    print(log_message, file=log)


class Crawler_Base():
    def __init__(self):
        self.time_format = "%Y.%m.%d"
        self.base_url = "https://search.naver.com/search.naver"
        self.nso_format = "so%3Add%2Cp%3Afrom{}to{}%2Ca%3Aall"
        self.query_dict = {
            "where": "news",
            "query": None,
            "sm": "tab_srt",
            "sort": 0,
            "photo": 0,
            "field": 0,
            "reporter_article": "",
            "pd": 3,
            "ds": None,
            "de": None,
            "docid": "",
            "nso": None,
            "mynews": 0,
            "start": 1,
            "refresh_start": 0,
            "related": 0
        }
        # clear cmd(terminal)
        os.system("cls")
        print_log("Crawler Start")

    def question_1(self, search_word):
        # input search word
        self.query_dict['query'] = search_word
        print_log("search word: {}".format(self.query_dict['query']))

        # input start & end date
        self.date_match(key="ds", desp="start")
        self.date_match(key="de", desp="end")
        self.query_dict['nso'] = self.nso_format.format(self.query_dict['ds'].replace(".", ""), self.query_dict['de'].replace(".", ""))

        # 첫번째 페이지에서 네이버 뉴스 포멧팅된 기사 링크 수집
        main = self.get_news()
        links_list = []
        links = main.find_all("a")
        for link in links:
            if link['href'].startswith("https://news.naver.com/main/read.nhn"):
                links_list.append(link['href'])
        print_log(json.dumps(links_list, ensure_ascii=False, indent='\t'))

        # get article count
        self.total_num = int(re.findall('\d+', main.select_one("div.section_head > div.title_desc.all_my > span").text.split('/')[1].replace(",", ""))[0])
        print_log("article total num: {}".format(self.total_num))

    def question_2(self):
        # 링크를 담을 list 선언
        links_list = []

        # 객체 내부에 있는 전체 기사수를 활용한 for문
        print_log("find naver news link start")

        for i in range(self.total_num // 10 + 1):
            # i 값으로 객체 내부의 dict 안에 start 값을 변경
            self.query_dict['start'] = 10 * i + 1
            if i != int(self.total_num // 10):
                num_range = "{}-{}".format(self.query_dict['start'], self.query_dict['start'] + 9)
            else:
                num_range = "{}-{}".format(self.query_dict['start'], self.total_num)
            print_log("find naver new link in {}".format(num_range))

            main = self.get_news()
            # a 태그 전부를 list 형태로 받은뒤 특정 url 형식으로 시작하는 href값을 list에 넣는다.
            links = main.find_all("a")
            for link in links:
                if link['href'].startswith("https://news.naver.com/main/read.nhn"):
                    links_list.append(link['href'])

        # 혹시 모를 url 중복을 제거 한다.
        self.naver_news_links = list(set(links_list))
        self.naver_news_num = len(self.naver_news_links)
        print_log("total naver news num: {}".format(self.naver_news_num))
        print_log(json.dumps(self.naver_news_links, ensure_ascii=False, indent='\t'))

    def question_3(self):
        news_list = []
        print_log("get title and text start")
        for i, link in enumerate(self.naver_news_links):
            print_log("working.....({}/{})".format(i + 0, self.naver_news_num))
            # html 가져오기
            html = urlopen(link)
            source = html.read()
            html.close()
            soup = BeautifulSoup(source, 'html.parser')

            # title 얻기
            title = soup.select_one("#articleTitle").text

            # text 얻기
            contents = soup.select_one("#articleBodyContents")
            # 불필요한 Tag(script / a) 정제 작업
            scripts = contents.find_all("script")
            for s in scripts:
                s.extract()
            a_tags = contents.find_all("a")
            for a in a_tags:
                a.extract()
            # list apppend
            news_list.append(
                {
                    "link": link,
                    "title": title,
                    "text": contents.text
                }
            )
        # title list & text list 만들기
        print_log("make title list")
        self.title_list = list(map(lambda news: news['title'], news_list))
        print_log(json.dumps(self.title_list, ensure_ascii=False, indent='\t'))

    def question_4(self):
        # 정규 표현식을 활용한 title 출력
        print_log("classify with re start")
        for title in self.title_list:
            p = re.compile(".*금리.*인하.*")
            if p.match(title) is not None:
                print_log(title)

    def get_news(self):
        # query_dict 값을 자동으로 쿼리로 인코딩하여 요청 수행 및 main 파트만 retrun 한다.
        url = "{}?{}".format(self.base_url, urlencode(self.query_dict))
        html = urlopen(url)
        source = html.read()
        html.close()

        soup = BeautifulSoup(source, 'html.parser')

        return soup.select_one("#main_pack > div.news.mynews.section._prs_nws")

    def date_match(self, key, desp):
        while True:
            self.query_dict[key] = input("please input {} date(yyyy.mm.dd): ".format(desp))

            if re.match('^[0-9]{4}.[0-9]{2}.[0-9]{2}$', self.query_dict[key]) is None:
                print('Invalid date string format, please enter again\n')
            else:
                try:
                    datetime.strptime(self.query_dict[key], self.time_format)
                    if key == "de":
                        tdel = datetime.strptime(self.query_dict[key], self.time_format) - datetime.strptime(self.query_dict["ds"], self.time_format)
                        if tdel.days < 0:
                            print("please enter date after start date\n")
                            continue
                    print_log("{} date: {}".format(desp, self.query_dict[key]))
                    break
                except ValueError:
                    print('Invalid date string, please enter again\n')


# crawler class 선언
naver_news = Crawler_Base()
# answer 1)
naver_news.question_1(search_word="이자율")
# answer 2)
naver_news.question_2()
# answer 3)
naver_news.question_3()
# answer 4)
naver_news.question_4()