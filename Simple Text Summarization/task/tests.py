from test.tests import open

if __name__ == '__main__':    with open('news.xml', 'w', encoding='utf-8') as f:        f.write(news_text)    TextSummarization().run_tests()