import json
import bson



class SentAn(object):

    def analysis_word(self, word):
        with open('data/senti_word.json', encoding='utf-8', mode='r') as f:
            data = json.load(f)

        search_cnt = 0

        for i in data:
            if word in (i['word'], i['word_root']):
                print()
                print('어근 : ' + i['word'])
                print('극성 : ' + i['word_root'])
                print('점수 : ' + i['polarity'])
                print()
                search_cnt += 1

        if search_cnt == 0:
            print('검색 결과 없음')


if __name__ == "__main__":

    stan = SentAn()

    print()
    print("한국어 감성사전")
    print("종료 : 0")
    print("범위 : -2, -1, 0, 1, 2")
    print()

    while True:
        print()
        word_name = input("word : ")
        word_name = word_name.strip(" ")
        if word_name != 0:
            stan.analysis_word(word_name)

        elif word_name == 0:
            print()
            print("종료")
            break
