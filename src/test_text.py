import numpy as np
from joblib import load

from src import tftoidf as tfidf

# 텍스트 준비하기 --- ( ※ 1)
text1 = """
여당은 정 총리의 모두발언에 대하여
"""
text2 = """
이란 美 보복공격에 코스피 1%·코스닥 3%대 급락
"""
text3 = """
그 밖의 지역에서는 미세먼지가 좋음을 보이겠습니다.
"""

# TF-IDF 사전 읽어 들이기 --- (*2)
tfidf.load_dic("/app/text/genre-tdidf.dic") #heroku
#tfidf.load_dic("../text/genre-tdidf.dic") #local- __main__
#tfidf.load_dic("./text/genre-tdidf.dic") #local- flask

# 텍스트 지정해서 판별하기 --- (*4)
def check_genre(text):
    # 레이블 정의하기
    LABELS = ["정치", "경제", "날씨"]
    # TF-IDF 벡터로 변환하기 -- (*5)
    data = tfidf.calc_text(text)
    # MLP로 예측하기 --- (*6)
    model = load('/app/text/text-sklearn.model') #heroku
    #model = load('../text/text-sklearn.model') #local- __main__
    #model = load('./text/text-sklearn.model')  # local- flask
    pre = model.predict(np.array([data]))[0]
    n = pre.argmax()
    print(LABELS[pre], "(", pre, ")")
    return LABELS[pre], float(pre), int(n)

if __name__ == '__main__':
    check_genre(text1)
    check_genre(text2)
    check_genre(text3)