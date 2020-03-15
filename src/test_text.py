import numpy as np
from joblib import load

from src import tftoidf as tfidf

# 텍스트 준비하기 --- ( ※ 1)
text1 = """
대통령은 이반 방한일정에 맞추어서
"""
text2 = """
이란 美 보복공격에 코스피 1%·코스닥 3%대 급락
"""
text3 = """
그 밖의 지역에서는 미세먼지가 좋음을 보이겠습니다.

이번 주에는 미세먼지가 많을 것으로 예상되므로 노약자는 외출을 자제하는 것이 좋습니다.
"""

# TF-IDF 사전 읽어 들이기 --- (*2)
tfidf.load_dic("/text/genre-tdidf.dic")


# 텍스트 지정해서 판별하기 --- (*4)
def check_genre(text):
    # 레이블 정의하기
    LABELS = ["정치", "경제", "날씨"]
    # TF-IDF 벡터로 변환하기 -- (*5)
    data = tfidf.calc_text(text)
    # MLP로 예측하기 --- (*6)
    model = load('/text/text-sklearn.model')
    pre = model.predict(np.array([data]))[0]
    n = pre.argmax()
    print(LABELS[pre], "(", pre, ")")
    return LABELS[pre], float(pre), int(n)

if __name__ == '__main__':
    check_genre(text1)
    check_genre(text2)
    check_genre(text3)