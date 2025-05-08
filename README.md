# Reverse-curve-codeswitching
# STEP 1: 필요한 라이브러리 설치
!pip install bert-score pandas

# STEP 2: 라이브러리 불러오기
import pandas as pd
from bert_score import score

# STEP 3: 사용자에게 CSV 파일 업로드 요청
from google.colab import files
uploaded = files.upload()

# STEP 4: 파일 읽기 (파일명 자동 인식)
import io
filename = next(iter(uploaded))
df = pd.read_csv(io.BytesIO(uploaded[filename]))

# STEP 5: 정답 문장과 모델 응답을 비교
# "문장A" = 프롬프트 / "문장B" = 모델 응답 (또는 정답 문장)

references = df["문장B"].tolist()
candidates = df["문장A"].tolist()

P, R, F1 = score(candidates, references, lang="en", verbose=True)

# STEP 6: 결과 저장
df["BERTScore_P"] = P
df["BERTScore_R"] = R
df["BERTScore_F1"] = F1

# STEP 7: 결과 미리보기 및 저장
df.head()
df.to_csv("bert_score_results.csv", index=False)
files.download("bert_score_results.csv")
