# STS Model 데모 웹 만들기

## STS Model
> Semantic Text Similarity (STS) Task
> - 두 문장의 의미적 유사도를 점수로 나타내기(0 ~ 5)
## 모델명과 사용 데이터셋
- 모델명: KLUE Roberta small
- 데이터셋: 두 문장과 그들의 유사도를 가지고 있는 데이터셋을 활용하였고 Train: 9324개, Valid: 550개, Test: 1100개의 문장 데이터

## 데모 웹
- 위 데이터셋으로 학습시킨 모델을 가져와 사용자가 한국어 문장을 입력하면 의미 유사도를 점수로 만드는 웹사이트를 만들었다. 
- 사용한 툴 : Streamlit

## 소개
https://user-images.githubusercontent.com/99182998/235419407-ed14b6a4-2107-4df4-8723-736916ab8a2b.mp4

## 주의사항
- Model.pt 파일이 용량이 커서 깃헙 저장소에는 담지 못했다. 


