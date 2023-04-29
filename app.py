import streamlit as st
import transformers
import torch
import torchmetrics
import lightning.pytorch as pl
from PIL import Image



class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs)


class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, predict_path):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.predict_path = predict_path
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=160)
    def tokenizing(self, input):
        data = []
    # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
        text = '[SEP]'.join(input)
        outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
        data.append(outputs['input_ids'])
        return data

    def preprocessing(self, data):
        targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self,stage=None):
        
        # predict_data = 리스트 형태의 두 문장
        predict_inputs, predict_targets = self.preprocessing(predict_data)
        self.predict_dataset = Dataset(predict_inputs, [])

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)



class Model(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1)
        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        self.loss_func = torch.nn.L1Loss()

    def forward(self, x):
        x = self.plm(x)['logits']

        return x





st.title('문맥적 유사도 측정')
st.subheader('Semantic textual similarity(STS)')
st.write('by 이준범')
image = Image.open('STS_캡처.PNG')
st.image(image)
st.markdown('''문맥적 유사도(STS)란 두 텍스트가 얼마나 유사한지 판단하는 NLP Task입니다. 
일반적으로 두 개의 문장을 입력하고, 이러한 문장쌍이 얼마나 의미적으로 서로 유사한지를 판단합니다.''')

st.markdown('0점이면 전혀 관련없고 5점이면 의미가 같은 문장입니다.')

sentence1 = st.text_input('첫번째 문장을 입력하세요.')


sentence2 = st.text_input('두번째 문장을 입력하세요.')

if sentence1 and sentence2:
    predict_data = [sentence1, sentence2]

    dataloader = Dataloader('klue/roberta-small', 1, False, predict_path = predict_data)


    trainer = pl.Trainer()

    model = torch.load('C:/Users/LG/Desktop/이준범/네이버 부스트캠프/8주차/미션/model.pt')
    predictions = trainer.predict(model=model, datamodule=dataloader)

    answer = float(predictions[0].squeeze())
    answer = round(answer, 3)

    st.subheader(answer)
