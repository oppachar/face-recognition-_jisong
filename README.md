# face-recognition-_jisong
구글 Teachable Machine 이용하여 모델링 하였습니돠,,, 
하지만 데이터 셋이 너무 작음,, ㅜㅡㅜ



labels를 보면 나와있듯 



0 --> 계란형 

1 --> 각진형 

2 --> 마름모형 ( 역삼각형 ) 

3 --> 둥근형  

4 --> 하트형 
 



# Colab
아직 정확도는 61.33% 라 수정 작업중입니다....

분류 모델 --> ResNet 34 사용 

epochs = 50, batch size =6
lr=0.001, momentum=0.9



-------------------------- 
 
 
transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),     #모델에 들어갈 사진 
    transforms.RandomHorizontalFlip(), # 데이터 좌우 전환 
    transforms.RandomChoice([
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),   # 데이터 명암, 조도, 등ㅇ등ㅇ 조절 
        transforms.RandomGrayscale(p=0.2),       # 그레이스케일
        transforms.RandomAffine(      # 각조조절
            degrees=10, translate=(0.2, 0.2), 
            scale=(0.8, 1.2), shear=15, resample=Image.BILINEAR)
    ]),
    transforms.ToTensor(), # PIL 형태의 이미지나 ndarray 를 PyTorch 가 이해할 수 있는 tensor 자료형으로 바꾸어 주는 역할
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 정규화(normalization)
])
