# occluded-face-detection
얼굴에 가려진 부분이 있는 지 찾아내는 모델

<img src="https://github.com/lee-jingu/occluded-face-detection/blob/main/asset/output.gif" height="480">

[Notion](https://aiappbox.notion.site/Occluded-Face-Detection-bac15f3cfa024ef8bd0cab59726c487f) 에 가면 더욱 자세한 실험 설계와 EDA를 볼 수 있다.


## Environment

```
OS: ubuntu 20.04.4 LTS
Python: 3.8.10
```

## Requirements

```
opencv-python==4.6.0
padnas==1.4.3
matplotlib==3.5.2
sklearn==1.1.2

torch==1.12.0+cu113
torchvision==0.13.0+cu113
torchsummary==1.5.1
```

## Setup

```bash
git clone https://github.com/lee-jingu/occluded-face-detection.git

cd occluded-face-detection
pip3 install -r requirements.txt
```

## Run Demo

```bash
python3 occluded_face/demo.py --dir ${이미지_디렉토리} --verbose
```
