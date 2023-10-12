print(201927518 + "박동윤")import argparse
import torch
import torch.optim as optim
import torch.nn as nn

# 커맨드 라인 인수를 파싱하기 위한 ArgumentParser 생성
parser = argparse.ArgumentParser(description="Print name and ID or perform linear regression")

# --action, --name, --id 인수를 정의
parser.add_argument("--action", choices=["print", "linear"], help="Action to perform: 'print' or 'linear'")
parser.add_argument("--name", type=str, help="Your name")
parser.add_argument("--id", type=str, help="Your ID")

# --x1, --y1, --x2, --y2 인수를 정의
parser.add_argument("--x1", type=float, help="X-coordinate of the first point")
parser.add_argument("--y1", type=float, help="Y-coordinate of the first point")
parser.add_argument("--x2", type=float, help="X-coordinate of the second point")
parser.add_argument("--y2", type=float, help="Y-coordinate of the second point")

# 커맨드 라인 인수를 파싱
args = parser.parse_args()

if args.action == "print":
    # --print 인수가 제공되었을 때 이름과 학번을 출력
    if args.name and args.id:
        print(f"Name: {args.name}, ID: {args.id}")
    else:
        print("Both name and ID are required for printing.")
elif args.action == "linear":
    # --linear 인수가 제공되었을 때 선형 회귀 수행
    if args.x1 is not None and args.y1 is not None and args.x2 is not None and args.y2 is not None:
        # 입력 데이터
        x = torch.tensor([[args.x1, args.x2]])
        y = torch.tensor([[args.y1, args.y2]])

        # 선형 회귀 모델 정의
        class LinearRegression(nn.Module):
            def __init__(self):
                super(LinearRegression, self).__init__()
                self.linear = nn.Linear(1, 1)

            def forward(self, x):
                return self.linear(x)

        model = LinearRegression()

        # 손실 함수와 옵티마이저 설정
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # 훈련
        for epoch in range(1000):
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        # 훈련 결과 출력
        print("선형 회귀 모델 파라미터:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.data.item()}")

        # 예측 결과 출력
        with torch.no_grad():
            predicted = model(x)
            for i in range(len(predicted)):
                print(f"예측 결과 {i + 1}: {predicted[i][0].item()}")
    else:
        print("Both sets of coordinates (x1, y1, x2, y2) are required for linear regression.")
else:
    print("Please specify an action: 'print' or 'linear'.")
