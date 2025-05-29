import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# 1.데이터 로드
df = pd.read_csv("C:\myproject\midtermtest\python-2\data.csv")

# 2. 데이터 전처리
drop_cols = ['Unnamed: 0', 'EmployeeCount', 'StandardHours']
df.drop(columns=drop_cols, inplace=True)

df['이직여부'] = df['이직여부'].map({'Yes': 1, 'No': 0})

label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 피처 선택
features = [
    'Age', '출장빈도', '집까지거리', '근무환경만족도', '야근여부', '월급여', '총경력', '현회사근속년수'
]
# Age: 젊은 직월일수록 이직 가능성이 높다고 판단
# 출장빈도: 출장 빈도가 높을수록 높은 피로도로 인해 이직 가능성이 높다고 판단
# 집까지거리: 집과 거리가 먼 직원은 통근 스트레스가 높을 가능성이 있음
# 근무환경만족도: 근무 환경의 만족도는 이직 여부에 큰 영향을 준다고 생각함
#야근여부: 야근이 많으면 피로도 상승으로 인한 이직 가능성 증가
#월급여: 보상이 적절하지 않으면 이직 유인이 될수있음
#총경력: 경력에 따른 안전성 or 유연성 
#현회사근속년수: 회사에 오래 있을수록 이직 가능성이 낮을 것이라 판단

X = df[features]
y = df['이직여부']

# 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=13
)

# 모델 학습
model = LogisticRegression(random_state=13, solver='liblinear')
model.fit(X_train, y_train)

# 예측 및 성능 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("정확도:", accuracy)
print("혼동 행렬:\n", conf_matrix)

# 이직으로 예측된 직원 수
predicted_yes = np.sum(y_pred)
print("이직 예측 직원 수:", predicted_yes)

# 이직 확률 기준 상위 5명
y_prob = model.predict_proba(X_test)[:, 1]
top5_idx = np.argsort(y_prob)[-5:][::-1]
top5 = X_test[top5_idx]
top5_df = pd.DataFrame(top5, columns=features)
print("이직 가능성 상위 5명:")
print(top5_df)

# 신입사원 데이터 예측
new_employees = pd.DataFrame([
    {"Age": 29, "출장빈도": "Travel_Rarely", "부서": "Research & Development",
     "집까지거리": 5, "학력수준": 3, "전공분야": "Life Sciences", "근무환경만족도": 2,
     "성별": "Male", "시간당급여": 70, "업무몰입도": 3, "직급": 1, "직무": "Laboratory Technician",
     "업무만족도": 2, "결혼상태": "Single", "월급여": 2800, "이전회사경험수": 1,
     "야근여부": "Yes", "연봉인상률": 12, "성과등급": 3, "대인관계만족도": 2, "스톡옵션등급": 0,
     "총경력": 4, "연간교육횟수": 2, "워라밸": 2, "현회사근속년수": 1, "현직무근속년수": 1,
     "최근승진후경과년수": 0, "현상사근속년수": 1},
    {"Age": 42, "출장빈도": "Non-Travel", "부서": "Human Resources",
     "집까지거리": 10, "학력수준": 4, "전공분야": "Human Resources", "근무환경만족도": 3,
     "성별": "Female", "시간당급여": 85, "업무몰입도": 3, "직급": 3, "직무": "Human Resources",
     "업무만족도": 4, "결혼상태": "Married", "월급여": 5200, "이전회사경험수": 2,
     "야근여부": "No", "연봉인상률": 14, "성과등급": 3, "대인관계만족도": 3, "스톡옵션등급": 1,
     "총경력": 18, "연간교육횟수": 3, "워라밸": 3, "현회사근속년수": 7, "현직무근속년수": 4,
     "최근승진후경과년수": 1, "현상사근속년수": 3},
    {"Age": 35, "출장빈도": "Travel_Frequently", "부서": "Sales",
     "집까지거리": 2, "학력수준": 2, "전공분야": "Marketing", "근무환경만족도": 1,
     "성별": "Male", "시간당급여": 65, "업무몰입도": 2, "직급": 2, "직무": "Sales Executive",
     "업무만족도": 1, "결혼상태": "Single", "월급여": 3300, "이전회사경험수": 3,
     "야근여부": "Yes", "연봉인상률": 11, "성과등급": 3, "대인관계만족도": 2, "스톡옵션등급": 0,
     "총경력": 10, "연간교육횟수": 2, "워라밸": 2, "현회사근속년수": 2, "현직무근속년수": 1,
     "최근승진후경과년수": 1, "현상사근속년수": 1}
])

new_X = new_employees[features].copy()
for col in new_X.columns:
    if col in label_encoders:
        new_X[col] = label_encoders[col].transform(new_X[col])

new_X_scaled = scaler.transform(new_X)
new_pred = model.predict(new_X_scaled)
print("신입사원 이직 예측 결과:", new_pred)

coefs = model.coef_[0]
importance = pd.Series(coefs, index=features).sort_values(key=abs, ascending=False)
print("피처 중요도 상위 3개:")
print(importance.head(3))

#1. Age: 나이가 많을수록 이직 확률이 낮음을 의미함.
#2. 야근여부: 야근을 하는 경우 이직 가능성이 증가함을 보여줌.
#3. 근무환경만족도: 근무환경 만족도가 낮을수록 이직 가능성이 커짐.
