import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Загружаем данные
train_data = pd.read_csv('train.csv', low_memory=False)
test_data = pd.read_csv('test.csv', low_memory=False)


# Очистка колонки Age
def clean_age_column(df):
    df['Age'] = df['Age'].astype(str).str.replace(r'[^0-9-]', '', regex=True)  # Удаляем все, кроме цифр и '-'
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')  # Преобразуем в число, ошибки -> NaN
    df['Age'].fillna(df['Age'].median(), inplace=True)  # Заполняем NaN медианой
    df.loc[df['Age'] < 0, 'Age'] = df['Age'].median()  # Убираем аномальные значения


# Применяем очистку к обоим датасетам
clean_age_column(train_data)
clean_age_column(test_data)


# Предобработка данных
def preprocess_data(df, is_train=True):
    df = df.copy()

    # Заполняем пропуски медианой для числовых признаков
    numeric_cols = ['Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
                    'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Credit_History_Age']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].median(), inplace=True)

    # Кодируем Payment_Behaviour
    df['Payment_Behaviour'] = df['Payment_Behaviour'].astype('category').cat.codes

    if is_train:
        # Кодируем целевую переменную
        label_encoder = LabelEncoder()
        df['Credit_Score'] = label_encoder.fit_transform(df['Credit_Score'])
        return df, label_encoder
    return df


train_data, label_encoder = preprocess_data(train_data)
test_data = preprocess_data(test_data, is_train=False)

# Выбираем признаки и целевую переменную
features = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
            'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Credit_History_Age', 'Payment_Behaviour']
X = train_data[features].values
y = train_data['Credit_Score'].values

# Масштабируем данные
scaler = StandardScaler()
X = scaler.fit_transform(X)
test_X = scaler.transform(test_data[features].values)

# Разделяем данные на тренировочные и валидационные
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Преобразуем в тензоры
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
test_X_tensor = torch.tensor(test_X, dtype=torch.float32)


# Создаем нейросеть
class CreditScoreModel(nn.Module):
    def __init__(self):
        super(CreditScoreModel, self).__init__()
        self.fc1 = nn.Linear(len(features), 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Инициализация модели, функции потерь и оптимизатора
model = CreditScoreModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    # Оценка на валидации
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}')

# Предсказание на тесте
model.eval()
with torch.no_grad():
    test_outputs = model(test_X_tensor)
    predictions = torch.argmax(test_outputs, dim=1).numpy()

# Декодируем предсказания
predicted_labels = label_encoder.inverse_transform(predictions)

# Сохраняем результат
submission = pd.DataFrame({'Credit_Score': predicted_labels})
submission.to_csv('submission.csv', index=False)
print('Готово! Результаты сохранены в submission.csv')
