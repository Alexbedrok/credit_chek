

Описание

Данное веб-приложение использует нейросеть на PyTorch для предсказания кредитного рейтинга клиентов.

Основные признаки (фичи):

Age – возраст клиента (исправляем аномалии, например, -500)

Annual_Income – годовой доход

Monthly_Inhand_Salary – ежемесячная зарплата

Num_Bank_Accounts – количество банковских счетов

Outstanding_Debt – непогашенный долг

Credit_Utilization_Ratio – коэффициент использования кредита

Credit_History_Age – возраст кредитной истории

Payment_Behaviour – поведение по платежам

Целевая переменная: Credit_Score (Good, Standard, Poor)

Файлы проекта

app.py – Flask-приложение для загрузки данных и предсказания

model.pth – обученная нейросеть на PyTorch

templates/index.html – веб-интерфейс (форма загрузки)

uploads/ – папка для загруженных файлов

README.md – описание проекта (этот файл)

🛠️ Установка и запуск

1Установить зависимости:

pip install flask pandas torch scikit-learn

2 Сохранить обученную модель (если еще не сохранена):

import torch
# Подставь свою модель
torch.save(model.state_dict(), "model.pth")

3 Запустить сервер:

python app.py

4 Открыть браузер и перейти по адресу:

http://127.0.0.1:5000/

📤 Как пользоваться?

Загрузите CSV-файл с данными клиентов

Нажмите кнопку "Предсказать"

Скачайте submission.csv с результатами

🌍 Разворачивание на сервере

Можно развернуть на Heroku, Render, AWS и других сервисах.
Если нужно, помогу с деплоем! 🚀
