# ⚡ Быстрый старт SberIndexNavigator

## 🚀 Запуск в одну команду

### macOS/Linux
```bash
./start_app.sh
```

### Windows
```bash
start_app.bat
```

## 🛑 Остановка

### macOS/Linux
```bash
./stop_app.sh
```

### Windows
```bash
Ctrl+C
```

## 📍 URL приложения
http://localhost:8501

## 🎯 Готовые демо-вопросы
1. **"Покажи динамику потребительских расходов в Москве за 2023 год"**
2. **"Сравни доступность рынков в Казани и Владивостоке"**
3. **"Где самые проблемные муниципалитеты по транспортной доступности?"**

## 📋 Что делают скрипты
- ✅ Проверяют все зависимости
- ✅ Активируют виртуальное окружение
- ✅ Останавливают старые процессы
- ✅ Запускают с оптимальными настройками
- ✅ Показывают понятные сообщения об ошибках

## 🆘 Если что-то не работает
1. Убедитесь что `.env` файл настроен
2. Проверьте что `venv` создано: `python3 -m venv venv`
3. Установите зависимости: `pip install -r requirements.txt`
4. Запустите тесты: `python test_basic.py` 