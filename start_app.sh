#!/bin/bash

# 🧭 SberIndexNavigator - Скрипт запуска приложения
# Автоматический запуск Streamlit приложения с оптимальными настройками

echo "🧭 Запуск SberIndexNavigator..."

# Проверяем, что мы в правильной директории
if [ ! -f "app.py" ]; then
    echo "❌ Ошибка: файл app.py не найден в текущей директории"
    echo "Убедитесь, что вы находитесь в корневой папке проекта SberIndexNavigator"
    exit 1
fi

# Проверяем виртуальное окружение
if [ ! -d "venv" ]; then
    echo "❌ Виртуальное окружение не найдено"
    echo "Создайте виртуальное окружение: python3 -m venv venv"
    exit 1
fi

# Активируем виртуальное окружение
echo "📦 Активация виртуального окружения..."
source venv/bin/activate

# Проверяем, что streamlit установлен
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit не установлен"
    echo "Установите зависимости: pip install -r requirements.txt"
    exit 1
fi

# Проверяем .env файл
if [ ! -f ".env" ]; then
    echo "⚠️  Предупреждение: .env файл не найден"
    echo "Убедитесь, что OPENAI_API_KEY настроен в переменных окружения"
fi

# Останавливаем предыдущие процессы streamlit (если есть)
echo "🔄 Остановка предыдущих процессов..."
pkill -f "streamlit run" 2>/dev/null || true

# Ждем немного для корректного завершения процессов
sleep 2

# Запускаем приложение
echo "🚀 Запуск Streamlit приложения..."
echo "📍 URL: http://localhost:8501"
echo "📍 Network URL будет показан после запуска"
echo ""
echo "Для остановки приложения нажмите Ctrl+C"
echo "================================================"

# Запуск с оптимальными настройками
streamlit run app.py \
    --server.port 8501 \
    --server.headless true \
    --server.fileWatcherType none \
    --browser.gatherUsageStats false \
    --logger.level info

echo ""
echo "👋 SberIndexNavigator остановлен" 