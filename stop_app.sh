#!/bin/bash

# 🛑 SberIndexNavigator - Скрипт остановки приложения
# Корректная остановка всех процессов Streamlit

echo "🛑 Остановка SberIndexNavigator..."

# Останавливаем все процессы streamlit
pkill -f "streamlit run" 2>/dev/null

# Проверяем результат
if [ $? -eq 0 ]; then
    echo "✅ Приложение остановлено успешно"
else
    echo "ℹ️  Активные процессы не найдены"
fi

# Показываем статус портов
echo "📍 Проверка порта 8501..."
if lsof -i :8501 >/dev/null 2>&1; then
    echo "⚠️  Порт 8501 все еще занят"
    echo "Процессы на порту 8501:"
    lsof -i :8501
else
    echo "✅ Порт 8501 свободен"
fi

echo "👋 Готово!" 