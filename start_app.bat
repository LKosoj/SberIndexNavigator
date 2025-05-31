@echo off
chcp 65001 >nul

REM 🧭 SberIndexNavigator - Скрипт запуска приложения для Windows
REM Автоматический запуск Streamlit приложения с оптимальными настройками

echo 🧭 Запуск SberIndexNavigator...
echo.

REM Проверяем, что мы в правильной директории
if not exist "app.py" (
    echo ❌ Ошибка: файл app.py не найден в текущей директории
    echo Убедитесь, что вы находитесь в корневой папке проекта SberIndexNavigator
    pause
    exit /b 1
)

REM Проверяем виртуальное окружение
if not exist "venv" (
    echo ❌ Виртуальное окружение не найдено
    echo Создайте виртуальное окружение: python -m venv venv
    pause
    exit /b 1
)

REM Активируем виртуальное окружение
echo 📦 Активация виртуального окружения...
call venv\Scripts\activate.bat

REM Проверяем, что streamlit установлен
streamlit --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Streamlit не установлен
    echo Установите зависимости: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Проверяем .env файл
if not exist ".env" (
    echo ⚠️  Предупреждение: .env файл не найден
    echo Убедитесь, что OPENAI_API_KEY настроен в переменных окружения
    echo.
)

REM Останавливаем предыдущие процессы streamlit (если есть)
echo 🔄 Остановка предыдущих процессов...
taskkill /f /im "streamlit.exe" >nul 2>&1
taskkill /f /im "python.exe" /fi "WINDOWTITLE eq streamlit*" >nul 2>&1

REM Ждем немного для корректного завершения процессов
timeout /t 2 /nobreak >nul

REM Запускаем приложение
echo 🚀 Запуск Streamlit приложения...
echo 📍 URL: http://localhost:8501
echo 📍 Network URL будет показан после запуска
echo.
echo Для остановки приложения нажмите Ctrl+C
echo ================================================
echo.

REM Запуск с оптимальными настройками
streamlit run app.py ^
    --server.port 8501 ^
    --server.headless true ^
    --server.fileWatcherType none ^
    --browser.gatherUsageStats false ^
    --logger.level info

echo.
echo 👋 SberIndexNavigator остановлен
pause 