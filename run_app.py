"""
Обёртка для PyInstaller. Запускает Streamlit программно (не через 'streamlit run'),
что нужно для сборки в один .exe. Этот файл не редактируется — вся логика
приложения живёт в app.py, который подхватывается отсюда.
"""
import os
import sys
import streamlit.web.cli as stcli


def resource_path(relative_path):
    """Находит путь к файлу и внутри .exe (PyInstaller распаковывает во временную
    папку sys._MEIPASS), и при обычном запуске 'python run_app.py' для отладки."""
    if hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(resource_path("app.py"))))
    sys.argv = [
        "streamlit",
        "run",
        resource_path("app.py"),
        "--global.developmentMode=false",
        "--server.headless=false",
        "--browser.gatherUsageStats=false",
    ]
    sys.exit(stcli.main())
