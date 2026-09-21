"""
PyInstaller hook — без него сборка находит модуль streamlit по коду, но не
находит его package-метаданные (version, entry_points), из-за чего .exe
падает при старте с ошибкой вида 'No module named streamlit.runtime' или
'PackageNotFoundError'. Положить в папку ./hooks/ рядом с run_app.py — этот
файл не редактируется.
"""
from PyInstaller.utils.hooks import copy_metadata, collect_data_files

datas = copy_metadata("streamlit")
datas += collect_data_files("streamlit")
