# zero.kz Crawler

Парсинг каталога казнета zero.kz → markdown через docling.

## Этапы

1. `parse_sitemap.py` — парсинг sitemap.xml, получение списка всех URL
2. `crawl_pages.py` — скачивание страниц через crawl4ai
3. `convert_to_md.py` — конвертация HTML → markdown через docling

## Установка

```bash
cd zerokz_crawler
pip install -r requirements.txt
```

## Запуск

```bash
# 1. Получить список URL из sitemap
python parse_sitemap.py

# 2. Скачать страницы
python crawl_pages.py

# 3. Конвертировать в markdown
python convert_to_md.py
```
