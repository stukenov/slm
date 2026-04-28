# Google TRC — Terminal Management Setup

## Быстрый старт

### 1. Настрой credentials

Скопируй скачанный ключ сервисного аккаунта в эту папку:
```bash
cp ~/Downloads/sozkz-tpu-key.json ~/slm/trc/credentials/sozkz-tpu-key.json
```

### 2. Установи переменную окружения
```bash
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/slm/trc/credentials/sozkz-tpu-key.json"
export GOOGLE_CLOUD_PROJECT="sozkz-trc"
```

Или добавь в `~/.bashrc`:
```bash
echo 'export GOOGLE_APPLICATION_CREDENTIALS="$HOME/slm/trc/credentials/sozkz-tpu-key.json"' >> ~/.bashrc
echo 'export GOOGLE_CLOUD_PROJECT="sozkz-trc"' >> ~/.bashrc
```

### 3. Установи gcloud CLI (если не установлен)
```bash
# macOS
brew install google-cloud-sdk

# Linux
curl https://sdk.cloud.google.com | bash

# После установки:
gcloud auth activate-service-account sozkz-tpu@sozkz-trc.iam.gserviceaccount.com \
  --key-file=$HOME/slm/trc/credentials/sozkz-tpu-key.json
gcloud config set project sozkz-trc
```

### 4. Используй скрипты
```bash
cd ~/slm/trc/scripts/

# Посмотреть все TPU во всех зонах
python3 list_tpus.py

# Создать TPU
python3 create_tpu.py --type v5e --zone us-central1-a --spot

# Удалить TPU
python3 delete_tpu.py --name sozkz-v5e-worker --zone us-central1-a

# Проверить квоту
python3 check_quota.py
```

## Структура папки

```
trc/
├── README.md              — этот файл
├── credentials/           — ключи доступа (НЕ коммитить в git!)
│   └── sozkz-tpu-key.json
├── scripts/               — скрипты управления TPU
│   ├── config.py          — конфигурация проекта и зон
│   ├── list_tpus.py       — листинг всех TPU
│   ├── create_tpu.py      — создание TPU VM
│   ├── delete_tpu.py      — удаление TPU VM
│   └── check_quota.py     — проверка квоты
└── .gitignore             — игнорирует credentials/
```

## Сервисный аккаунт

- **Email:** sozkz-tpu@sozkz-trc.iam.gserviceaccount.com
- **Роли:** tpu.admin, compute.viewer
- **Ключ:** credentials/sozkz-tpu-key.json

## Квота (подробности в /slm/GOOGLE_TRC_GRANT.md)

| TPU | Чипов | Тип | Зона |
|-----|-------|-----|------|
| v6e | 64 | spot | us-east1-d |
| v4 | 32 | spot | us-central2-b |
| v4 | 32 | on-demand | us-central2-b |
| v5e | 64 | spot | us-central1-a |
| v6e | 64 | spot | europe-west4-a |
| v5e | 64 | spot | europe-west4-b |
