# Langfuse локально — для `s4b_langfuse.py`

Потрібен лише Docker (Desktop на Windows/macOS, ≥ 6 ГБ пам'яті в Settings →
Resources). Перший запуск тягне ~3 ГБ образів — робіть це до заняття, не під час.

## Запуск

```bash
cd 09_lecture_security_observability/demo_example/langfuse_docker
docker compose up -d
curl -s http://localhost:3000/api/public/health     # чекати {"status":"OK",...}
```

Ключі створює сам Langfuse при першому старті (`LANGFUSE_INIT_*` у compose),
тому в UI клікати нічого не треба — просто покладіть у `demo_example/.env`:

```
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-shopbot-demo-0000
LANGFUSE_SECRET_KEY=sk-lf-shopbot-demo-0000
```

І запускайте демо з папки `demo_example`:

```bash
python s4b_langfuse.py     # наприкінці друкує пряме посилання на трасу
```

UI: <http://localhost:3000>, логін `demo@shopbot.local` / `shopbot-demo-2026`,
проєкт `shopbot-lecture-09`.

## Керування

```bash
docker compose stop        # зупинити, дані лишаються
docker compose down        # прибрати контейнери, дані лишаються
docker compose down -v     # ⚠️ повний reset разом із даними і ключами
docker compose logs -f langfuse-web
```

## Якщо щось пішло не так

- **порт 3000 зайнятий** — поміняйте `"3000:3000"` у `docker-compose.yml`
  (і `LANGFUSE_HOST` у `.env`); те саме з `9090` для minio;
- **health-check мовчить** — найчастіше clickhouse не вистачає пам'яті,
  дивіться `docker compose logs clickhouse`;
- **змінили `LANGFUSE_INIT_*`, а Langfuse їх не бачить** — вони читаються лише
  при першому старті порожньої бази, потрібен `down -v`;
- **docker підняти не вийшло** — є безкоштовна хмара: <https://cloud.langfuse.com>
  → Settings → API Keys, і ті самі три змінні в `.env`. Код `s4b` не змінюється.

Паролі, `SALT` і `ENCRYPTION_KEY` у compose демонстраційні й лежать у git
відкрито. Для будь-чого, крім навчання, їх треба замінити — про це і є лекція.
