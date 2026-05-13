# Coleta de Dados — Telegram e Reddit (Replicação SENTINEL)

Esse repositório faz parte de uma pesquisa de mestrado em cibersegurança. O objetivo aqui é replicar a coleta de dados do SENTINEL — um framework da George Washington University que usa mensagens do Telegram para detectar ameaças cibernéticas antes que aconteçam.

## O que tem aqui

Um notebook para coletar mensagens de grupos públicos de cibersegurança no Telegram e um script Python para coletar posts recentes do Reddit por subreddit ou por usuário. Os dados brutos não estão no repositório por questões de privacidade.

## Antes de rodar

Você vai precisar de uma conta no Telegram e de credenciais de API — consegue em [my.telegram.org](https://my.telegram.org). O processo leva uns 5 minutos.

Com as credenciais em mãos, sincroniza as dependências:
```bash
uv sync
```

Depois cria um arquivo `.env` na raiz do projeto com:
```
API_ID=seu_api_id
API_HASH=seu_api_hash
REDDIT_CLIENT_ID=seu_client_id
REDDIT_CLIENT_SECRET=seu_client_secret
REDDIT_USER_AGENT=analise-database-cybersec/0.1 by u/seu_usuario
```

Tem um `.env.example` no repositório como referência.

## Coleta no Reddit

Para coletar posts dos últimos 6 meses dos subreddits padrão:

```bash
uv run python coleta/coletar_reddit.py
```

Os subreddits padrão são:
`r/cybersecurity`, `r/netsec`, `r/Malware`, `r/threatintelligence`, `r/ReverseEngineering` e `r/hacking`.

Para coletar fontes específicas:

```bash
uv run python coleta/coletar_reddit.py --months 6 --subreddits cybersecurity netsec Malware --redditors example_user
```

O script salva um JSON por fonte em `reddit_replica_jsons/`, contendo título, texto, autor, permalink, score, comentários e metadados básicos do post.

## Coleta no Telegram

Para fazer uma carga incremental até o instante da execução:

```bash
uv run python coleta/coletar_telegram.py
```

O script lê os JSONs já existentes em `sentinel_replica_jsons/`, identifica o timestamp mais recente salvo por grupo e busca mensagens novas até `datetime.now(timezone.utc)`. Cada registro novo salva `date`, `datetime_utc` e `timestamp`. Por padrão ele reaproveita 2 dias de sobreposição para evitar perdas na coleta incremental.

Se quiser forçar um ponto inicial manual:

```bash
uv run python coleta/coletar_telegram.py --start-date 2025-06-30
```

Para recoletar a base completa com timestamps corretos:

```bash
uv run python coleta/coletar_telegram.py --full-refresh --start-date 2023-01-01
```

Durante `--full-refresh`, o script cria `.bak` antes de substituir arquivos existentes. Se a nova coleta vier menor que o arquivo atual, ela é salva como `.refresh.json` e o original não é substituído, a menos que `--allow-shrink` seja informado.

Se `--end-date` for omitido, a coleta vai até hoje/agora em UTC. Para limitar o período:

```bash
uv run python coleta/coletar_telegram.py --full-refresh --start-date 2023-01-01 --end-date 2025-06-30
```

Também dá para limitar a grupos específicos:

```bash
uv run python coleta/coletar_telegram.py --groups itsectalk cybersecurityexperts
```

O notebook `coleta/coletar_telegram.ipynb` continua útil para exploração manual, mas o script é o caminho mais direto para atualizar a base.

## Sobre os dados

Coletamos 10 dos 16 grupos originais do SENTINEL. Os demais foram deletados ou perderam o histórico entre a publicação do paper (2025) e a nossa coleta (março 2026) — o que por si só já é um achado interessante sobre reprodutibilidade de datasets em redes sociais.

## Referência

Purba et al. *Towards Automated and Explainable Threat Hunting with Generative AI*. DSN 2025.
