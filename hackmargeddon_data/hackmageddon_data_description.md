# hackmageddon_normalized.csv — Descrição dos Dados

**8.009 eventos de cyberataques** (2023–2026), coletados do Hackmageddon.

Este arquivo é o nosso rótulo principal para o projeto.

---

## Colunas

| Coluna | Preench. | Descrição |
|---|---|---|
| `id_raw` | 96.5% | ID original da fonte |
| `date_reported_raw` | 100% | Data como aparece na fonte (string) |
| `date_occurred_raw` | 65.6% | Data em que o ataque ocorreu (string) |
| `date_discovered_raw` | 67.2% | Data em que foi descoberto (string) |
| `author` | 54.3% | Ator responsável (ex: BlackCat ALPHV, APT28) |
| `target` | ~100% | Vítima/alvo |
| `description` | ~100% | Descrição textual do incidente |
| `attack_raw` | ~100% | Tipo de ataque — texto original |
| `attack_norm` | 100% | **Tipo de ataque normalizado** (rótulo técnico) |
| `target_class` | 100% | Setor da vítima (39 categorias) |
| `attack_class` | 97.1% | **Motivação/classe** do ataque |
| `country` | 96.0% | País da vítima |
| `link` | ~100% | URL da fonte do incidente |
| `initial_access` | **0%** | Vazia — não usável |
| `source_timeline_url` | 100% | URL da timeline Hackmageddon |
| `source_year` | 100% | Ano |
| `row_hash` | 100% | Hash de deduplicação |
| `date_reported` | 99.3% | Data reportada parseada (ISO) |
| `date_occurred` | 23.9% | Data de ocorrência parseada (ISO) |
| `date_discovered` | 33.7% | Data de descoberta parseada (ISO) |
| `parse_status_date_reported` | 100% | Status do parse da data reportada |
| `parse_status_date_occurred` | 100% | Status do parse da data de ocorrência |
| `parse_status_date_discovered` | 100% | Status do parse da data de descoberta |

---

## Rótulos Principais

### `attack_norm` — tipo de ataque normalizado (126 valores únicos)

Distribuição concentrada no topo:

| Valor | Contagem | % |
|---|---|---|
| `malware` | 2.475 | 30.9% |
| `unknown` | 1.550 | 19.4% |
| `vulnerability` | 951 | 11.9% |
| `account takeover` | 765 | 9.6% |
| `targeted attack` | 652 | 8.1% |
| `ransomware` | 559 | 7.0% |
| `ddos` | 249 | 3.1% |
| `scam` | 148 | 1.8% |
| `coordinated inauthentic behavior` | 143 | 1.8% |
| `malicious script injection` | 69 | 0.9% |
| ... (116 outros) | ~408 | ~5.1% |

### `attack_class` — motivação/categoria do ataque

| Valor | Contagem |
|---|---|
| `Cyber Crime` | 5.413 |
| `Cyber Espionage` | 885 |
| `Hacktivism` | 336 |
| `Cyber Warfare` | 238 |
| `Unknown` | 208 |

> Atenção: existem duplicatas de codificação (`CC` = `Cyber Crime`, `CE` = `Cyber Espionage`, `H` = `Hacktivism`, `CW` = `Cyber Warfare`) que precisam ser normalizadas.

### `target_class` — setor da vítima (39 categorias)

Top valores: `Multiple Industries`, `Individual`, `Public admin and defence`, `Human health and social work`, `Finance and insurance`, `Education`, `Manufacturing`, entre outros.

---

## Distribuição por Ano

| Ano | Eventos |
|---|---|
| 2023 | 3.520 |
| 2024 | 2.804 |
| 2025 | 1.150 |
| 2026 | 535 |

---

## Pontos de Atenção

1. **`initial_access` está 100% vazia** — coluna inutilizável no estado atual.
2. **`attack_norm` tem cauda longa** — 80%+ do dado está em ~7 classes; considerar agregação das classes raras.
3. **`attack_class` tem codificações duplicadas** — `CC`, `CE`, `H`, `CW` precisam ser unificadas com os valores por extenso.
4. **`unknown` representa 19%** do `attack_norm` — ruído relevante se for usado como rótulo alvo.
5. **Datas de ocorrência e descoberta têm cobertura baixa** — `date_occurred` com 23.9% e `date_discovered` com 33.7% parseados com sucesso.
