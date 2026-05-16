# `temportal_graph`: grafo temporal para previsao de ameacas

Este modulo constroi e treina um grafo temporal heterogeneo para prever, a partir de mensagens de grupos de Telegram, o volume e os tipos de incidentes ciberneticos reportados pelo Hackmageddon no dia seguinte.

Apesar dos nomes internos usarem `weekly`, a tarefa atual nao e uma previsao semanal agregada. O que existe hoje e uma janela movel diaria de 7 dias: para cada dia de snapshot `D`, o modelo recebe as mensagens observadas de `D-6` ate `D` e tenta prever os eventos Hackmageddon reportados em `D+1`.

## Como reproduzir

Os dois comandos principais sao:

```bash
uv run build_weekly_graph.py
uv run train_weekly.py
```

O primeiro comando gera:

- `temportal_graph/output/weekly_dataset.pt`: objeto `HeteroData` do PyTorch Geometric.
- `temportal_graph/output/weekly_metadata.json`: estatisticas do grafo, distribuicoes e parametros de construcao.

O segundo comando gera:

- `temportal_graph/output/weekly_metrics.json`: metricas de treino, validacao e teste.
- `temportal_graph/output/weekly_training_history.png`
- `temportal_graph/output/weekly_val_count_confusion_matrix.png`
- `temportal_graph/output/weekly_test_count_confusion_matrix.png`
- `temportal_graph/output/weekly_test_type_metrics.png`

Tambem existe uma visualizacao HTML:

```bash
uv run visualize_weekly.py
```

Ela le `output/weekly_dataset.pt` e escreve `output/graph_viz.html`.

## Entradas

O grafo usa duas fontes principais.

1. Mensagens classificadas deterministicamente em `sentinel_replica_jsons_deterministic/`.
2. Eventos Hackmageddon normalizados em `hackmargeddon_data/hackmageddon_normalized.csv`.

As mensagens do Telegram sao a entrada observavel do modelo. Cada registro contem, quando disponivel:

- `source_file`: arquivo/grupo de origem.
- `date`: data da mensagem.
- `message`: texto original.
- `cleaning.normalized_message`: texto normalizado.
- `classification.entities`: entidades extraidas, como URLs, CVEs, dominios, paises, setores, malware, atores, produtos e outros indicadores.

O Hackmageddon e usado como alvo supervisionado. A data usada e `date_reported`; portanto, o modelo aprende a prever o que sera reportado no dia seguinte, nao necessariamente a data real de ocorrencia do ataque.

## Grupos usados e ignorados

Por padrao, o builder carrega todos os JSONs de `sentinel_replica_jsons_deterministic/`, mas remove os grupos definidos em `--exclude-groups`.

Exclusoes padrao:

- `bellingcat`
- `WokeIntelDrops`
- `itsectalk`
- `itsecalert`

No estado atual dos dados locais, `bellingcat` e `itsecalert` nao aparecem como JSONs nesse diretorio, mas continuam configurados como exclusoes preventivas. Os grupos efetivamente encontrados foram:

| Grupo | Registros | Com data | Inicio | Fim | Mensagens com entidades | Uso |
|---|---:|---:|---|---|---:|---|
| `HackingBlogsGroup` | 152 | 152 | 2024-07-03 | 2025-09-10 | 137 | entrada do grafo |
| `PHOfficial` | 171 | 171 | 2023-01-02 | 2025-08-06 | 171 | entrada do grafo |
| `WokeIntelDrops` | 1 | 1 | 2023-03-03 | 2023-03-03 | 0 | ignorado por padrao |
| `cissp` | 562 | 562 | 2023-01-10 | 2026-05-08 | 547 | entrada do grafo |
| `cloudandcybersecurity` | 3 | 3 | 2023-07-15 | 2023-11-24 | 3 | entrada do grafo |
| `cybdetective` | 154 | 154 | 2023-01-02 | 2026-05-08 | 148 | entrada do grafo |
| `cybersecurityexperts` | 5045 | 5045 | 2023-01-01 | 2026-05-09 | 880 | entrada do grafo |
| `hackers_asylum` | 148 | 148 | 2023-01-17 | 2026-05-08 | 123 | entrada do grafo |
| `itsectalk` | 18 | 18 | 2023-02-05 | 2026-04-02 | 3 | ignorado por padrao |

Depois das exclusoes, entraram no grafo 7 grupos e 6.235 mensagens com data valida.

## Espaco de tempo analisado

O intervalo de mensagens de entrada vai de 2023-01-01 ate 2026-05-09, considerando os grupos incluidos. Os snapshots diarios construidos totalizam 1.155 dias.

A estrategia temporal e:

- Snapshot `D`: representa um dia de referencia.
- Janela de entrada: mensagens entre `D-6` e `D`, inclusive.
- Alvo: eventos Hackmageddon com `date_reported = D+1`.
- Minimo de mensagens: o snapshot so e criado se a janela tiver pelo menos 3 mensagens.
- Granularidade do alvo: diaria.

Como o Hackmageddon tem meses com cobertura irregular, o builder marca cada snapshot como `observed` ou `uncovered_month`. Um mes e considerado confiavel quando tem pelo menos 10 dias positivos no Hackmageddon (`coverage_month_min_positive_days = 10`). O treino usa apenas snapshots com label observado por meio de `train_mask`, mas o dataset preserva tambem snapshots sem cobertura.

Meses considerados confiaveis no artefato atual:

`2023-01` a `2025-02`, exceto meses sem cobertura suficiente depois disso, e `2026-01`, `2026-02`, `2026-03`.

Numeros principais:

- 1.155 snapshots diarios.
- 825 snapshots com label observado.
- 330 snapshots em meses sem cobertura confiavel.
- 782 snapshots positivos.
- 67,7% de snapshots positivos no total.
- 94,8% de positivos entre os snapshots com label observado.
- Media de eventos Hackmageddon por snapshot observado: 9,12.
- Mediana de eventos por snapshot observado: 9.
- Maximo observado em um dia alvo: 34 eventos.

## Estrategia analisada

A pergunta modelada e:

> Dadas as mensagens recentes de canais de ciberseguranca no Telegram, quais tipos e qual volume de incidentes ciberneticos serao reportados amanha no Hackmageddon?

Isso transforma a base em uma tarefa de forecasting supervisionado. O modelo nao classifica diretamente se uma mensagem individual e ameaca; ele usa mensagens, entidades, grupos e dias como contexto para prever propriedades do dia seguinte.

A saida e multitarefa:

1. Bucket de contagem de eventos em `D+1`.
2. Tipos de ameaca presentes em `D+1`.
3. Contagem aproximada por tipo de ameaca em `D+1`.

Os buckets de volume sao:

| Classe | Significado |
|---|---|
| `0` | nenhum evento reportado no dia seguinte |
| `1-2` | 1 ou 2 eventos |
| `3-5` | 3 a 5 eventos |
| `6-10` | 6 a 10 eventos |
| `11-15` | 11 a 15 eventos |
| `16+` | 16 ou mais eventos |

Distribuicao dos buckets nos 825 snapshots com label observado:

| Bucket | Snapshots |
|---|---:|
| `0` | 43 |
| `1-2` | 122 |
| `3-5` | 112 |
| `6-10` | 215 |
| `11-15` | 202 |
| `16+` | 131 |

## Estrutura do grafo

O dataset e um `torch_geometric.data.HeteroData`. Ele tem quatro tipos de no:

| Tipo de no | Quantidade | Significado |
|---|---:|---|
| `message` | 6.235 | Mensagem de Telegram usada como evidencia textual e semantica |
| `entity` | 2.457 | Entidade extraida da mensagem, como URL, CVE, dominio, handle, malware, setor etc. |
| `group` | 7 | Grupo/canal de origem da mensagem |
| `day` | 1.155 | Snapshot diario de previsao |

Um no e uma unidade do grafo que recebe um vetor de atributos. No caso deste projeto:

- No `message`: representa uma mensagem individual.
- No `entity`: representa uma entidade canonica mencionada em uma ou mais mensagens.
- No `group`: representa a origem da mensagem.
- No `day`: representa um dia de snapshot, com labels futuros associados.

Uma aresta e uma relacao entre dois nos. As arestas indicam como a informacao deve circular no grafo durante o HGT.

| Aresta | Quantidade | Significado |
|---|---:|---|
| `message -> mentions -> entity` | 4.880 | Mensagem menciona entidade |
| `entity -> mentioned_by -> message` | 4.880 | Reverso de mencao |
| `message -> posted_in -> group` | 6.235 | Mensagem pertence a grupo |
| `group -> has_message -> message` | 6.235 | Reverso da relacao grupo-mensagem |
| `message -> observed_before_snapshot -> day` | 42.174 | Mensagem pertence a janela de 7 dias do snapshot |
| `day -> has_observed_message -> message` | 42.174 | Reverso da janela temporal |
| `message -> similar_to -> message` | 18.341 | Mensagem semanticamente similar a outra mensagem |
| `message -> rev_similar_to -> message` | 18.341 | Reverso da similaridade |
| `entity -> co_occurs -> entity` | 28.346 | Entidades aparecem juntas em mensagens |
| `entity -> rev_co_occurs -> entity` | 28.346 | Reverso da coocorrencia |
| `day -> next -> day` | 1.154 | Sequencia temporal entre snapshots consecutivos |

As principais regras de criacao de arestas sao:

- `message-entity`: uma aresta para cada entidade extraida da mensagem.
- `message-group`: uma aresta entre a mensagem e o grupo do arquivo de origem.
- `message-day`: uma mensagem se conecta a todos os snapshots cuja janela de 7 dias a contem.
- `message-message`: mensagens semanticamente similares sao ligadas usando embeddings; por padrao, top-k 8 e similaridade minima 0,70.
- `entity-entity`: entidades que coocorrem na mesma mensagem sao ligadas com peso baseado em frequencia, diversidade de grupos e decaimento temporal.
- `day-day`: cada dia aponta para o proximo, preservando a ordem temporal.

## Atributos dos nos

### `message`

Cada mensagem recebe:

- Embedding textual Qwen de 1.024 dimensoes.
- Flag `ioc_present`, indicando se ha entidades extraidas.
- `log1p(numero_de_entidades)`.

Por padrao, coordenadas de tempo absoluto foram removidas (`include_absolute_time = false`) para reduzir vazamento temporal e overfitting por data global.

### `entity`

Cada entidade recebe:

- One-hot do tipo da entidade.
- `log1p(numero_de_mensagens_que_mencionam_a_entidade)`.
- Flag indicando se veio de regex.
- Flag indicando se veio de NER.

Distribuicao dos tipos de entidade:

| Tipo | Quantidade |
|---|---:|
| `url` | 1.870 |
| `cve` | 257 |
| `domain` | 100 |
| `handle` | 68 |
| `victim` | 26 |
| `country` | 20 |
| `product` | 19 |
| `vulnerability` | 18 |
| `email` | 16 |
| `malware` | 12 |
| `ttp` | 12 |
| `sector` | 10 |
| `tool` | 10 |
| `threat_actor` | 9 |
| `hash` | 7 |
| `wallet` | 2 |
| `ip_address` | 1 |

### `group`

Cada grupo recebe:

- `log1p(numero_de_mensagens_do_grupo)`.
- Proporcao das mensagens totais que vieram daquele grupo.

### `day`

Cada snapshot diario recebe:

- Media normalizada dos embeddings das mensagens na janela.
- `log1p(numero_de_mensagens_observadas_na_janela)`.
- One-hot do dia da semana.

Como `include_hackmageddon_context = false`, eventos Hackmageddon anteriores nao entram como features de contexto. Isso evita que a fonte de label vire tambem sinal de entrada.

## Embeddings e modelos utilizados

### Embedding textual

O builder usa por padrao:

- Backend: `qwen`
- Modelo: `Qwen/Qwen3-Embedding-0.6B`
- Dimensao: 1.024
- Cache: `graph/cache/qwen3_embedding_0_6b`

Esse modelo e usado como encoder de embeddings, nao como classificador gerativo. Ele transforma cada texto em um vetor denso. Esses vetores alimentam:

- features dos nos `message`;
- media dos embeddings nos nos `day`;
- calculo de similaridade semantica para arestas `message -> similar_to -> message`.

Existe tambem um backend alternativo `tfidf_svd`, usado principalmente para smoke tests e execucoes sem baixar modelo.

### GNN: HGT

O modelo principal e `WeeklyMultiTaskForecaster`, definido em `weekly/model.py`.

Ele usa `HGTConv`, uma camada de grafo heterogeneo que trata tipos diferentes de nos e arestas separadamente. Isso e importante porque uma relacao `message -> entity` nao tem o mesmo significado que `message -> day` ou `day -> day`.

Fluxo interno:

1. Cada tipo de no passa por uma projecao linear para `hidden_dim = 128`.
2. O grafo passa por 2 camadas `HGTConv`, cada uma com 2 heads de atencao.
3. Cada camada usa conexao residual, `LayerNorm`, ReLU e dropout.
4. O resultado e um embedding contextualizado para cada no.

O HGT permite que um no `day`, por exemplo, incorpore informacao das mensagens conectadas a ele, dos grupos dessas mensagens, das entidades mencionadas e das relacoes de similaridade/coocorrencia.

### GRU temporal

Depois do encoder HGT, os embeddings dos nos `day` passam por uma GRU:

```python
gru_out, _ = self.temporal_gru(h_dict["day"].unsqueeze(0))
```

Essa GRU processa a sequencia de dias na ordem em que aparece no dataset. A ideia e que o estado de um dia carregue informacao dos dias anteriores, capturando continuidade temporal que as arestas `day -> next -> day` sozinhas nao necessariamente modelam de forma suficiente.

Na pratica:

- HGT aprende estrutura relacional: mensagens, entidades, grupos, similaridades e snapshots.
- GRU aprende dinamica temporal entre snapshots diarios.

### Atencao sobre mensagens da janela

Apos a GRU, o modelo faz um pooling com atencao sobre as mensagens ligadas a cada dia por `message -> observed_before_snapshot -> day`.

Para cada snapshot, ele compara o embedding do dia com os embeddings das mensagens da janela e aprende pesos de atencao. A representacao final do dia combina:

- embedding temporal do no `day`;
- resumo ponderado das mensagens observadas naquela janela.

Essa representacao final alimenta as cabecas de predicao.

## Processo de predicao

Para cada dia `D`:

1. O builder seleciona as mensagens dos ultimos 7 dias: `[D-6, D]`.
2. Essas mensagens entram no grafo com seus grupos, entidades e similaridades.
3. O no `day` de `D` recebe as features agregadas da janela.
4. O HGT propaga informacao entre os diferentes tipos de no.
5. A GRU processa a sequencia de representacoes dos dias.
6. A atencao escolhe quais mensagens da janela sao mais relevantes para o snapshot.
7. O modelo prediz o que sera reportado no Hackmageddon em `D+1`.

As saidas sao:

- `count_bucket`: classificacao multiclasse do volume de eventos.
- `type_multilabel`: classificacao multilabel dos tipos de ameaca.
- `type_counts`: regressao em escala logaritmica para contagem por tipo.

A funcao de perda combina:

- `cross_entropy` para bucket de contagem.
- `binary_cross_entropy_with_logits` para tipos multilabel.
- `smooth_l1_loss` para contagens por tipo em `log1p`.

Pesos usados no treino atual:

| Componente | Peso |
|---|---:|
| `count_bucket_loss` | 1,00 |
| `type_multilabel_loss` | 0,25 |
| `type_count_loss` | 0,10 |

Para lidar com desbalanceamento:

- Buckets de contagem usam pesos `inverse_sqrt`.
- Labels multilabel usam `pos_weight`, limitado por `type_pos_weight_max = 20`.
- Thresholds multilabel sao escolhidos por melhor F1 na validacao (`threshold_policy = val_f1`).

## Labels e distribuicoes

Classes de ataque Hackmageddon usadas:

| Classe | Snapshots positivos |
|---|---:|
| `Cyber Crime` | 765 |
| `Cyber Espionage` | 447 |
| `Hacktivism` | 239 |
| `Cyber Warfare` | 148 |
| `Unknown` | 254 |

Tipos de ameaca com suporte positivo:

| Tipo | Snapshots positivos | Eventos |
|---|---:|---:|
| `other_cyber` | 669 | 2.306 |
| `malware` | 646 | 2.360 |
| `account_takeover` | 416 | 705 |
| `vulnerability_or_exploit` | 408 | 1.051 |
| `ransomware` | 276 | 523 |
| `ddos_or_disruption` | 186 | 259 |
| `fraud_or_scam` | 120 | 143 |
| `disinformation_or_influence` | 71 | 137 |
| `credential_theft` | 38 | 42 |

Os demais tipos existem no espaco de labels, mas ficaram sem exemplos positivos neste recorte:

`botnet_or_c2`, `data_breach_or_leak`, `exfiltration`, `initial_access_activity`, `insider_threat`, `lateral_movement`, `phishing`, `physical_or_hybrid_threat`, `privilege_escalation`, `reconnaissance`, `supply_chain_compromise`, `wiper_or_destruction`.

Isso derruba o macro-F1 multilabel, porque labels sem suporte ou rarissimas entram como tarefas muito dificeis ou impossiveis no recorte atual.

## Split de treino, validacao e teste

O split e cronologico e balanceado o suficiente para que cada particao tenha mais de uma classe de bucket. A politica usada foi `balanced_chronological`.

| Split | Snapshots | Inicio | Fim | Media de eventos | `0` | `1-2` | `3-5` | `6-10` | `11-15` | `16+` |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| Treino | 537 | 2023-01-01 | 2024-07-12 | 10,19 | 23 | 70 | 68 | 113 | 149 | 114 |
| Validacao | 123 | 2024-07-13 | 2024-11-18 | 7,86 | 8 | 21 | 12 | 45 | 26 | 11 |
| Teste | 165 | 2024-11-19 | 2026-03-30 | 6,58 | 12 | 31 | 32 | 57 | 27 | 6 |

Note que existe uma queda na media de eventos do treino para o teste. Isso indica mudanca temporal na distribuicao do alvo e torna o teste mais realista.

## Parametros do treino atual

| Parametro | Valor |
|---|---:|
| Epocas | 100 |
| Dispositivo | `cuda` |
| `hidden_dim` | 128 |
| Camadas HGT | 2 |
| Heads | 2 |
| Dropout | 0,2 |
| Learning rate | 0,001 |
| Weight decay | 0,0001 |
| Seed | 42 |

O historico mostra reducao consistente da loss:

| Epoca | Loss | Macro-F1 validacao bucket |
|---:|---:|---:|
| 1 | 1,9920 | 0,0582 |
| 10 | 1,8789 | 0,1168 |
| 20 | 1,4927 | 0,2887 |
| 40 | 1,1303 | 0,2998 |
| 60 | 0,7906 | 0,3450 |
| 80 | 0,5907 | 0,3633 |
| 100 | 0,4383 | 0,3420 |

## Resultados

### Bucket de contagem

| Split | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Treino | 0,8994 | 0,9283 | 0,8989 |
| Validacao | 0,3984 | 0,3420 | 0,3920 |
| Teste | 0,3818 | 0,3136 | 0,3845 |

No teste, o modelo supera os baselines registrados:

| Metodo | Accuracy teste | Macro-F1 teste |
|---|---:|---:|
| Modelo HGT+GRU | 0,3818 | 0,3136 |
| Bucket majoritario do treino | 0,1636 | 0,0469 |
| Bucket do dia anterior | 0,2606 | 0,2153 |

A matriz de confusao de teste mostra que o modelo aprende melhor a regiao central da distribuicao, especialmente `1-2`, `6-10` e parcialmente `11-15`. As classes extremas `0`, `3-5` e `16+` continuam mais instaveis.

Predicoes de bucket no teste:

| Bucket | Verdadeiro | Predito |
|---|---:|---:|
| `0` | 12 | 8 |
| `1-2` | 31 | 28 |
| `3-5` | 32 | 14 |
| `6-10` | 57 | 55 |
| `11-15` | 27 | 42 |
| `16+` | 6 | 18 |

### Regressao de contagem

| Split | MAE | RMSE | Media real | Media predita |
|---|---:|---:|---:|---:|
| Treino | 2,70 | 3,93 | 10,19 | 9,81 |
| Validacao | 3,06 | 4,15 | 7,86 | 8,32 |
| Teste | 3,70 | 4,98 | 6,58 | 8,07 |

No teste, a media predita ficou acima da real. Isso sugere que o modelo ainda carrega parte do regime de maior volume visto no treino.

### Tipos de ameaca multilabel

| Split | Micro-F1 | Macro-F1 |
|---|---:|---:|
| Treino | 0,6941 | 0,2499 |
| Validacao | 0,6921 | 0,2554 |
| Teste | 0,5828 | 0,1870 |

No teste, os melhores resultados aparecem nos tipos mais frequentes:

| Tipo | Suporte | Predito | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|---:|---:|
| `other_cyber` | 127 | 132 | 0,8333 | 0,8661 | 0,8494 | 0,8961 |
| `malware` | 121 | 113 | 0,8673 | 0,8099 | 0,8376 | 0,9081 |
| `ransomware` | 93 | 126 | 0,6270 | 0,8495 | 0,7215 | 0,6817 |
| `account_takeover` | 88 | 85 | 0,7176 | 0,6932 | 0,7052 | 0,6740 |
| `vulnerability_or_exploit` | 21 | 127 | 0,1575 | 0,9524 | 0,2703 | 0,1566 |
| `ddos_or_disruption` | 21 | 87 | 0,1264 | 0,5238 | 0,2037 | 0,2117 |
| `fraud_or_scam` | 14 | 118 | 0,1102 | 0,9286 | 0,1970 | 0,1718 |
| `disinformation_or_influence` | 10 | 46 | 0,0870 | 0,4000 | 0,1429 | 0,1346 |
| `credential_theft` | 2 | 28 | 0,0000 | 0,0000 | 0,0000 | 0,0177 |

O modelo tende a ter recall alto e precision baixa em labels raros. Isso e esperado com thresholds otimizados por F1 em validacao e forte desbalanceamento.

## Interpretacao geral

O experimento mostra que a estrutura temporal-relacional traz sinal acima de baselines simples para volume de eventos. O ganho e mais claro na previsao de buckets do que na previsao fina de todos os tipos.

Pontos fortes:

- O modelo aprende padroes temporais melhores que repetir o bucket do dia anterior.
- Os tipos frequentes (`malware`, `other_cyber`, `ransomware`, `account_takeover`) tem F1 razoavel no teste.
- A arquitetura separa bem fontes de informacao: texto, entidade, grupo, similaridade e tempo.
- Hackmageddon nao entra como contexto de entrada, evitando vazamento direto do alvo.

Limitacoes:

- O dataset tem forte desbalanceamento: poucos dias sem ataque e varios tipos sem nenhum exemplo positivo.
- O teste tem media de eventos menor que o treino, criando drift temporal.
- O alvo e `date_reported` do Hackmageddon, nao necessariamente a data real de ataque.
- O macro-F1 multilabel e penalizado por labels ausentes ou muito raros.
- A pasta se chama `temportal_graph`, provavelmente por typo de `temporal_graph`; os scripts funcionam com o nome atual.

## Arquivos principais

- `build_weekly_graph.py`: CLI para construir o grafo.
- `train_weekly.py`: CLI para treinar o modelo e salvar metricas.
- `visualize_weekly.py`: gera visualizacao HTML do grafo.
- `weekly/builder.py`: logica de construcao do `HeteroData`.
- `weekly/model.py`: arquitetura HGT + GRU + atencao.
- `weekly/loader.py`: leitura de mensagens classificadas e Hackmageddon.
- `weekly/features.py`: features de mensagens, dias e cobertura Hackmageddon.
- `weekly/metrics.py`: metricas de bucket, regressao e multilabel.
- `weekly/splits.py`: splits cronologicos.
- `weekly/config.py`: labels, buckets e normalizacoes.
