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
| `HackingBlogsGroup` | 188 | 188 | 2024-07-03 | 2025-09-10 | 166 | entrada do grafo |
| `PHOfficial` | 1058 | 1058 | 2023-01-01 | 2026-05-09 | 1058 | entrada do grafo |
| `RedPacketSecurity` | 56230 | 56230 | 2023-01-01 | 2026-05-09 | 56230 | entrada do grafo |
| `WokeIntelDrops` | 1 | 1 | 2023-03-03 | 2023-03-03 | 0 | ignorado por padrao |
| `cissp` | 1241 | 1241 | 2023-01-09 | 2026-05-08 | 1216 | entrada do grafo |
| `cloudandcybersecurity` | 44 | 44 | 2023-01-19 | 2026-02-21 | 38 | entrada do grafo |
| `cybdetective` | 908 | 908 | 2023-01-02 | 2026-05-09 | 879 | entrada do grafo |
| `cybersecurityexperts` | 20000 | 20000 | 2023-01-01 | 2026-05-09 | 1787 | entrada do grafo |
| `hackers_asylum` | 227 | 227 | 2023-01-17 | 2026-05-08 | 190 | entrada do grafo |
| `itsectalk` | 44 | 44 | 2023-02-05 | 2026-04-26 | 5 | ignorado por padrao |
| `twittercvenews` | 8596 | 8596 | 2023-01-01 | 2023-05-23 | 8596 | entrada do grafo |
| `vxunderground` | 2420 | 2420 | 2023-01-01 | 2026-05-08 | 1059 | entrada do grafo |

Depois das exclusoes, entraram no grafo 10 grupos e 90.912 mensagens com data valida.

## Espaco de tempo analisado

O intervalo de mensagens de entrada vai de 2023-01-01 ate 2026-05-09, considerando os grupos incluidos. Os snapshots diarios construidos totalizam 1.225 dias.

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

- 1.225 snapshots diarios.
- 879 snapshots com label observado.
- 346 snapshots em meses sem cobertura confiavel.
- 833 snapshots positivos.
- 68,0% de snapshots positivos no total.
- 94,8% de positivos entre os snapshots com label observado.
- Media de eventos Hackmageddon por snapshot observado: 9,04.
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

Distribuicao dos buckets nos 879 snapshots com label observado:

| Bucket | Snapshots |
|---|---:|
| `0` | 46 |
| `1-2` | 132 |
| `3-5` | 120 |
| `6-10` | 233 |
| `11-15` | 212 |
| `16+` | 136 |

## Estrutura do grafo

O dataset e um `torch_geometric.data.HeteroData`. Ele tem quatro tipos de no:

| Tipo de no | Quantidade | Significado |
|---|---:|---|
| `message` | 90.912 | Mensagem de Telegram usada como evidencia textual e semantica |
| `entity` | 117.354 | Entidade extraida da mensagem, como URL, CVE, dominio, handle, malware, setor etc. |
| `group` | 10 | Grupo/canal de origem da mensagem |
| `day` | 1.225 | Snapshot diario de previsao |

Um no e uma unidade do grafo que recebe um vetor de atributos. No caso deste projeto:

- No `message`: representa uma mensagem individual.
- No `entity`: representa uma entidade canonica mencionada em uma ou mais mensagens.
- No `group`: representa a origem da mensagem.
- No `day`: representa um dia de snapshot, com labels futuros associados.

Uma aresta e uma relacao entre dois nos. As arestas indicam como a informacao deve circular no grafo durante o HGT.

| Aresta | Quantidade | Significado |
|---|---:|---|
| `message -> mentions -> entity` | 152.191 | Mensagem menciona entidade |
| `entity -> mentioned_by -> message` | 152.191 | Reverso de mencao |
| `message -> posted_in -> group` | 90.912 | Mensagem pertence a grupo |
| `group -> has_message -> message` | 90.912 | Reverso da relacao grupo-mensagem |
| `message -> observed_before_snapshot -> day` | 634.740 | Mensagem pertence a janela de 7 dias do snapshot |
| `day -> has_observed_message -> message` | 634.740 | Reverso da janela temporal |
| `message -> similar_to -> message` | 378.995 | Mensagem semanticamente similar a outra mensagem |
| `message -> rev_similar_to -> message` | 378.995 | Reverso da similaridade |
| `entity -> co_occurs -> entity` | 280.086 | Entidades aparecem juntas em mensagens |
| `entity -> rev_co_occurs -> entity` | 280.086 | Reverso da coocorrencia |
| `day -> next -> day` | 1.224 | Sequencia temporal entre snapshots consecutivos |

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
| `url` | 78.609 |
| `cve` | 31.551 |
| `domain` | 3.712 |
| `hash` | 2.720 |
| `handle` | 177 |
| `victim` | 169 |
| `ip_address` | 247 |
| `email` | 36 |
| `malware` | 21 |
| `country` | 20 |
| `vulnerability` | 20 |
| `product` | 19 |
| `sector` | 14 |
| `threat_actor` | 12 |
| `tool` | 12 |
| `ttp` | 12 |
| `wallet` | 3 |

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
| `Cyber Crime` | 815 |
| `Cyber Espionage` | 473 |
| `Hacktivism` | 251 |
| `Cyber Warfare` | 159 |
| `Unknown` | 271 |

Tipos de ameaca usados (apos agrupamento):

| Tipo | Snapshots positivos (treino / val / teste) |
|---|---:|
| `other_cyber` | 473 / 108 / 133 |
| `malware` | 468 / 90 / 129 |
| `rare_threat` | 423 / 83 / 52 |
| `account_takeover` | 305 / 50 / 92 |
| `ransomware` | 126 / 83 / 93 |

Os 17 tipos com suporte baixo ou nulo foram agrupados em `rare_threat`:
`botnet_or_c2`, `credential_theft`, `data_breach_or_leak`, `ddos_or_disruption`, `disinformation_or_influence`, `exfiltration`, `fraud_or_scam`, `initial_access_activity`, `insider_threat`, `lateral_movement`, `phishing`, `physical_or_hybrid_threat`, `privilege_escalation`, `reconnaissance`, `supply_chain_compromise`, `vulnerability_or_exploit`, `wiper_or_destruction`.

## Split de treino, validacao e teste

O split e cronologico e balanceado o suficiente para que cada particao tenha mais de uma classe de bucket. A politica usada foi `balanced_chronological`.

| Split | Snapshots | Inicio | Fim | Media de eventos | `0` | `1-2` | `3-5` | `6-10` | `11-15` | `16+` |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| Treino | 573 | 2023-01-01 | 2024-07-26 | 10,04 | 26 | 73 | 76 | 124 | 157 | 117 |
| Validacao | 131 | 2024-07-27 | 2024-12-04 | 8,04 | 8 | 24 | 9 | 51 | 27 | 12 |
| Teste | 175 | 2024-12-05 | 2026-03-30 | 6,50 | 12 | 35 | 35 | 58 | 28 | 7 |

Note que existe uma queda na media de eventos do treino para o teste. Isso indica mudanca temporal na distribuicao do alvo e torna o teste mais realista.

## Parametros do treino atual

| Parametro | Valor |
|---|---:|
| Epocas | 120 |
| Dispositivo | `cuda` |
| `hidden_dim` | 96 |
| Camadas HGT | 2 |
| Heads | 2 |
| Dropout | 0,3 |
| Learning rate | 0,001 |
| Weight decay | 0,0001 |
| Seed | 22 |

O historico mostra reducao consistente da loss:

| Epoca | Loss | Macro-F1 validacao bucket |
|---:|---:|---:|
| 1 | 2,0244 | 0,0836 |
| 10 | 1,9163 | 0,0570 |
| 20 | 1,8195 | 0,1855 |
| 40 | 1,4055 | 0,2791 |
| 60 | 1,2950 | 0,2910 |
| 80 | 1,2559 | 0,2990 |
| 100 | 1,2375 | 0,2826 |
| 120 | 1,2483 | 0,2990 |

## Resultados

### Bucket de contagem

| Split | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Treino | 0,5096 | 0,5119 | 0,5022 |
| Validacao | 0,4733 | 0,2990 | 0,4436 |
| Teste | 0,4000 | 0,2463 | 0,3409 |

No teste, o modelo supera os baselines registrados:

| Metodo | Accuracy teste | Macro-F1 teste |
|---|---:|---:|
| Modelo HGT+GRU | 0,4000 | 0,2463 |
| Bucket majoritario do treino | 0,1600 | 0,0460 |
| Bucket do dia anterior | 0,2800 | 0,2387 |

A matriz de confusao de teste mostra que o modelo aprende melhor a regiao central da distribuicao, especialmente `1-2` e `6-10`. As classes extremas `0`, `3-5` e `16+` continuam mais instaveis.

Predicoes de bucket no teste:

| Bucket | Verdadeiro | Predito |
|---|---:|---:|
| `0` | 12 | 0 |
| `1-2` | 35 | 50 |
| `3-5` | 35 | 10 |
| `6-10` | 58 | 62 |
| `11-15` | 28 | 53 |
| `16+` | 7 | 0 |

### Regressao de contagem

| Split | MAE | RMSE | Media real | Media predita |
|---|---:|---:|---:|---:|
| Treino | 3,17 | 4,36 | 10,04 | 10,56 |
| Validacao | 2,88 | 3,98 | 8,04 | 7,34 |
| Teste | 3,46 | 4,59 | 6,50 | 8,40 |

No teste, a media predita (8,40) ficou acima da real (6,50). Isso sugere que o modelo ainda carrega parte do regime de maior volume visto no treino.

### Tipos de ameaca multilabel

| Split | Micro-F1 | Macro-F1 |
|---|---:|---:|
| Treino | 0,8550 | 0,8393 |
| Validacao | 0,8481 | 0,8389 |
| Teste | 0,7581 | 0,7391 |

No teste, resultado por tipo:

| Tipo | Suporte | Predito | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|---:|---:|
| `malware` | 129 | 128 | 0,9063 | 0,8992 | 0,9027 | 0,9064 |
| `other_cyber` | 133 | 125 | 0,8720 | 0,8195 | 0,8450 | 0,8925 |
| `account_takeover` | 92 | 125 | 0,6800 | 0,9239 | 0,7834 | 0,6973 |
| `ransomware` | 93 | 106 | 0,5943 | 0,6774 | 0,6332 | 0,6151 |
| `rare_threat` | 52 | 125 | 0,3760 | 0,9038 | 0,5311 | 0,5030 |

O agrupamento em `rare_threat` melhorou expressivamente o macro-F1 multilabel (de 0,19 para 0,74 no teste), eliminando as 17 classes impossíveis de prever. O `rare_threat` ainda tem precision baixa (0,38) porque o modelo dispara com frequencia, mas o recall alto (0,90) indica que ao menos captura os dias em que algum tipo raro ocorre.

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
- O `rare_threat` tem recall alto mas precision baixa: o modelo aprende que "algo raro pode acontecer" mas nao distingue qual tipo especificamente.
- A pasta se chama `temporal_graph`, provavelmente por typo de `temporal_graph`; os scripts funcionam com o nome atual.

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
