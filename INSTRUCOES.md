# Nota Secreta — solução de referência comentada

Este projeto contém uma **versão comentada e simplificada** do jogo **Nota Secreta**,
usada como base para a implementação do agente estratégico da disciplina.

A ideia é que você possa:

- entender a arquitetura do sistema;
- rodar partidas localmente;
- testar seu agente em modo mock ou com um modelo real;
- modificar principalmente `llm_agent.py` e, se desejar, `base_agent.py`.

---

## 1. Visão geral da arquitetura

O projeto combina dois estilos de comunicação:

- **REST/FastAPI** entre os agentes e o serviço LLM centralizado (`llm_service.py`);
- **A2A / JSON-RPC** entre o Game Master e os agentes.

Em uma execução típica:

1. o `run_game.py` sobe o serviço LLM;
2. sobe o `game_master.py`;
3. sobe 1 agente estratégico e 5 agentes aleatórios;
4. registra os agentes no Game Master;
5. executa uma partida completa;
6. salva um log da partida em `logs/`.

---

## 2. Estrutura dos arquivos

Arquivos principais:

- `fasta2a.py`: mini-implementação de `A2AApp` e `@tool`
- `base_agent.py`: utilidades comuns para agentes
- `llm_service.py`: serviço LLM centralizado (real ou mock)
- `game_master.py`: coordenação da partida, votação, pontuação e logs
- `llm_agent.py`: agente estratégico a ser estudado e modificado
- `random_agent.py`: baseline aleatório
- `run_game.py`: sobe tudo e executa uma partida completa
- `render_log_readable.py`: transforma logs em uma visualização mais legível
- `brazilian_songs.csv`: base de músicas usada pelo jogo
- `tests/`: testes auxiliares

---

## 3. O que você deve modificar

Em geral, os arquivos mais importantes para o aluno são:

- `llm_agent.py`
- `base_agent.py` (opcional)

Você pode reorganizar a lógica interna do agente, desde que preserve a interface esperada
pelo restante da infraestrutura.

As ferramentas (tools) esperadas do agente são:

- `receive_hand(hand)`
- `choose_card()`
- `send_clue(lyrics, max_words=6)`
- `select_card_by_clue(clue)`
- `vote(clue, options, my_chosen_card)`

---

## 4. Instalação

Crie e ative um ambiente virtual:

```bash
python3 -m venv venv
source venv/bin/activate
```

Instale as dependências:

```bash
python3 -m pip install -r requirements.txt
```

---

## 5. Execução rápida

### 5.1. Rodar em modo mock

Esse modo não usa um modelo real e é útil para validar rapidamente a arquitetura:

```bash
python3 run_game.py --force-mock
```

### 5.2. Rodar com um modelo GGUF real

```bash
python3 run_game.py --model /caminho/do/modelo.gguf
```

Exemplo:

```bash
python3 run_game.py --model ~/Documentos/LLM/Phi-3.5-mini-instruct-Q4_K_M.gguf
```

---

## 6. Opções úteis do `run_game.py`

### Subir 6 agentes estratégicos

```bash
python3 run_game.py --all-strategic --force-mock
```

ou:

```bash
python3 run_game.py --all-strategic --model /caminho/do/modelo.gguf
```

### Alterar a base de músicas

```bash
python3 run_game.py --db outra_base.csv --force-mock
```

### Ajustar concorrência do serviço LLM

```bash
python3 run_game.py --model /caminho/do/modelo.gguf --llm-max-concurrency 1
```

---

## 7. Logs

Ao final da partida, o Game Master salva um log JSON em:

```text
logs/
```

O caminho do log também é mostrado no terminal ao fim da execução.

Esses logs ajudam a entender:

- qual agente foi narrador em cada rodada;
- qual dica foi produzida;
- quais cartas foram jogadas;
- como os votos foram distribuídos;
- como a pontuação evoluiu ao longo da partida.

---

## 8. Como ler os logs

Para transformar um log em uma visualização mais legível:

```bash
python3 render_log_readable.py logs/partida_xxx.json
```

---

## 9. Observações sobre a base de músicas

A base CSV deve conter, no mínimo, as colunas:

- `id`
- `title`
- `artist`
- `lyrics`

A base fornecida aqui serve para testes e desenvolvimento local.
Na avaliação, vai ser usada uma base oficial definida pelo professor.

---

## 10. Objetivo pedagógico

O foco deste trabalho não é apenas “fazer um agente funcionar”, mas construir
um **sistema multiagente baseado em LLM**.

Por isso, espera-se que o agente:

- use a LLM para decisões semânticas;
- lide com respostas imperfeitas de forma robusta;
- preserve o protocolo esperado pela infraestrutura.

Em outras palavras:

> a implementação interna pode variar, mas a interface externa do agente deve continuar compatível.

---

## 11. Resumo

Use esta versão do projeto para:

- entender a arquitetura;
- rodar testes locais;
- modificar o agente estratégico;
- experimentar diferentes prompts e estratégias.

Fluxo mínimo recomendado:

1. rodar `python3 run_game.py --force-mock`
2. rodar `python3 run_game.py --model ...`
3. inspecionar os logs
4. modificar `llm_agent.py`
5. repetir os testes
