## Configuração inicial

```bash
python3 -m venv venv
source venv/bin/activate
```
e
```bash
python3 -m pip install -r requirements.txt
```

Baixar o modelo de LLM [aqui](https://huggingface.co/mrmage/Phi-3.5-mini-instruct-Q4_K_M-GGUF/tree/main)

obs: o arquivo a ser baixado é o que contem a extensão `.gguf`

## Iniciar rapidamente o programa
### MOCKUP: 
```bash
python3 run_game.py --force-mock
```
### Usando modelo:
```bash
python3 run_game.py --model ~/Downloads/phi-3.5-mini-instruct-Q4_K_M.gguf
```

## Arrumando e vendo os resultados
```bash
python3 render_log_readable.py logs/partida_xxx.json
```

## O que alterar?
Pelo que entendi o foco aqui vai ser modificar o arquivo `llm_agent.py`

NAO MEXER NA INTERFACE DE COMUNICAÇÃO COM O PROTOCOLO!

mas o funcionamnto interno pode :)

## Ideias
Os agentes aleatórios são burros (toda ação deles é literalmente um rand) e pelas regras do jogo, vi que casos improváveis são penalizados com o ganho de nenhum ponto (para se todos votarem na sua carta sendo o narrador ou nenhum voto na carta do narrador), Por isso a estratégia adotada tenta equilibrar ambiguidade e associação temática

### Como Narrador
A escolha da carta não é aleatória.

Primeiro a mão é compactada utilizando apenas:
- título
- artista
- trecho representativo (normalmente o refrão)

Essas informações são enviadas para a LLM, que recebe a tarefa de escolher a música mais ambígua da mão.

**Prompt:** 
```python 
SYSTEM_INSTRUCTION + 
        "Escolha a música mais AMBÍGUA da mão.\n" 
        "Uma música ambígua possui múltiplas interpretações possíveis.\n"
        "Evite músicas com tema muito óbvio.\n"
        "Evite músicas extremamente específicas.\n\nSuas cartas: {hand_json}\n\n"
        "Responda no formato: {{\"chosen_id\": <id>}}"
```

Caso a LLM falhe, o agente calcula uma medida de ambiguidade baseada na similaridade entre as músicas da mão e escolhe a carta mais "genérica" semanticamente.

### Geração da Dica
Após escolher a música, o agente tenta extrair um trecho representativo da letra (preferencialmente um refrão detectado automaticamente).

Esse trecho é enviado para a LLM, que deve produzir uma dica curta e indireta.

Restrições:

não copiar a letra
não usar palavras do título
ser metafórica
possuir no máximo 6 palavras

**Prompt:**
```python
SYSTEM_INSTRUCTION +
        "Objetivo:\n"
        "Criar uma dica que faça alguns jogadores acertarem e alguns errarem.\n\n"
        "Regras:\n"
        "- nao copie trechos da letra\n"
        "- nao use palavras do titulo\n"
        "- seja indireto\n"
        "- seja metaforico\n"
        "- máximo 6 palavras\n"
        "- evite temas genéricos como amor, vida, felicidade, tristeza\n\n"
        "Letra: {short_lyrics}"
        "Responda apenas: {{\"dica\": \"<frase>\"}}"
```
Depois da geração, filtros adicionais verificam se a dica:
- contém palavras do título
- copia trechos da letra
- viola o limite de tamanho

Se necessário, mecanismos de fallback geram automaticamente uma dica válida.

### Como melômano (carta isca)
Ao receber uma dica, o agente tenta escolher uma música da própria mão que possa ser confundida com a música do narrador.

Para isso, cada carta recebe uma pontuação baseada em quão relacionada ela parece estar com a dica.

Primeiro é calculado um ranking heurístico usando:
- palavras iguais entre a dica e a música (similaridade lexical)
- palavras parecidas ou com partes em comum (similaridade por trigramas)
- semelhança geral entre os textos, mesmo quando não usam exatamente as mesmas palavras (similaridade vetorial por cosseno)

Onde são analisados:

- o título da música;
- o artista;
- um trecho representativo da letra (normalmente o refrão).

Após essa análise, a LLM recebe as cartas resumidas e escolhe aquela que considera mais compatível com a dica.

**Prompt:**
```json
SYSTEM_INSTRUCTION +
        "Selecione a 'Carta Isca' com maior similaridade tematica com a dica.\n"
        "Dica: \"{clue}\"\nSuas cartas: {hand_json}\n\n"
        "Responda no formato: {{\"chosen_id\": <id>}}"
```

Caso a LLM falhe, o agente utiliza o ranking heurístico.

### Votando
A votação atualmente é totalmente heurística.

Para cada carta candidata é calculado um score baseado em:
- título
- artista
- trecho representativo da letra

O score combina:
- correspondência de palavras
- semelhança parcial de palavras
- similaridade vetorial por cosseno

Para realizar essa comparação, são utilizados:
- o título da música;
- o artista;
- um trecho representativo da letra.

Após calcular as pontuações, as cartas são ordenadas da mais compatível para a menos compatível com a dica.

As duas cartas com maior pontuação são escolhidas como voto.