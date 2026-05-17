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
python3 run_game.py --model ~/Downloads/Phi-3.5-mini-instruct-Q4_K_M.gguf
```

## Arrumando e vendo os resultados
```bash
python3 render_log_readable.py logs/partida_xxx.json
```

## O que alterar?
Pelo que entendi o foco aqui vai ser modificar o arquivo `llm_agent.py`

NAO MEXER NA INTERFACE DE COMUNICAÇÃO COM O PROTOCOLO!

mas o funcionamnto interno pode

## Ideias
Os agentes aleatórios são burros (toda ação deles é literalmente um rand) e pelas regras do jogo, vi que casos improváveis são penalizados com o ganho de nenhum ponto (para se todos votarem na sua carta sendo o narrador ou nenhum voto na carta do narrador), então o prompt não precisa ser perfeitamente elaborado. Como Narrador eu pensei em montar a dica com base num tema da música, e evitar usar palavras presentes no título da música
**Prompt:** `Música: [letra]. Tema central (uma palavra): ? Dicas: ? (6 palavras).`
**Resultado esperado:** `"Tema: Saudade | Dica: O tempo não apaga sua falta"`
Já como melônamo, pensei em enviar no prompt apenas o título da música junto com um trecho que se repete da música para ela analisar junto com a dica dada
**Prompt:** `Dada a dica [X] e estas 6 músicas (titulo + trecho repetido), retorne apenas os indices de duas músicas que melhor se encaixam na dica. Responda em JSON: {'votes': [idx1,idx2]}`
e para tratar o envio da carta, mandar as cartas da mão e pedir para ela escolher uma que se pareça com a dica do narrador (mesmo não sendo a dele) pra enganar os outros agentes e tentar ganhar votos.
**Prompt:** `Dica: "{clue}". Minha mão (título + trecho): {resumo_mao}. Qual música melhor se associa a dica para enganar os adversários e atrair votos para mim? Responda em JSON: {{"chosen_id": <id>}}`