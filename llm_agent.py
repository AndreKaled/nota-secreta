from __future__ import annotations

import argparse
import json
import re
import asyncio
import logging
from typing import Any, Dict, List
from collections import Counter
import math

from base_agent import BaseAgent, STOPWORDS
from fasta2a import A2AApp, tool

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = A2AApp(name="LLMAgent")


class LLMAgent(BaseAgent):
    """
    Agente híbrido baseado em LLM + heurísticas.
    Estratégia:
    1. Escolha da carta:
       - LLM escolhe a música mais ambígua na mão (propositalmente eu julguei ser a melhor estratégia).
       - fallback baseado em similaridade média da mão.

    2. Geração de dica:
       - LLM produz dica metafórica.
       - filtros removem cópia literal de título e letra.
       - fallback gera dica automaticamente.

    3. Seleção de carta isca:
       - LLM tenta escolher a melhor correspondência.
       - fallback usa score heurístico.

    4. Votação:
       - ranking heurístico baseado em similaridade.
    """
    # HIPERPARÂMETROS DA LLM
    TEMP_NARRADOR = 0.3
    TEMP_MELOMANO = 0.1
    MAX_TOKENS_CHOOSE = 25
    MAX_TOKENS_CLUE = 35
    MAX_TOKENS_SELECT = 25
    MAX_TOKENS_VOTE = 20
    STOP_CRITERIA = ["###"]
    TIMEOUT = 60.0

    # TEMPLATES DE PROMPTS
    SYSTEM_INSTRUCTION = (
        "Responda SOMENTE JSON. Nenhum texto antes. Nenhum texto depois."
    )
    PROMPT_CHOOSE_CARD = (
        SYSTEM_INSTRUCTION +
        "Escolha a música mais AMBÍGUA da mão.\n"
        "Uma música ambígua possui múltiplas interpretações possíveis.\n"
        "Evite músicas com tema muito óbvio.\n"
        "Evite músicas extremamente específicas.\n\nSuas cartas: {hand_json}\n\n"
        "Responda no formato: {{\"chosen_id\": <id>}}"
    )
    PROMPT_SEND_CLUE = (
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
    )
    PROMPT_SELECT_CARD = (
        SYSTEM_INSTRUCTION +
        "Selecione a 'Carta Isca' com maior similaridade tematica com a dica.\n"
        "Dica: \"{clue}\"\nSuas cartas: {hand_json}\n\n"
        "Responda no formato: {{\"chosen_id\": <id>}}"
    )

    def __init__(self, name: str, llm_url: str):
        """Inicializa o agente mapeando a infraestrutura base"""
        super().__init__(name=name, llm_url=llm_url, request_timeout=60.0)
        self.hand: List[Dict[str, Any]] = []

    @tool()
    async def receive_hand(self, hand: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Tool 1: Sincroniza e armazena o estado atual das cartas na mão do agente."""
        self.hand = list(hand)
        logger.info(f"[receive_hand] Recebidas {len(hand)} cartas")
        return {"status": "ok", "hand_size": len(self.hand)}

    @tool()
    async def choose_card(self) -> Dict[str, Any]:
        """
        Tool 2: Determina a carta do Narrador.
        A LLM tenta selecionar a música mais ambígua da mão.
        Se a resposta for inválida, utiliza uma heurística baseada
        na similaridade entre as músicas disponíveis.
        """
        if not self.hand:
            raise RuntimeError("Hand is empty")

        # Reduz o tamanho das cartas antes de enviar para a LLM
        compact_hand = self._compact_hand(self.hand)

        # Montagem do prompt
        prompt = self.PROMPT_CHOOSE_CARD.format(
            hand_json=json.dumps(compact_hand, ensure_ascii=False)
        )
        logger.info(f"[choose_card] Avaliando {len(compact_hand)} cartas")
        raw = await self._query_llm(
            prompt,
            self.MAX_TOKENS_CHOOSE,
            self.TEMP_NARRADOR,
        )
        logger.info(f"[choose_card] Resposta LLM: {raw}")

        # Extrai o id mesmo se a resposta vier fora do formato esperado
        chosen_id = self._safe_extract(raw, "chosen_id")
        logger.info(f"[choose_card] chosen_id extraído: {chosen_id}")
        chosen_card = self._get_song_by_id(chosen_id)

        # Fallback se a LLM retornar lixo ou der timeout
        if chosen_card is None:
            if not self.hand:
                return {"chosen_card": {}}

            chosen_card = max(self.hand,key=self._ambiguity_score)
            logger.warning(f"[choose_card] Fallback acionado. Carta escolhida: {chosen_card['id']}")

        logger.info(f"[choose_card] Carta escolhida: {chosen_card['title']} ({chosen_card['id']})")
        return {"chosen_card": chosen_card}

    @tool()
    async def send_clue(self, lyrics: str, max_words: int = 6) -> Dict[str, Any]:
        """
        Tool 3: Gera uma dica para a música escolhida.

        A dica é validada para evitar palavras do título,
        trechos da letra e excesso de palavras.
        """
        # Localiza o objeto da música no estado local do agente
        song = self._find_song_by_lyrics(lyrics)
        title = song.get("title", "") if song else ""
        logger.info(f"[send_clue] Música encontrada: {title}")

        # cria uma lista negra de palavras proibidas (título da música limpo de stopwords)
        title_clean = re.sub(r"[^\w\s]", "", title.lower())
        forbidden = {w for w in title_clean.split() if w not in STOPWORDS}

        # usa o refrão como trecho representativo da música
        refrao = self._extract_refrao(lyrics)

        # reduz a quantidade de texto enviada para a LLM
        # para diminuir custo e ruído na geração
        if refrao == lyrics:
            short_lyrics = " ".join(lyrics.split()[:25])
        else:
            short_lyrics = " ".join(refrao.split()[:25])   

        prompt = self.PROMPT_SEND_CLUE.format(short_lyrics=short_lyrics)
        logger.debug(f"[send_clue] Trecho enviado: {short_lyrics[:150]}")

        raw = await self._query_llm(
            prompt,
            self.MAX_TOKENS_CLUE,
            self.TEMP_NARRADOR,
        )
        logger.info(f"[send_clue] Resposta LLM: {raw}")

        # Tenta extrair a dica do JSON estruturado
        data = self._extract_json(raw)
        clue = data.get("dica", "")
        logger.debug(f"[send_clue] JSON extraído: {data}")
        
        # tenta limpar lixo caso o formato json falhe mas exista algum texto gerado
        if clue:
            clue = clue.strip()
        else:
            clue = self._sanitize_clue(raw.strip(),max_words=max_words)

        # Filtro de Obviedade
        if clue:
            clue_words = set()
            for w in clue.lower().split():
                wc = "".join(ch for ch in w if ch.isalnum())
                if wc and wc not in STOPWORDS:
                    clue_words.add(wc)

            # bloqueia vazamento de palavras que tem no título
            if clue_words & forbidden:
                clue = ""
                logger.info("[send_clue] Dica descartada por conter palavras do título")
            # bloqueia cópia literal da letra original
            elif self._is_literal_substring_of_lyrics(clue, lyrics):
                clue = ""
                logger.info("[send_clue] Dica descartada por copiar trecho da letra")

        # força truncamento para 6 palavras
        if clue:
            words = clue.split()
            if len(words) > max_words:
                clue = " ".join(words[:max_words])

        # Fallback
        if not clue:
            tokens = title_clean.split()
            first = tokens[0] if tokens else ""
            # Mapeamento fixo de sinônimos abstratos para cobrir termos comuns das músicas
            synonyms = {
                "amor": "paixao intensa", "saudade": "nostalgia profunda",
                "sol": "luz radiante", "mar": "oceano infinito",
                "ceu": "infinito azul", "noite": "escuridao serena",
                "vida": "jornada eterna", "morte": "sono eterno",
                "casa": "lar doce", "rua": "caminho aberto",
                "cidade": "caos urbano", "rio": "aguas correntes",
                "festa": "alegria contagiante", "samba": "ritmo brasileiro",
                "medo": "sensacao fria", "paz": "calma serena",
                "flor": "beleza natural", "coracao": "sentimento puro",
            }
            clue = synonyms.get(first, "")
            # se nada der certo tenta extrair cegamente radicais da própria letra
            if not clue:
                clue = self._fallback_clue_from_lyrics(lyrics, max_words)
            logger.warning(f"[send_clue] Fallback acionado -> {clue}")
        
        logger.info(f"[send_clue] Dica final: {clue}")
        return {"clue": clue}
   
    @tool()
    async def select_card_by_clue(self, clue: str) -> Dict[str, Any]:
        """
        Tool 4: Escolhe uma carta da mão para combinar com a dica.

        Primeiro gera um ranking usando medidas de similaridade.
        Depois consulta a LLM para escolher a carta.

        Se a LLM falhar, utiliza a melhor carta do ranking.
        """
        if not self.hand:
            raise RuntimeError("Hand is empty")

        # Pré-processa e rankea a mão inteira localmente
        ranked = []
        for song in self.hand:
            score = self._candidate_score(clue, song)
            ranked.append((score, song))

            logger.debug(
                f"[select_card_by_clue] "
                f"{song['title']} -> score={score}"
            )

        # ordena do maior score para o menor
        ranked.sort(key=lambda x: x[0], reverse=True)
        logger.info(
            f"[select_card_by_clue] "
            f"Analisando {len(self.hand)} cartas completas"
        )

        compact_hand = self._compact_hand(self.hand)

        prompt = self.PROMPT_SELECT_CARD.format(
            clue=clue,
            hand_json=json.dumps(compact_hand, ensure_ascii=False)
        )

        raw = await self._query_llm(
            prompt,
            self.MAX_TOKENS_SELECT,
            self.TEMP_MELOMANO,
        )

        logger.info(f"[select_card_by_clue] Resposta LLM: {raw}")

        chosen_id = self._safe_extract(raw, "chosen_id")
        chosen_card = self._get_song_by_id(chosen_id)
        # Retorna a escolha da LLM se válida, se nao der, aciona o topo do ranking local
        if chosen_card is not None:
            return {"chosen_card": chosen_card}

        logger.warning("[select_card_by_clue] Fallback heurístico acionado")
        return {"chosen_card": ranked[0][1]}

    @tool()
    async def vote(self, clue: str, options: List[Dict[str, Any]], my_chosen_card: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool 5: Escolhe em quais cartas votar.

        A votação é feita apenas com heurísticas de similaridade,
        sem utilizar a LLM.
        """
        # Isola o índice da própria carta jogada na mesa para evitar autovotação (seria bem burro fazer isso)
        my_idx = next(i for i, option in enumerate(options) if option["id"] == my_chosen_card["id"])
        logger.info(f"[vote] Votando para dica: {clue}")
        scored = []
        for idx, option in enumerate(options):
            if idx == my_idx:
                continue # pula a propria carta
            score = self._candidate_score(clue,option)
            scored.append((score, idx))
            logger.debug(f"[vote] {option['title']} -> {score}")
        
        scored.sort(reverse=True)
        # Isola os índices originais das duas faixas com os maiores scores semânticos
        votes = [
            idx
            for _, idx in scored[:2]
            ]
        logger.info(f"[vote] Votos finais: {votes}")
        return {"votes": votes[:2]}

    # MÉTODOS AUXILIARES DE SUPORTE
    def _find_song_by_lyrics(self, lyrics: str) -> dict | None:
        """Procura uma música pela letra."""
        for song in self.hand:
            if song.get("lyrics", "") == lyrics:
                return song
        return None
    
    def _is_literal_substring_of_lyrics(self, clue: str, lyrics: str) -> bool:
        """Verifica se a dica gerada é um plágio/cópia direta da letra"""
        clue_clean = re.sub(r"[^\w\s]", "", clue.lower()).strip()
        lyrics_clean = re.sub(r"[^\w\s]", "", lyrics.lower()).strip()
        
        if not clue_clean:
            return False
        return clue_clean in lyrics_clean
    
    def _sanitize_clue(self, raw_text: str, max_words: int) -> str:
        """Remove caracteres de JSON quando a resposta não está formatada corretamente."""
         # remove caracteres de formatação que sobraram
        clean_text = re.sub(r'[\{\}\"\'\:\,]', '', raw_text)
        
        # pega a linha mais longa ou a primeira linha populada
        lines = [line.strip() for line in clean_text.split('\n') if line.strip()]
        if not lines:
            return ""
            
        candidate = lines[0]
        # aplica trava mecânica preemptiva de palavras
        words = candidate.split()
        return " ".join(words[:max_words])
    
    def _fallback_clue_from_lyrics(self, lyrics: str, max_words: int) -> str:
        """Varre a letra, remove stopwords e fatia as primeiras palavras como dica de emergência."""
        words = [w for w in lyrics.split() if w not in STOPWORDS]
        if not words:
            words = lyrics.split()
        return " ".join(words[:max_words])

    def _safe_extract(self, raw: str, key: str, default: Any = None) -> Any:
        """
        Extrator que tenta resgatar chaves em strings JSON.
        Primeiro tenta decodificar nativamente; se houver quebra ou truncamento, 
        faz uma busca via expressões regulares.
        """
        try:
            # Se vier um JSON limpo, usa ele (o melhor cenário)
            data = json.loads(raw)
            return data.get(key, default)
        except:
            pass

        # Se o JSON está quebrado ou sujo, caça o valor específico na string bruta
        if key == "chosen_id":
            # Procura por qualquer número seguido de chave de fechamento ou fim de linha
            m = re.search(r'"chosen_id"\s*:\s*(\d+)', raw)
            if m: return int(m.group(1))
            m = re.search(r'\b(\d+)\b', raw) # Pega o primeiro número isolado que encontrar
            if m: return int(m.group(0))

        elif key == "votos":
            # Procura o padrão de lista [x, y]
            m = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*\]', raw)
            if m: return [int(m.group(1)), int(m.group(2))]
        
        elif key == "dica":
            m = re.search(r'"dica"\s*:\s*"([^"]+)"', raw)
            if m: return m.group(1)
            
        return default

    def _normalize_line(self, line: str) -> str:
        """Remove acentuações, pontuações e excesso de espaços"""
        line = line.lower()
        line = re.sub(r"[^\w\s]", "", line)
        line = re.sub(r"\s+", " ", line)
        return line.strip()

    def _normalize_words(self, text: str) -> set[str]:
        """Converte um texto em um conjunto de palavras normalizadas."""
        return set(re.findall(r"\w+", text.lower()))
    
    async def _query_llm(self,prompt: str,max_tokens: int,temperature: float,) -> str:
        """
        Função que executa as requisições para a LLM, para evitar 
        reescrever o mesmo código 4 vezes diferentes e sem diferença 
        prática além das variáveis
        """
        try:
            raw = await asyncio.wait_for(
                self.llm_generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=self.STOP_CRITERIA,
                ),
                timeout=self.TIMEOUT,
            )
            return raw
        except (asyncio.TimeoutError, Exception) as e:
            logger.error(f"[LLM] erro: {e}")
            return ""
        
    def _compact_song(self,song: Dict[str, Any],words: int = 40,) -> Dict[str, Any]:
        """Reduz o tamanho dos dados textuais de uma música mantendo os metadados e o núcleo do refrão."""
        refrao = self._extract_refrao(song.get("lyrics", ""))

        trecho = " ".join(
            refrao.split()[:words]
        )

        return {
            "id": song["id"],
            "titulo": song.get("title", ""),
            "artista": song.get("artist", ""),
            "trecho": trecho,
        }
    
    def _compact_hand(self,hand: List[Dict[str, Any]],words: int = 40,) -> List[Dict[str, Any]]:
        """Compacta todas as cartas da mão"""
        return [
            self._compact_song(song, words)
            for song in hand
        ]
    
    def _get_song_by_id(self, song_id: int) -> Dict[str, Any] | None:
        """Retorna a música correspondente ao id dado"""
        for song in self.hand:
            if song["id"] == song_id:
                return song
        return None
    
    def _similarity_score(self,clue: str,text: str) -> int:
        """Algoritmo de correspondência exata: pontua cruzamentos diretos e correspondências parciais."""
        clue_words = self._normalize_words(clue)
        text_words = self._normalize_words(text)
        score = 0

        for word in clue_words:
            if word in text_words:
                score += 3 # Peso alto para termos idênticos
            for candidate in text_words:
                if word in candidate or candidate in word:
                    score += 1 # Existe uma interseção parcial

        return score
    
    def _extract_json(self, raw: str) -> dict:
        """Tenta decodificar o dicionário mapeando os delimitadores estruturais."""
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        m = re.search(r"\{[\s\S]*?\}", raw)
        if m:
            try:
                data = json.loads(m.group())
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

        return {}

    def _extract_refrao(self, lyrics: str) -> str:
        """
        Tenta identificar o refrão procurando linhas repetidas na letra.

        Caso nenhuma repetição relevante seja encontrada,
        retorna a letra original.
        """
        lines = [
            self._normalize_line(line)
            for line in lyrics.splitlines()
            if line.strip()
        ]

        counter = Counter(lines)

        repeated = [
            line
            for line, count in counter.items()
            if count >= 2 and len(line.split()) >= 3
        ]

        if repeated:
            repeated.sort(key=len, reverse=True)
            return repeated[0]

        return lyrics

    def _candidate_score(self,clue: str,song: Dict[str, Any]) -> float:
        """
        Calcula uma pontuação de compatibilidade entre uma dica
        e uma música.

        Combina correspondência de palavras e similaridade vetorial
        sobre título, artista e trecho da letra.
        """
        score = 0.0

        title = song.get("title","")
        artist = song.get("artist", "")
        refrao = self._extract_refrao(song.get("lyrics", ""))
        trecho = " ".join(refrao.split()[:25])

        # heuristica antiga
        score += self._similarity_score(clue, title) * 4
        score += self._similarity_score(clue, artist)
        score += self._similarity_score(clue, trecho) * 2

        # similaridade vetorial
        score += self._vector_similarity(clue, title) * 20
        score += self._vector_similarity(clue, artist) * 5
        score += self._vector_similarity(clue, trecho) * 40

        return score

    def _text_to_vector(self, text: str) -> Counter:
        """Vetoriza strings espalhando os termos em tokens puros e janelas deslizantes de 3 caracteres."""
        words = re.findall(r"\w+", text.lower())
        features = []
        for word in words:
            if word in STOPWORDS:
                continue
            features.append(word)

            if len(word) >= 3:
                for i in range(len(word) - 2):
                    features.append(word[i:i + 3]) # Captura de radicais similares
        return Counter(features)

    def _calculate_cosine(self, vec1: Counter, vec2: Counter) -> float:
        """
        Calcula a similaridade por cosseno entre dois vetores.

        Quanto mais próximo de 1, mais parecidos os vetores são.
        Quanto mais próximo de 0, menos elementos eles compartilham.
        """
        intersection = set(vec1) & set(vec2)
        dot_product = sum(
            vec1[k] * vec2[k]
            for k in intersection
        )
        magnitude1 = math.sqrt(
            sum(v * v for v in vec1.values())
        )
        magnitude2 = math.sqrt(
            sum(v * v for v in vec2.values())
        )
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def _vector_similarity(self, clue: str, text: str) -> float:
        """Calcula a similaridade por cosseno entre dois textos."""
        clue_vec = self._text_to_vector(clue)
        text_vec = self._text_to_vector(text)
        return self._calculate_cosine(
            clue_vec,
            text_vec,
        )

    def _ambiguity_score(self, song: Dict[str, Any]) -> float:
        """
        Mede a ambiguidade de uma música calculando a mediana de sua proximidade 
        semântica em relação a todas as outras cartas presentes na mão atual.
        """

        refrao = self._extract_refrao(song.get("lyrics", ""))
        trecho = " ".join(refrao.split()[:25])
        similarities = []
        for other in self.hand:
            if other["id"] == song["id"]:
                continue
            other_refrao = self._extract_refrao(
                other.get("lyrics", "")
            )
            other_trecho = " ".join(
                other_refrao.split()[:25]
            )
            similarities.append(
                self._vector_similarity(
                    trecho,
                    other_trecho
                )
            )
        if not similarities:
            return 0.0
        
        similarities.sort()
        return similarities[len(similarities) // 2]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("game_master_url")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--llm-url", default="http://127.0.0.1:9000")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    agent = LLMAgent(name=args.name or f"LLMAgent_{args.port}", llm_url=args.llm_url)
    app.register(agent)
    app.run(host=args.host, port=args.port)

if __name__ == "__main__":
    main()