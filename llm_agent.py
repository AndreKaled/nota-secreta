from __future__ import annotations

import argparse
import json
import re
import asyncio
import logging
from typing import Any, Dict, List
from collections import Counter

from base_agent import BaseAgent, STOPWORDS
from fasta2a import A2AApp, tool

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = A2AApp(name="LLMAgent")


class LLMAgent(BaseAgent):
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
        "- seja metafórico\n"
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
    PROMPT_VOTE = (
        SYSTEM_INSTRUCTION +
        "Selecione os índices das DUAS musicas candidatas com maior probabilidade de serem a carta do Narrador.\n"
        "Dica: \"{clue}\"\nCandidatas: {candidates_json}\n\n"
        "Responda no formato: {{\"votos\": [<id1>, <id2>]}}"
    )

    def __init__(self, name: str, llm_url: str):
        super().__init__(name=name, llm_url=llm_url, request_timeout=60.0)
        self.hand: List[Dict[str, Any]] = []

    @tool()
    async def receive_hand(self, hand: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.hand = list(hand)
        logger.info(f"[receive_hand] Recebidas {len(hand)} cartas")
        return {"status": "ok", "hand_size": len(self.hand)}

    @tool()
    async def choose_card(self) -> Dict[str, Any]:
        if not self.hand:
            raise RuntimeError("Hand is empty")

        # Passo 1: Avaliação de Ambiguidade das Cartas
        compact_hand = self._compact_hand(self.hand)

        # Passo 2: Montagem do Prompt dinâmico (Chamada da LLM)
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

        # Passo 3: Interpolação e Captura Segura
        # ------------------------------------------------------------------     
        chosen_id = self._safe_extract(raw, "chosen_id")
        logger.info(f"[choose_card] chosen_id extraído: {chosen_id}")
        chosen_card = self._get_song_by_id(chosen_id)

        if chosen_card is None:
            # fallback
            if not self.hand:
                return {"chosen_card": {}}
            # --------------------------------------------------------------
            # Passo 5: Heurística de Fallback Determinística
            # --------------------------------------------------------------
            lengths = [len(song.get("lyrics", "")) for song in self.hand]
            ordered = sorted(lengths)
            median = ordered[len(ordered) // 2]

            best_idx = min(
                range(len(self.hand)),
                key=lambda i: abs(lengths[i] - median)
            )

            chosen_card = self.hand[best_idx]
            logger.warning(f"[choose_card] Fallback acionado. Carta escolhida: {chosen_card['id']}")

        logger.info(f"[choose_card] Carta escolhida: {chosen_card['title']} ({chosen_card['id']})")
        return {"chosen_card": chosen_card}

    @tool()
    async def send_clue(self, lyrics: str, max_words: int = 6) -> Dict[str, Any]:
        # Passo 1: Limpeza e Preparação dos Dados
        song = self._find_song_by_lyrics(lyrics)
        title = song.get("title", "") if song else ""
        logger.info(f"[send_clue] Música encontrada: {title}")

        title_clean = re.sub(r"[^\w\s]", "", title.lower())
        forbidden = {w for w in title_clean.split() if w not in STOPWORDS}

        # Passo 2: Extração Semântica Indireta (Chamada da LLM)
        refrao = self._extract_refrao(lyrics)

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

        if clue:
            clue = clue.strip()
        else:
            clue = self._sanitize_clue(raw.strip(),max_words=max_words)
        # Passo 3: Filtro de Obviedade
        if clue:
            clue_words = set()
            for w in clue.lower().split():
                wc = "".join(ch for ch in w if ch.isalnum())
                if wc and wc not in STOPWORDS:
                    clue_words.add(wc)

            if clue_words & forbidden:
                clue = ""
                logger.info("[send_clue] Dica descartada por conter palavras do título")
            elif self._is_literal_substring_of_lyrics(clue, lyrics):
                clue = ""
                logger.info("[send_clue] Dica descartada por copiar trecho da letra")

        # Passo 4: Ajuste de Restrição do Protocolo (max 6 palavras)
        if clue:
            words = clue.split()
            if len(words) > max_words:
                clue = " ".join(words[:max_words])

        # Passo 5: Mecanismo de Contingência (Fallback)
        if not clue:
            tokens = title_clean.split()
            first = tokens[0] if tokens else ""
            synonyms = {
                "amor": "paixao intensa", "saudade": "nostalgia profunda",
                "sol": "luz radiante", "mar": "oceano infinito",
                "ceu": "infinito azul", "noite": "escuridao serena",
                "vida": "jornada eterna", "morte": "sono eterno",
                "casa": "lar doce", "rua": "caminho aberto",
                "cidade": "caos urbano", "rio": "aguas correntes",
                "festa": "alegria contagiante", "samba": "ritmo brasileiro",
                "medo": "sensacao fria", "paz": "calma serena",
                "flor": "belez natural", "coracao": "sentimento puro",
            }
            clue = synonyms.get(first, "")
            if not clue:
                clue = self._fallback_clue_from_lyrics(lyrics, max_words)
            logger.warning(f"[send_clue] Fallback acionado -> {clue}")
        
        logger.info(f"[send_clue] Dica final: {clue}")
        return {"clue": clue}
   
    @tool()
    async def select_card_by_clue(self, clue: str) -> Dict[str, Any]:
        if not self.hand:
            raise RuntimeError("Hand is empty")

        ranked = []
        for song in self.hand:
            score = self._candidate_score(clue, song)
            ranked.append((score, song))

            logger.debug(
                f"[select_card_by_clue] "
                f"{song['title']} -> score={score}"
            )

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

        if chosen_card is not None:
            return {"chosen_card": chosen_card}

        logger.warning("[select_card_by_clue] Fallback heurístico acionado")
        return {"chosen_card": ranked[0][1]}

    @tool()
    async def vote(self, clue: str, options: List[Dict[str, Any]], my_chosen_card: Dict[str, Any]) -> Dict[str, Any]:
        my_idx = next(i for i, option in enumerate(options) if option["id"] == my_chosen_card["id"])
        logger.info(f"[vote] Votando para dica: {clue}")
        scored = []
        for idx, option in enumerate(options):
            if idx == my_idx:
                continue
            score = self._candidate_score(clue,option)
            scored.append((score, idx))
            logger.debug(f"[vote] {option['title']} -> {score}")
        scored.sort(reverse=True)
        votes = [
            idx
            for _, idx in scored[:2]
            ]
        logger.info(f"[vote] Votos finais: {votes}")
        return {"votes": votes[:2]}

    def _find_song_by_lyrics(self, lyrics: str) -> dict | None:
        for song in self.hand:
            if song.get("lyrics", "") == lyrics:
                return song
        return None
    
    def _is_literal_substring_of_lyrics(self, clue: str, lyrics: str) -> bool:
        """Verifica se a dica gerada é um pedaço literal copiado da letra."""
        clue_clean = re.sub(r"[^\w\s]", "", clue.lower()).strip()
        lyrics_clean = re.sub(r"[^\w\s]", "", lyrics.lower()).strip()
        
        if not clue_clean:
            return False
        return clue_clean in lyrics_clean
    
    def _sanitize_clue(self, raw_text: str, max_words: int) -> str:
        """Tenta extrair uma dica usável caso a LLM ignore o formato JSON."""
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
        """Extrai as primeiras palavras da letra como última opção de emergência."""
        words = [w for w in lyrics.split() if w not in STOPWORDS]
        if not words:
            words = lyrics.split()
        return " ".join(words[:max_words])

    def _safe_extract(self, raw: str, key: str, default: Any = None) -> Any:
        """Tenta extrair o valor da resposta bruta, reconstruindo o JSON se necessário."""
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

        elif key == "tema":
            m = re.search(r'"tema"\s*:\s*"([^"]+)"', raw)
            if m: return m.group(1)
        
        elif key == "dica":
            m = re.search(r'"dica"\s*:\s*"([^"]+)"', raw)
            if m: return m.group(1)
            
        return default

    def _normalize_line(self, line: str) -> str:
        line = line.lower()
        line = re.sub(r"[^\w\s]", "", line)
        line = re.sub(r"\s+", " ", line)
        return line.strip()

    def _normalize_words(self, text: str) -> set[str]:
        """normaliza palavras no texto e devolve como um conjunto"""
        return set(re.findall(r"\w+", text.lower()))
    
    async def _query_llm(self,prompt: str,max_tokens: int,temperature: float,) -> str:
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
        return [
            self._compact_song(song, words)
            for song in hand
        ]
    
    def _compact_candidates(self,options: List[Dict[str, Any]],excluded_idx: int) -> List[Dict[str, Any]]:
        candidates = []
        for i, opt in enumerate(options):
            if i == excluded_idx:
                continue

            candidates.append({
                "indice_original": i,
                "titulo": opt.get("title", ""),
                "trecho": " ".join(
                    opt.get("lyrics", "").split()[:30]
                ),
            })
        return candidates
    
    def _get_song_by_id(self, song_id: int) -> Dict[str, Any] | None:
        for song in self.hand:
            if song["id"] == song_id:
                return song
        return None
    
    def _similarity_score(self,clue: str,text: str) -> int:
        clue_words = self._normalize_words(clue)
        text_words = self._normalize_words(text)
        score = 0

        for word in clue_words:
            if word in text_words:
                score += 3
            for candidate in text_words:
                if word in candidate or candidate in word:
                    score += 1

        return score
    
    def _extract_json(self, raw: str) -> dict:
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

    def _candidate_score(self,clue: str,song: Dict[str, Any]) -> int:
        score = 0
        score += (self._similarity_score(clue,song.get("title", "")) * 4)
        score += (self._similarity_score(clue,song.get("artist", "")))
        refrao = self._extract_refrao(song.get("lyrics", ""))
        trecho = " ".join(refrao.split()[:25])
        score += (self._similarity_score(clue,trecho) * 2)
        return score

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