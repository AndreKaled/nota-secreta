from __future__ import annotations

"""Agente estratégico MEGA simples.
Marco Cristo, 2026

Objetivo desta versão:
- servir como ponto de partida;
- manter a interface esperada pela infraestrutura;
- ser funcional, para vcs terem um exemplo que roda.

Características:
- escolhe a carta do narrador por uma heurística muito simples;
- gera dica com a LLM, mas com prompt beeeem básico;
- escolhe carta e votos com regras ingênuas;
- não tenta otimizar de verdade para vencer o baseline aleatório.
"""

import argparse
import json
import random
import re
import asyncio
from typing import Any, Dict, List

from base_agent import BaseAgent, STOPWORDS
from fasta2a import A2AApp, tool

app = A2AApp(name="LLMAgent")


class LLMAgent(BaseAgent):
    # ----------------------------------------------------------------------
    # HIPERPARÂMETROS DA LLM
    # ----------------------------------------------------------------------
    TEMP_NARRADOR = 0.5
    TEMP_MELOMANO = 0.2
    
    MAX_TOKENS_CHOOSE = 35
    MAX_TOKENS_CLUE = 35
    MAX_TOKENS_SELECT = 35
    MAX_TOKENS_VOTE = 35
    
    STOP_CRITERIA = ["###"]

    TIMEOUT = 60.0

    # ----------------------------------------------------------------------
    # TEMPLATES DE PROMPTS
    # ----------------------------------------------------------------------
    SYSTEM_INSTRUCTION = (
        "Inicie sua resposta DIRETAMENTE com a chave '{{'. Não escreva introduções, "
        "explicações ou saudações como 'A resposta é:'. Responda APENAS o objeto JSON "
        "estruturado."
    )

    PROMPT_CHOOSE_CARD = (
        SYSTEM_INSTRUCTION +
        "Escolha uma musica da sua mao para ser a carta do Narrador.\n"
        "Criterio: Escolha a musica que permita gerar uma dica sutil "
        "— uma musica que possua metaforas ricas ou mais de um tema "
        "implicito, evitando titulos muito descritivos que entreguem "
        "a resposta imediatamente.\n\n"
        "Suas cartas: {hand_json}\n\n"
        "Responda APENAS com JSON no formato exato: "
        '{{"chosen_id": <id_da_carta_escolhida>}}'
    )

    PROMPT_SEND_CLUE = (
        SYSTEM_INSTRUCTION +
        "Identifique o sentimento/tema central da musica em UMA palavra. "
        "Depois gere uma frase curta (max 6 palavras) baseada "
        "apenas nesse tema abstrato, sem usar substantivos concretos da letra.\n"
        "Letra:\n{short_lyrics}\n\n"
        'Gere apenas o objeto JSON no formato exato: {{"tema": "palavra", "dica": "frase"}}'
    )

    PROMPT_SELECT_CARD = (
        SYSTEM_INSTRUCTION +
        "Voce esta em um jogo de associacao de musicas. "
        "A dica abaixo foi criada por outro jogador (o Narrador). "
        "Sua tarefa NAO e adivinhar a carta certa, mas sim selecionar "
        "da SUA propria mao a melhor 'Carta Isca'.\n\n"
        "Criterio: Escolha a carta que parecera a resposta correta "
        "para os outros jogadores confusos. Avalie qual musica "
        "possui maior similaridade tematica, metaforica ou por "
        "associacao direta com a dica recebida.\n\n"
        "Dica: \"{clue}\"\n\n"
        "Suas cartas: {hand_json}\n\n"
        "Gere apenas o objeto JSON no formato exato: "
        '{{"chosen_id": <id_da_carta>}}'
    )

    PROMPT_VOTE = (
        SYSTEM_INSTRUCTION +
        "Analise a dica do Narrador e as musicas candidatas abaixo.\n"
        "Selecione as DUAS opcoes distintas com maior probabilidade "
        "de terem gerado aquela dica.\n"
        "Procure conexoes tematicas, metaforas ou sinonimos "
        "entre a dica e os trechos/titulos das musicas.\n\n"
        "Dica: \"{clue}\"\n\n"
        "Candidatas: {candidates_json}\n\n"
        "Gere apenas o objeto JSON no formato exato: "
        '{{"votos": [indice_original_1, indice_original_2]}}'
    )

    def __init__(self, name: str, llm_url: str):
        super().__init__(name=name, llm_url=llm_url, request_timeout=60.0)
        self.hand: List[Dict[str, Any]] = []

    @tool()
    async def receive_hand(self, hand: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.hand = list(hand)
        return {"status": "ok", "hand_size": len(self.hand)}

    @tool()
    async def choose_card(self) -> Dict[str, Any]:
        if not self.hand:
            raise RuntimeError("Hand is empty")

        # ------------------------------------------------------------------
        # Passo 1: Avaliação de Ambiguidade das Cartas
        # ------------------------------------------------------------------
        compact_hand = [
            {
                "id": song["id"],
                "titulo": song.get("title", ""),
                "trecho": " ".join(song.get("lyrics", "").split()[:40]),
            }
            for song in self.hand
        ]

        # ------------------------------------------------------------------
        # Passo 2: Montagem do Prompt dinâmico (Chamada da LLM)
        # ------------------------------------------------------------------
        prompt = self.PROMPT_CHOOSE_CARD.format(
            hand_json=json.dumps(compact_hand, ensure_ascii=False)
        )

        try:
            raw = await asyncio.wait_for(
                self.llm_generate(prompt, 
                    max_tokens=self.MAX_TOKENS_CHOOSE, 
                    temperature=self.TEMP_NARRADOR, 
                    stop=self.STOP_CRITERIA
                ),
                timeout=self.TIMEOUT,
            )
        except (asyncio.TimeoutError, Exception):
            raw = ""

        # ------------------------------------------------------------------
        # Passo 3: Interpolação e Captura Segura
        # ------------------------------------------------------------------
        chosen_id = None
        if raw:
            try:
                data = json.loads(raw)
                if isinstance(data, dict):
                    chosen_id = data.get("chosen_id")
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if m:
                    try:
                        data = json.loads(m.group())
                        if isinstance(data, dict):
                            chosen_id = data.get("chosen_id")
                    except json.JSONDecodeError:
                        pass

        valid_ids = {song["id"] for song in self.hand}

        if chosen_id is not None and chosen_id in valid_ids:
            # --------------------------------------------------------------
            # Passo 4: Mapeamento de Protocolo
            # --------------------------------------------------------------
            chosen_card = next(s for s in self.hand if s["id"] == chosen_id)
        else:
            if not self.hand:
                return {"chosen_card": {}}
            # --------------------------------------------------------------
            # Passo 5: Heurística de Fallback Determinística
            # --------------------------------------------------------------
            lengths = [len(song.get("lyrics", "")) for song in self.hand]
            ordered = sorted(lengths)
            median = ordered[len(ordered) // 2]

            best_idx = 0
            best_dist = abs(lengths[0] - median)
            for i in range(1, len(self.hand)):
                dist = abs(lengths[i] - median)
                if dist < best_dist:
                    best_idx = i
                    best_dist = dist

            chosen_card = self.hand[best_idx]

        return {"chosen_card": chosen_card}

    @tool()
    async def send_clue(self, lyrics: str, max_words: int = 6) -> Dict[str, Any]:
        # ------------------------------------------------------------------
        # Passo 1: Limpeza e Preparação dos Dados
        # ------------------------------------------------------------------
        song = self._find_song_by_lyrics(lyrics)
        title = song.get("title", "") if song else ""

        title_clean = re.sub(r"[^\w\s]", "", title.lower())
        forbidden = {w for w in title_clean.split() if w not in STOPWORDS}

        # ------------------------------------------------------------------
        # Passo 2: Extração Semântica Indireta (Chamada da LLM)
        # ------------------------------------------------------------------
        short_lyrics = " ".join(lyrics.split()[:60])
        prompt = self.PROMPT_SEND_CLUE.format(short_lyrics=short_lyrics)

        try:
            raw = await asyncio.wait_for(
                self.llm_generate(
                    prompt,
                    max_tokens=self.MAX_TOKENS_CLUE,
                    temperature=self.TEMP_NARRADOR,
                    stop=self.STOP_CRITERIA
                ),
                timeout=self.TIMEOUT
            )
        except asyncio.TimeoutError:
            raw = "" # forçado a cair no fallback

        # ------------------------------------------------------------------
        # Tenta extrair a dica do JSON estruturado
        # ------------------------------------------------------------------
        clue = ""
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                clue = data.get("dica", "")
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                    if isinstance(data, dict):
                        clue = data.get("dica", "")
                except json.JSONDecodeError:
                    pass

        if clue:
            clue = clue.strip()
        else:
            clue = self._sanitize_clue(raw.strip(), max_words=max_words, lyrics=lyrics)

        # ------------------------------------------------------------------
        # Passo 3: Filtro de Obviedade
        # ------------------------------------------------------------------
        if clue:
            clue_words = set()
            for w in clue.lower().split():
                wc = "".join(ch for ch in w if ch.isalnum())
                if wc and wc not in STOPWORDS:
                    clue_words.add(wc)

            if clue_words & forbidden:
                clue = ""
            elif self._is_literal_substring_of_lyrics(clue, lyrics):
                clue = ""

        # ------------------------------------------------------------------
        # Passo 4: Ajuste de Restrição do Protocolo (max 6 palavras)
        # ------------------------------------------------------------------
        if clue:
            words = clue.split()
            if len(words) > max_words:
                clue = " ".join(words[:max_words])

        # ------------------------------------------------------------------
        # Passo 5: Mecanismo de Contingência (Fallback)
        # ------------------------------------------------------------------
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

        return {"clue": clue}

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
    
    def _sanitize_clue(self, raw_text: str, max_words: int, lyrics: str) -> str:
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

    @tool()
    async def select_card_by_clue(self, clue: str) -> Dict[str, Any]:
        if not self.hand:
            raise RuntimeError("Hand is empty")

        # ------------------------------------------------------------------
        # Passo 1: Extração e Compactação da Mão
        # ------------------------------------------------------------------
        compact_hand = [
            {
                "id": song["id"],
                "titulo": song.get("title", ""),
                "trecho": " ".join(song.get("lyrics", "").split()[:40]),
            }
            for song in self.hand
        ]

        # ------------------------------------------------------------------
        # Passo 2: Montagem do Prompt Dinâmico e Chamada
        # ------------------------------------------------------------------
        prompt = self.PROMPT_SELECT_CARD.format(
            clue=clue,
            hand_json=json.dumps(compact_hand, ensure_ascii=False)
        )

        try:
            raw = await asyncio.wait_for(
                self.llm_generate(
                    prompt, 
                    max_tokens=self.MAX_TOKENS_SELECT, 
                    temperature=self.TEMP_MELOMANO, 
                    stop=self.STOP_CRITERIA
                ),
                timeout=self.TIMEOUT,
            )
        except (asyncio.TimeoutError, Exception):
            raw = ""

        # ------------------------------------------------------------------
        # Passo 3: Interpolação e Captura Segura
        # ------------------------------------------------------------------
        chosen_id = None
        if raw:
            try:
                data = json.loads(raw)
                if isinstance(data, dict):
                    chosen_id = data.get("chosen_id")
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if m:
                    try:
                        data = json.loads(m.group())
                        if isinstance(data, dict):
                            chosen_id = data.get("chosen_id")
                    except json.JSONDecodeError:
                        pass

        valid_ids = {song["id"] for song in self.hand}

        if chosen_id is not None and chosen_id in valid_ids:
            # --------------------------------------------------------------
            # Passo 4: Mapeamento de Protocolo
            # --------------------------------------------------------------
            chosen_card = next(s for s in self.hand if s["id"] == chosen_id)
        else:
            # --------------------------------------------------------------
            # Passo 5: Política de Fallback Cego
            # --------------------------------------------------------------
            clue_words = self._normalize_words(clue)
            best_score = -1
            best_idx = 0
            for idx, song in enumerate(self.hand):
                title_words = self._normalize_words(song.get("title", ""))
                score = len(clue_words.intersection(title_words))
                if score > best_score:
                    best_score = score
                    best_idx = idx
            chosen_card = self.hand[best_idx]

        return {"chosen_card": chosen_card}

    @tool()
    async def vote(self, clue: str, options: List[Dict[str, Any]], my_chosen_card: Dict[str, Any]) -> Dict[str, Any]:
        # ------------------------------------------------------------------
        # Passo 1: Isolamento de Identidade
        # ------------------------------------------------------------------
        my_idx = next(i for i, option in enumerate(options) if option["id"] == my_chosen_card["id"])

        # ------------------------------------------------------------------
        # Passo 2: Compactação e Anonimização do Cenário
        # ------------------------------------------------------------------
        candidates = []
        for i, opt in enumerate(options):
            if i == my_idx:
                continue
            candidates.append({
                "indice_original": i,
                "titulo": opt.get("title", ""),
                "trecho": " ".join(opt.get("lyrics", "").split()[:30]),
            })

       # ------------------------------------------------------------------
        # Passo 3: Montagem do Prompt Dinâmico e Chamada
        # ------------------------------------------------------------------
        prompt = self.PROMPT_VOTE.format(
            clue=clue,
            candidates_json=json.dumps(candidates, ensure_ascii=False)
        )

        try:
            raw = await asyncio.wait_for(
                self.llm_generate(
                    prompt, 
                    max_tokens=self.MAX_TOKENS_VOTE, 
                    temperature=self.TEMP_MELOMANO, 
                    stop=self.STOP_CRITERIA
                ),
                timeout=self.TIMEOUT,
            )
        except (asyncio.TimeoutError, Exception):
            raw = ""

        # ------------------------------------------------------------------
        # Passo 4: Interpolação e Sanidade de Regras
        # ------------------------------------------------------------------
        votes = None
        if raw:
            try:
                data = json.loads(raw)
                if isinstance(data, dict):
                    votes = data.get("votos")
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if m:
                    try:
                        data = json.loads(m.group())
                        if isinstance(data, dict):
                            votes = data.get("votos")
                    except json.JSONDecodeError:
                        pass

        valid = (
            isinstance(votes, list)
            and len(votes) == 2
            and votes[0] != votes[1]
            and my_idx not in votes
            and all(0 <= v < len(options) for v in votes)
        )

        if not valid:
            # --------------------------------------------------------------
            # Passo 5: Política de Fallback Heurístico
            # --------------------------------------------------------------
            clue_words = self._normalize_words(clue)
            scored: List[tuple[int, int]] = []
            for idx, option in enumerate(options):
                if idx == my_idx:
                    continue
                title_words = self._normalize_words(option.get("title", ""))
                score = len(clue_words.intersection(title_words))
                scored.append((score, idx))

            scored.sort(reverse=True)

            votes = []
            for _, idx in scored:
                if idx != my_idx and idx not in votes:
                    votes.append(idx)
                if len(votes) == 2:
                    break

            if len(votes) < 2:
                for idx in range(len(options)):
                    if idx != my_idx and idx not in votes:
                        votes.append(idx)
                    if len(votes) == 2:
                        break

        return {"votes": votes[:2]}

    def _normalize_words(self, text: str) -> set[str]:
        # normaliza palavras no texto e devolve como um conjunto
        cleaned = []
        for token in text.lower().split():
            token = "".join(ch for ch in token if ch.isalnum())
            if token:
                cleaned.append(token)
        return set(cleaned)


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
