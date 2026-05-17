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
import random
from typing import Any, Dict, List

from base_agent import BaseAgent
from fasta2a import A2AApp, tool

app = A2AApp(name="LLMAgent")


class LLMAgent(BaseAgent):
    def __init__(self, name: str, llm_url: str):
        super().__init__(name=name, llm_url=llm_url, request_timeout=60.0)

    @tool()
    async def receive_hand(self, hand: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.hand = list(hand)
        return {"status": "ok", "hand_size": len(self.hand)}

    @tool()
    async def choose_card(self) -> Dict[str, Any]:
        # Heurística mega simples:
        # escolhe a carta cuja letra truncada tem tamanho mais próximo da mediana.
        # Pelamor, n vao me entregar isso de volta!!! x-(
        if not self.hand:
            raise RuntimeError("Hand is empty")

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

        return {"chosen_card": self.hand[best_idx]}

    @tool()
    async def send_clue(self, lyrics: str, max_words: int = 6) -> Dict[str, Any]:
        # Prompt bem simples
        # Um exemplo de cm se comunicar com a LLM
        short_lyrics = " ".join(lyrics.split()[:60])

        prompt = (
            "Crie uma dica curta para um jogo de associacao.\n"
            f"Use no maximo {max_words} palavras.\n"
            "Responda apenas com a dica.\n\n"
            f"Letra:\n{short_lyrics}\n\n"
            "Dica:"
        )

        raw = await self.llm_generate(
            prompt,
            max_tokens=20,
            temperature=0.4,
            stop=["\n\n", "\nResposta:", "\nAnswer:", "###"],
        )

        clue = self._sanitize_clue(raw.strip(), max_words=max_words, lyrics=lyrics)

        if not clue:
            clue = "coisa estranha"

        return {"clue": clue}

    @tool()
    async def select_card_by_clue(self, clue: str) -> Dict[str, Any]:
        # Estratégia simples:
        # escolhe a carta cujo título tem mais palavras em comum com a dica.
        # Se nenhuma tiver interseção, escolhe aleatoriamente.
        if not self.hand:
            raise RuntimeError("Hand is empty")

        clue_words = self._normalize_words(clue)

        best_score = -1
        best_indices: List[int] = []

        for idx, song in enumerate(self.hand):
            title_words = self._normalize_words(song.get("title", ""))
            score = len(clue_words.intersection(title_words))
            if score > best_score:
                best_score = score
                best_indices = [idx]
            elif score == best_score:
                best_indices.append(idx)

        if best_score <= 0:
            chosen_idx = random.randrange(len(self.hand))
        else:
            chosen_idx = best_indices[0]

        return {"chosen_card": self.hand[chosen_idx]}

    @tool()
    async def vote(self, clue: str, options: List[Dict[str, Any]], my_chosen_card: Dict[str, Any]) -> Dict[str, Any]:
        # Estratégia simples:
        # tenta votar nas duas opções com maior interesecao entre dica e título.
        # Se n der certo, vota nas duas primeiras que não forem a própria carta.
        my_idx = next(i for i, option in enumerate(options) if option["id"] == my_chosen_card["id"])
        clue_words = self._normalize_words(clue)

        scored: List[tuple[int, int]] = []
        for idx, option in enumerate(options):
            if idx == my_idx:
                continue
            title_words = self._normalize_words(option.get("title", ""))
            score = len(clue_words.intersection(title_words))
            scored.append((score, idx))

        scored.sort(reverse=True)

        votes: List[int] = []
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
