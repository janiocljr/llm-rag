from __future__ import annotations

from typing import Any

import requests
import json


class APIClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 120):
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._timeout_long = 600


    def _get(self, path: str) -> dict[str, Any]:
        try:
            r = requests.get(f"{self._base}{path}", timeout=self._timeout)
            r.raise_for_status()
            return r.json()
        except requests.ConnectionError:
            return {"error": f"Não foi possível conectar a {self._base}. Verifique se a API está rodando."}
        except requests.Timeout:
            return {"error": "Timeout ao aguardar resposta da API."}
        except requests.HTTPError as exc:
            return {"error": f"HTTP {exc.response.status_code}: {exc.response.text}"}
        except Exception as exc:
            return {"error": str(exc)}

    def query_stream(
        self,
        question: str,
        top_k: int = 5,
        similarity_threshold: float = 0.35,
    ):
        url = f"{self._base}/api/v1/query/stream"
        payload = {
            "question": question,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
        }

        try:
            with requests.post(url, json=payload, stream=True, timeout=self._timeout_long) as r:
                r.raise_for_status()

                for raw in r.iter_lines(decode_unicode=True):
                    if raw is None:
                        continue
                    line = raw.strip()
                    if not line:
                        continue
                    if line.startswith("data:"):
                        data = line[len("data:"):].strip()
                        try:
                            obj = json.loads(data)
                        except Exception:
                            obj = {"type": "text", "text": data}
                        yield obj
        except requests.Timeout:
            yield {"type": "error", "error": "Timeout while waiting for stream."}
        except requests.HTTPError as exc:
            try:
                detail = exc.response.json().get("detail", exc.response.text)
            except Exception:
                detail = exc.response.text
            yield {"type": "error", "error": f"HTTP {exc.response.status_code}: {detail}"}
        except requests.ConnectionError:
            yield {"type": "error", "error": f"Não foi possível conectar a {self._base}."}
        except Exception as exc:
            yield {"type": "error", "error": str(exc)}

    def _post(self, path: str, payload: dict, timeout: int | None = None) -> dict[str, Any]:
        try:
            r = requests.post(
                f"{self._base}{path}",
                json=payload,
                timeout=timeout or self._timeout,
            )
            r.raise_for_status()
            return r.json()
        except requests.ConnectionError:
            return {"error": f"Não foi possível conectar a {self._base}. Verifique se a API está rodando."}
        except requests.Timeout:
            return {"error": "Timeout ao aguardar resposta da API."}
        except requests.HTTPError as exc:
            try:
                detail = exc.response.json().get("detail", exc.response.text)
            except Exception:
                detail = exc.response.text
            return {"error": f"HTTP {exc.response.status_code}: {detail}"}
        except Exception as exc:
            return {"error": str(exc)}


    def health(self) -> dict[str, Any]:
        """GET /health"""
        return self._get("/health")

    def stats(self) -> dict[str, Any]:
        """GET /api/v1/stats"""
        return self._get("/api/v1/stats")

    def ingest(self, force_reindex: bool = False) -> dict[str, Any]:
        """POST /api/v1/ingest"""
        return self._post("/api/v1/ingest", {"force_reindex": force_reindex}, timeout=self._timeout_long)

    def query(
        self,
        question: str,
        top_k: int = 5,
        similarity_threshold: float = 0.35,
    ) -> dict[str, Any]:
        """POST /api/v1/query"""
        return self._post(
            "/api/v1/query",
            {
                "question": question,
                "top_k": top_k,
                "similarity_threshold": similarity_threshold,
            },
            timeout=300,
        )
