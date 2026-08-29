"""LangGraph workflow that drives an explore/critique loop over a proof workspace.

Agents:
  - explorer: proposes proof moves (search DAG: Move from an OR state, with AND subgoals)
  - critic:   verdicts each move and decides whether to keep exploring or stop
  - formalizer:  loop at claim granularity: draft a Lean 4 translation, run a theorem-statement
                  equivalence review, invoke Lean, classify any error, attempt one repair,
                  commit the source/log as artifacts, and promote to lean_verified only
                  when the kernel genuinely accepts a sorry-free, axiom-clean statement

LLM clients differ only in *how* they call the model; prompting and response
normalisation live on the shared ``LLMClient`` base. The formal verifier is a
pluggable socket (heuristic now, Lean 4 later).
"""

from __future__ import annotations
import hashlib
import json
import os
import re
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from typing import Any, Dict, List, Optional
from proof_proto.proof_project import ProofProject

ROOT_STATE_ID = "root"
MAX_SUBGOALS_PER_MOVE = 5
DEFAULT_ITERATIONS = 2
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# Single source of truth for attempt statuses and the strings sent to the model.
ATTEMPT_STATUSES = ("conjectural","supported", "critic_accepted", "lean_verified", "refuted", "retracted")
VALID_ATTEMPT_STATUSES = set(ATTEMPT_STATUSES)

FALLBACK_EXPLORATION: dict[str, Any] = {
    "move_summary": "Begin by reformulating the goal into a more tractable equivalent form.",
    "claim_statement": "A useful reformulation or reduction makes the target easier to analyze.",
    "required_subgoals": [
        "Identify a meaningful reformulation or reduction of the goal.",
        "Establish the transformed statement or reduced case.",
    ],
}

DEFAULT_CRITIQUE: dict[str, Any] = {
    "decision": "continue",
    "reason": "The proposed move is plausible and worth testing in the current proof state.",
    "status": "supported",
}

FALLBACK_FORMALIZATION: dict[str, Any] = {
    "translatable": False,
    "lean_code": "",
    "lean_name": "",
    "explanation": "Could not parse a formalization response from the model.",
}

FALLBACK_EQUIVALENCE: dict[str, Any] = {
    "relation": "unclear",
    "notes": "Could not parse an equivalence-review response from the model.",
}


LEAN_ERROR_CATEGORIES = (
    "translation_ambiguity",
    "missing_definition_or_library_lemma",
    "elaboration_type_mismatch",
    "tactic_search_failure",
    "resource_timeout",
    "inconsistent_assumptions",
    "likely_mathematical_gap",
   "formal_statement_stronger_than_informal",
   "missing_mathlib_dependency",
   "stale_or_moved_mathlib_module"
)

MATHLIB_GUIDANCE = (
    "\nMathlib-specific rules for this environment:\n"
    "- Never write `import Mathlib` alone. Import the narrowest specific module\n"
    "  you actually need (e.g. `import Mathlib.Algebra.Ring.Parity`, not the whole\n"
    "  library) — bare `import Mathlib` takes several minutes to load.\n"
    "- Mathlib reorganizes module paths and renames declarations frequently.\n"
    "  If your proof fails with 'bad import' or 'unknown identifier'/'unknown\n"
    "  constant', this is very likely a stale name or moved module, not\n"
    "  necessarily wrong mathematics.\n"
    "- Never declare a variable, binder, or hypothesis using the same name as a\n"
    "  built-in type notation (ℕ, ℤ, ℝ, ℚ, etc.) — this silently shadows the real\n"
    "  type and produces confusing, unrelated-looking errors.\n"
)



# ---------------------------------------------------------------------------
# Prompt building and response parsing
# ---------------------------------------------------------------------------

def _explore_prompt(theorem: str, context: dict) -> str:
    previous = [a["move_summary"] for a in context.get("attempts", [])]
    return (
        f"You are an explorer agent for a mathematical proof project.\n"
        f"Theorem: {theorem}\n"
        f"Already attempted moves (do NOT repeat these): {json.dumps(previous)}\n"
        f"Current context: {json.dumps(context, indent=2)[:2000]}\n"
        f"Return ONLY a JSON object with exactly three keys:\n"
        f'  "move_summary" — a new distinct proof move\n'
        f'  "claim_statement" — what this move claims\n'
        f'  "required_subgoals" — a JSON array of strings naming subgoals this move'
        f" must establish before it succeeds (empty array if it closes the state directly)"
    )


def _critique_prompt(theorem: str, move_summary: str, claim_statement: str, context: dict) -> str:
    statuses = ", ".join(ATTEMPT_STATUSES)
    return (
        f"You are a critic agent for a mathematical proof project.\n"
        f"Theorem: {theorem}\n"
        f"Move summary: {move_summary}\n"
        f"Claim statement: {claim_statement}\n"
        f"Context: {json.dumps(context, indent=2)[:2000]}\n"
        f"Return ONLY a JSON object with keys:\n"
        f'  "decision": "continue" or "stop"\n'
        f'  "reason": explanation of your verdict\n'
        f'  "status": MUST be one of: {statuses}'
    )

def _formalize_prompt(theorem: str, move_summary: str, claim_statement: str, context: dict) -> str:
    return (
        f"You are a formalizer agent for a mathematical proof project. Your job is to\n"
        f"translate an informally-stated claim into Lean 4 syntax, not to invent a new\n"
        f"proof. Use `sorry` for any tactic steps you cannot fill in yet.\n"
        f"Theorem: {theorem}\n"
        f"Move summary: {move_summary}\n"
        f"Claim statement (the exact proposition to formalize): {claim_statement}\n"
        f"Context: {json.dumps(context, indent=2)[:2000]}\n"
        f"{MATHLIB_GUIDANCE}\n"
        f"Return ONLY a JSON object with exactly these keys:\n"
        f'  "translatable" — true or false: can this claim be meaningfully stated\n'
        f"    in Lean 4 right now (even with `sorry` in the proof body)?\n"
        f'  "lean_code" — a best-effort Lean 4 snippet (a `theorem`/`lemma` header plus a\n'
        f"    proof body); empty string if translatable is false\n"
        f'  "lean_name" — a valid Lean identifier (snake_case) naming the statement;\n'
        f"    empty string if translatable is false\n"
        f'  "explanation" — one or two sentences on why it is/isn\'t translatable, or on\n'
        f"    what was left as `sorry`"
    )

def _equivalence_prompt(theorem: str, claim_statement: str, lean_code: str) -> str:
    return (
        f"You are reviewing a Lean 4 translation for faithfulness to an informal claim.\n"
        f"Do not judge whether the Lean code compiles — only whether, if it did compile,\n"
        f"it would say the same mathematical thing as the informal claim: same quantifiers,\n"
        f"same domain, same strength.\n"
        f"Theorem being worked on: {theorem}\n"
        f"Informal claim: {claim_statement}\n"
        f"Lean 4 statement:\n{lean_code}\n"
        f"Return ONLY a JSON object with exactly these keys:\n"
        f'  "relation" — exactly one of: "equivalent", "stronger", "weaker", "unrelated"\n'
        f"    (\"stronger\" means the Lean statement claims more than the informal claim did;\n"
        f"    \"weaker\" means it claims less; \"unrelated\" means the translation misses the point)\n"
        f'  "notes" — one or two sentences justifying the verdict'
    )
def _repair_prompt(
    theorem: str, claim_statement: str, lean_code: str, diagnostic: str, category: str, context: dict,
) -> str:
    return (
        f"You are repairing a rejected Lean 4 translation for a mathematical proof project.\n"
        f"Theorem: {theorem}\n"
        f"Claim statement: {claim_statement}\n"
        f"Rejected Lean 4 code:\n{lean_code}\n"
        f"Rejection category: {category}\n"
        f"Diagnostic (compiler output or equivalence-review note): {diagnostic}\n"
        f"Context: {json.dumps(context, indent=2)[:1500]}\n"
        f"{MATHLIB_GUIDANCE}\n"
        f"Either produce a corrected Lean 4 translation, or if the diagnostic reveals a\n"
        f"genuine mathematical gap (not just a translation slip), say so honestly.\n"
        f"Return ONLY a JSON object with exactly these keys:\n"
        f'  "translatable" — true if you produced a corrected translation, false if the\n'
        f"    diagnostic exposes a real mathematical gap rather than a fixable translation issue\n"
        f'  "lean_code" — the corrected Lean 4 snippet; empty string if translatable is false\n'
        f'  "lean_name" — a valid Lean identifier (snake_case); empty string if translatable is false\n'
        f'  "explanation" — what was fixed, or what gap was exposed'
    )



def _load_local_env() -> None:
    """Populate os.environ from a repo-local .env file if present.

    This lets the app read GROQ_API_KEY without requiring the user to export it
    in every terminal session. The project root is chosen as the directory above
    this file, which makes the setup convenient in normal repo usage.
    """
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[1] / ".env",
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        for raw_line in resolved.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def _parse_json(text: str) -> dict:
    """Best-effort JSON parsing that tolerates code fences and prose."""
    if text:
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass
    stripped = text.strip()
    stripped = re.sub(r"^```[a-zA-Z]*\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {}


def _normalize_exploration(result: dict) -> dict:
    """Coerce an LLM exploration response into a well-formed move proposal."""
    move = result.get("move_summary")
    if not isinstance(move, str) or not move:
        return dict(FALLBACK_EXPLORATION)
    claim = result.get("claim_statement", "")
    subgoals = result.get("required_subgoals", [])
    if not isinstance(subgoals, list):
        subgoals = []
    return {
        "move_summary": move,
        "claim_statement": claim if isinstance(claim, str) else "",
        "required_subgoals": [s for s in subgoals if isinstance(s, str)],
    }


def _normalize_critique(result: dict) -> dict:
    """Coerce an LLM critique response into a well-formed verdict."""
    if "decision" not in result:
        return dict(DEFAULT_CRITIQUE)
    for key in ("decision", "status"):
        if key in result:
            result[key] = str(result[key]).lower()
    if result.get("status") not in VALID_ATTEMPT_STATUSES:
        result["status"] = "supported"
    return result

def _normalize_formalization(result: dict) -> dict:
    """Coerce an LLM formalization response into a well-formed translation draft."""
    if not isinstance(result, dict) or "translatable" not in result:
        return dict(FALLBACK_FORMALIZATION)
    lean_code = result.get("lean_code", "")
    lean_name = result.get("lean_name", "")
    explanation = result.get("explanation", "")
    return {
        "translatable": bool(result.get("translatable")),
        "lean_code": lean_code if isinstance(lean_code, str) else "",
        "lean_name": lean_name if isinstance(lean_name, str) else "",
        "explanation": explanation if isinstance(explanation, str) else "",
    }


def _normalize_equivalence(result: dict) -> dict:
    """Coerce an LLM equivalence-review response into a well-formed verdict."""
    if not isinstance(result, dict) or "relation" not in result:
        return dict(FALLBACK_EQUIVALENCE)
    relation = str(result.get("relation", "")).lower()
    if relation not in {"equivalent", "stronger", "weaker", "unrelated"}:
        relation = "unclear"
    notes = result.get("notes", "")
    return {"relation": relation, "notes": notes if isinstance(notes, str) else ""}


# ---------------------------------------------------------------------------
# LLM clients
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """Base interface for LLM-backed explorer/critic agents.

    Subclasses only implement ``_call(prompt) -> str``; prompting and
    response normalisation are shared here.
    """

    def explore(self, theorem: str, context: dict) -> dict:
        return _normalize_exploration(self._respond(_explore_prompt(theorem, context)))

    def critique(self, theorem: str, move_summary: str, claim_statement: str, context: dict) -> dict:
        return _normalize_critique(
            self._respond(_critique_prompt(theorem, move_summary, claim_statement, context))
        )
    def formalize(self, theorem: str, move_summary: str, claim_statement: str, context: dict) -> dict:
         return _normalize_formalization(
            self._respond(_formalize_prompt(theorem, move_summary, claim_statement, context))
        )

    def check_equivalence(self, theorem: str, claim_statement: str, lean_code: str) -> dict:
        return _normalize_equivalence(
            self._respond(_equivalence_prompt(theorem, claim_statement, lean_code))
        )

    def repair_formalization(
        self, theorem: str, claim_statement: str, lean_code: str, diagnostic: str, category: str, context: dict,
    ) -> dict:
        return _normalize_formalization(
            self._respond(_repair_prompt(theorem, claim_statement, lean_code, diagnostic, category, context))
        )


    def _respond(self, prompt: str) -> dict:
        return _parse_json(self._call(prompt))

    @abstractmethod
    def _call(self, prompt: str) -> str:
        raise NotImplementedError


class OllamaLLMClient(LLMClient):
    """Adapter for a local Ollama server."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: Optional[str] = None):
        self.base_url = base_url
        self.model = model or "llama3.1:8b"

    def _call(self, prompt: str) -> str:
        payload = json.dumps(
            {"model": self.model, "prompt": prompt, "stream": False, "format": "json"}
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
            return data.get("response", "")
        except Exception:
            return ""


# Backward-compatible alias for the previous name of the Ollama client.
SimpleLLMClient = OllamaLLMClient


class GeminiLLMClient(LLMClient):
    """Gemini-backed LLM client."""

    def __init__(self, model: Optional[str] = None):
        import google.generativeai as genai

        api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")).strip()
        genai.configure(api_key=api_key, transport="rest")
        self._model = genai.GenerativeModel(model or "gemini-2.0-flash")

    def _call(self, prompt: str) -> str:
        return self._model.generate_content(prompt).text


class GroqLLMClient(LLMClient):
    """Groq-backed LLM client — fast free tier."""

    def __init__(self, model: Optional[str] = None):
        _load_local_env()
        from groq import Groq

        key = os.environ.get("GROQ_API_KEY", "").strip()
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY is missing. Put it in .env at the repo root or export it in your shell."
            )
        self._client = Groq(api_key=key)
        self._model = model or "llama-3.1-8b-instant"

    def _call(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content


def make_llm_client(
    provider: str = "auto",
    model: Optional[str] = None,
    ollama_url: str = OLLAMA_BASE_URL,
) -> LLMClient:
    """Select an LLM client by provider preference, falling back to env vars.

    "auto" prefers Groq when GROQ_API_KEY is set, Gemini when opted in via
    USE_GEMINI, otherwise the local Ollama server.
    """
    _load_local_env()
    provider = (provider or "auto").lower()
    if provider == "auto":
        if os.environ.get("GROQ_API_KEY"):
            return GroqLLMClient(model=model)
        if os.environ.get("USE_GEMINI") and (
            os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        ):
            return GeminiLLMClient(model=model)
        return OllamaLLMClient(base_url=ollama_url, model=model)
    if provider == "groq":
        return GroqLLMClient(model=model)
    if provider == "ollama":
        return OllamaLLMClient(base_url=ollama_url, model=model)
    raise ValueError(f"Unknown LLM provider {provider!r} (choose from: auto, groq, ollama)")


# ---------------------------------------------------------------------------
# Formal verifier socket
# ---------------------------------------------------------------------------

class FormalVerifier:
    """Socket for a formal verifier — LLM/heuristic based now, Lean 4 later.

    To plug in Lean 4: subclass this, override verify(), return
    verdict='verified' only when `lake build` passes with no errors.
    """

    def verify(self, theorem: str, attempts: list) -> dict:
        """Given the full attempt chain, decide if the proof is complete.

        The default implementation is intentionally generic: it does not assume a
        specific theorem family or algebraic pattern. A proof is considered closed
        only when the attempt history includes an explicit accepted or verified
        status, leaving the actual certification to a stronger backend later.

        Returns:
            closed: bool — True if proof is complete
            reason: str  — explanation
        """
        accepted_statuses = {"critic_accepted", "lean_verified"}
        if any(a.get("status") in accepted_statuses for a in attempts):
            return {
                "closed": True,
                "reason": "The proof chain contains an explicit accepted or verified result.",
            }
        return {
            "closed": False,
            "reason": "The proof chain is not yet explicitly accepted or verified.",
        }


class Lean4Verifier(FormalVerifier):
    """Placeholder for future Lean 4 integration.

    When implemented: translate attempt chain to Lean 4 syntax,
    run `lake build`, parse output, return verified only on success.
    """

    def verify(self, theorem: str, attempts: list) -> dict:
        raise NotImplementedError("Lean 4 verifier not yet implemented — use FormalVerifier for now")


# ---------------------------------------------------------------------------
# Formalizer agent 
# ---------------------------------------------------------------------------

def _lean_identifier(text: str) -> str:
    """Turn an arbitrary string into a safe Lean 4 namespace/identifier fragment."""
    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", text).strip("_")
    if not cleaned or not cleaned[0].isalpha():
        cleaned = f"P_{cleaned}"
    return cleaned


def classify_lean_error(output: str, *, timed_out: bool = False) -> str:

    if timed_out:
        return "resource_timeout"
    text = output.lower()
    if (
        "unknown module prefix" in text
        or ("no directory" in text and ".olean" in text)
        or ("object file" in text and ".olean" in text and "does not exist" in text)
    ):
        return "missing_mathlib_dependency"
    if "bad import" in text:
        return "stale_or_moved_mathlib_module"
    if "unknown identifier" in text or "unknown constant" in text or "unknown namespace" in text:
        return "missing_definition_or_library_lemma"
    if "type mismatch" in text or "failed to synthesize" in text:
        return "elaboration_type_mismatch"
    if "unsolved goals" in text or "tactic" in text and "failed" in text:
        return "tactic_search_failure"
    if "inconsistent" in text:
        return "inconsistent_assumptions"
    return "likely_mathematical_gap"


class LeanChecker:

    def __init__(
        self,
        binary: str = "lean",
        timeout_seconds: int = 45,
        use_lake: bool = False,
        lake_project_dir: str = "",
    ):
        self.binary = binary
        self.timeout_seconds = timeout_seconds
        self.use_lake = use_lake
        self.lake_project_dir = lake_project_dir
        self._toolchain_cache: str = ""
        self._mathlib_revision_cache: str = ""

    def available(self) -> bool:
        if self.use_lake:
            return shutil.which("lake") is not None
        return shutil.which(self.binary) is not None

    def _resolve_toolchain(self) -> str:
        if not self._toolchain_cache:
            try:
                command = (
                    ["lake", "env", self.binary, "--version"]
                    if self.use_lake else [self.binary, "--version"]
                )
                proc = subprocess.run(
                    command, cwd=self.lake_project_dir or None,
                    capture_output=True, text=True, encoding="utf-8", timeout=10,
                )
                self._toolchain_cache = (proc.stdout or proc.stderr).strip()
            except Exception:
                self._toolchain_cache = ""
        return self._toolchain_cache

    def _resolve_mathlib_revision(self) -> str:
        if not self._mathlib_revision_cache and self.lake_project_dir:
            mathlib_dir = os.path.join(self.lake_project_dir, ".lake", "packages", "mathlib")
            try:
                proc = subprocess.run(
                    ["git", "-C", mathlib_dir, "rev-parse", "HEAD"],
                    capture_output=True, text=True, encoding="utf-8", timeout=10,
                )
                if proc.returncode == 0:
                    self._mathlib_revision_cache = proc.stdout.strip()
            except Exception:
                pass
        return self._mathlib_revision_cache

    def check(
        self,
        lean_code: str,
        *,
        proof_id: str = "",
        claim_id: str = "",
        toolchain: str = "",
        mathlib_revision: str = "",
        deny_sorry: bool = True,
    ) -> dict:
        """
        Returns a dict with: status, output (diagnostics), imports_used,
        axioms_used, timing_seconds, source_hash, error_category (only set
        on a genuine compiler failure). 
        """
        toolchain = toolchain or self._resolve_toolchain()
        mathlib_revision = mathlib_revision or (
            self._resolve_mathlib_revision() if self.use_lake else ""
        )

        imports_used = re.findall(r"^\s*import\s+(\S+)", lean_code, re.MULTILINE)

        source_hash = hashlib.sha256(lean_code.encode("utf-8")).hexdigest()[:16]
        base_result = {
            "proof_id": proof_id,
            "claim_id": claim_id,
            "toolchain": toolchain,
            "mathlib_revision": mathlib_revision,
            "source_hash": source_hash,
            "imports_used": imports_used,
            "axioms_used": [],
            "timing_seconds": 0.0,
            "error_category": "",
        }
        # ... everything else in the method body stays exactly as it is today
        if not lean_code or not lean_code.strip():
            return {**base_result, "status": "unavailable", "output": "No Lean code to check."}
        if not self.available():
            return {
                **base_result,
                "status": "unavailable",
                "output": f"'{self.binary}' not found on PATH; skipping local type-check.",
            }

        declared_axioms = re.findall(r"\baxiom\s+([A-Za-z_][A-Za-z0-9_']*)", lean_code)
        uses_sorry = bool(re.search(r"\bsorry\b", lean_code))

        handle = tempfile.NamedTemporaryFile(suffix=".lean", mode="w", delete=False, encoding="utf-8")
        started = time.monotonic()
        try:
            handle.write(lean_code)
            handle.close()

            command = (
                ["lake", "env", self.binary, handle.name]
                if self.use_lake else [self.binary, handle.name]
            )
            proc = subprocess.run(
                command, cwd=self.lake_project_dir or None,
                capture_output=True, text=True, encoding="utf-8", timeout=self.timeout_seconds,
            )
            elapsed = time.monotonic() - started
            output = proc.stdout + proc.stderr
            if proc.returncode != 0:
                return {
                    **base_result,
                    "status": "failed",
                    "output": output[-4000:],
                    "timing_seconds": elapsed,
                    "error_category": classify_lean_error(output),
                }
            uses_sorry = uses_sorry or "declaration uses 'sorry'" in output.lower()
            if uses_sorry and deny_sorry:
                return {
                    **base_result,
                    "status": "incomplete_sorry",
                    "output": output[-4000:],
                    "timing_seconds": elapsed,
                    "axioms_used": ["sorryAx"],
                }
            if declared_axioms:
                return {
                    **base_result,
                    "status": "untracked_axiom",
                    "output": output[-4000:],
                    "timing_seconds": elapsed,
                    "axioms_used": declared_axioms,
                }
            return {**base_result, "status": "verified", "output": output, "timing_seconds": elapsed}
        except subprocess.TimeoutExpired:
            return {
                **base_result,
                "status": "failed",
                "output": "Lean check timed out.",
                "timing_seconds": self.timeout_seconds,
                "error_category": classify_lean_error("", timed_out=True),
            }
        except OSError as exc:
            return {**base_result, "status": "unavailable", "output": f"Lean check errored: {exc}"}
        finally:
            try:
                os.unlink(handle.name)
            except OSError:
                pass

# ---------------------------------------------------------------------------
# LangGraph workflow
# ---------------------------------------------------------------------------

class WorkflowState(TypedDict, total=False):
    theorem: str
    project: ProofProject
    iteration: int
    max_iterations: int
    last_move: str
    last_claim: str
    last_move_id: str
    current_attempt_id: str
    last_critique: str
    last_critique_decision: str
    last_lean_status: str
    proof_closed: bool


class ProofWorkflow:
    """The explore/critique/formalize loop as a compiled LangGraph.

    init -> explore -> critique -> formalize, looping back to explore until the
    critic says stop, the formalizer's Lean check closes the proof, or the max
     iteration budget is exhausted..
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        verifier: Optional[FormalVerifier] = None,
        lean_checker: Optional[LeanChecker] = None,
        toolchain: str = "",
        mathlib_revision: str = "",
    ):
        self.llm = llm or OllamaLLMClient()
        self.verifier = verifier or FormalVerifier()
        self.lean_checker = lean_checker or LeanChecker()
        self.toolchain = toolchain
        self.mathlib_revision = mathlib_revision

    def build(self):
        workflow = StateGraph(WorkflowState)
        workflow.add_node("init", self._init_state)
        workflow.add_node("explore", self._explore)
        workflow.add_node("critique", self._critique)
        workflow.add_node("formalize", self._formalize)
        workflow.set_entry_point("init")
        workflow.add_edge("init", "explore")
        workflow.add_edge("explore", "critique")
        workflow.add_edge("critique", "formalize")
        workflow.add_conditional_edges(
            "formalize", self._should_continue, {"continue": "explore", END: END}
        )
        return workflow.compile()

    # --- nodes ------------------------------------------------------------

    @staticmethod
    def _init_state(state: WorkflowState) -> WorkflowState:
        state["iteration"] = state.get("iteration", 0)
        state["max_iterations"] = state.get("max_iterations", DEFAULT_ITERATIONS)
        return state

    def _explore(self, state: WorkflowState) -> WorkflowState:
        project: ProofProject = state["project"]
        context = project.context_for(ROOT_STATE_ID)
        exploration = self.llm.explore(state["theorem"], context)

        # Search DAG: the explorer proposes a Move from the root state (OR point),
        # optionally requiring subgoal States (AND points).
        move = project.add_move(
            ROOT_STATE_ID,
            exploration["move_summary"],
            kind="reduction",
            note=exploration.get("claim_statement", ""),
        )
        for subgoal_desc in exploration.get("required_subgoals", [])[:MAX_SUBGOALS_PER_MOVE]:
            project.add_subgoal(move["id"], subgoal_desc, ROOT_STATE_ID)

        # Attempt id is derived from the current count and carried in state so
        # the critic verdict lands on the exact attempt just explored.
        attempt_id = self._next_attempt_id(project)
        project.record_attempt(
            attempt_id,
            ROOT_STATE_ID,
            exploration["move_summary"],
            worker="explorer",
            note=exploration.get("claim_statement", ""),
            move_id=move["id"],
        )

        state["last_move"] = exploration["move_summary"]
        state["last_claim"] = exploration.get("claim_statement", "")
        state["last_move_id"] = move["id"]
        state["current_attempt_id"] = attempt_id
        state["iteration"] = state.get("iteration", 0) + 1
        return state

    def _critique(self, state: WorkflowState) -> WorkflowState:
        project: ProofProject = state["project"]
        context = project.context_for(ROOT_STATE_ID)
        critique = self.llm.critique(
            state["theorem"],
            state["last_move"],
            state.get("last_claim", ""),
            context,
        )
        project.mark_attempt(
            state["current_attempt_id"],
            critique["status"],
            critique.get("reason", ""),
        )
        state["last_critique"] = critique.get("reason", "")
        state["last_critique_decision"] = critique.get("decision", "continue")
        return state
    
    def _formalize(self, state: WorkflowState) -> WorkflowState:
        project: ProofProject = state["project"]
        theorem = state["theorem"]
        claim_statement = state.get("last_claim", "") or state["last_move"]
        attempt_id = state["current_attempt_id"]
        claim_id = f"claim-{attempt_id}"
        context = project.context_for(ROOT_STATE_ID)

        draft = self.llm.formalize(theorem, state["last_move"], claim_statement, context)
        state["last_lean_status"] = "not_translatable"

        if not draft["translatable"] or not draft["lean_code"]:
            project.add_claim(claim_id, claim_statement)
            project.link_produced_claim(attempt_id, claim_id)
            project.record_lean_formalization(
                claim_id, formalization_status="not_translatable",
                last_compiler_output=draft.get("explanation", ""),
            )
            return self._maybe_close(state)

        lean_code = draft["lean_code"]
        lean_name = draft["lean_name"] or _lean_identifier(claim_id)
        namespace = f"Proof_{_lean_identifier(project.proof_id)}"

    
        # claim, before we ever bother invoking Lean.
        review = self.llm.check_equivalence(theorem, claim_statement, lean_code)
        relation_to_category = {
            "stronger": "formal_statement_stronger_than_informal",
            "weaker": "translation_ambiguity",
            "unrelated": "translation_ambiguity",
            "unclear": "translation_ambiguity",
        }
        equivalence_category = relation_to_category.get(review["relation"], "")

        repaired_once = False
        if equivalence_category:
            repair = self.llm.repair_formalization(
                theorem, claim_statement, lean_code, review["notes"], equivalence_category, context,
            )
            repaired_once = True
            if repair["translatable"] and repair["lean_code"]:
                lean_code = repair["lean_code"]
                lean_name = repair["lean_name"] or lean_name
                review = self.llm.check_equivalence(theorem, claim_statement, lean_code)
                equivalence_category = relation_to_category.get(review["relation"], "")
            else:
                equivalence_category = "likely_mathematical_gap"

        final_status = equivalence_category or "pending"
        check: dict = {}
        if not equivalence_category:
            check = self.lean_checker.check(
                lean_code, proof_id=project.proof_id, claim_id=claim_id,
                toolchain=self.toolchain, mathlib_revision=self.mathlib_revision,
                deny_sorry=True,
            )
            final_status = check.get("status", "unavailable")

            if final_status == "failed" and not repaired_once:
                repair = self.llm.repair_formalization(
                    theorem, claim_statement, lean_code, check.get("output", ""),
                    check.get("error_category", ""), context,
                )
                if repair["translatable"] and repair["lean_code"]:
                    lean_code = repair["lean_code"]
                    lean_name = repair["lean_name"] or lean_name
                    check = self.lean_checker.check(
                        lean_code, proof_id=project.proof_id, claim_id=claim_id,
                        toolchain=self.toolchain, mathlib_revision=self.mathlib_revision,
                        deny_sorry=True,
                    )
                    final_status = check.get("status", "unavailable")

        source_artifact = project.write_artifact(
            name=f"{attempt_id}-lean", content=lean_code, kind="lean_snippet", attempt_id=attempt_id,
        )
        if check.get("output"):
            project.write_artifact(
                name=f"{attempt_id}-lean-log", content=check["output"],
                kind="lean_check_log", attempt_id=attempt_id,
            )

        resolved_toolchain = check.get("toolchain") or self.toolchain
        resolved_mathlib_revision = check.get("mathlib_revision") or self.mathlib_revision

        project.add_claim(claim_id, claim_statement)
        project.link_produced_claim(attempt_id, claim_id)
        project.record_lean_formalization(
            claim_id, lean_name=lean_name, lean_statement_path=source_artifact["path"],
            namespace=namespace, toolchain_hash=resolved_toolchain,
            mathlib_revision=resolved_mathlib_revision, formalization_status=final_status,
            last_compiler_output=check.get("output", review.get("notes", "")),
        )

        project.add_verification(
            attempt_id, claim_id, kind="lean", status=final_status, lean_name=lean_name,
            toolchain_hash=resolved_toolchain, mathlib_revision=resolved_mathlib_revision,
            error_category=check.get("error_category", equivalence_category),
            axioms_used=check.get("axioms_used", []), timing_seconds=check.get("timing_seconds", 0.0),
            source_hash=check.get("source_hash", ""),
        )

        if final_status == "verified":
            project.update_claim_status(
                claim_id, "lean_verified",
                reason="Lean 4 kernel accepted a sorry-free, axiom-clean statement.",
            )
            project.mark_attempt(
                attempt_id, "lean_verified", "Lean 4 kernel accepted the formalizer's translation.",
            )

        state["last_lean_status"] = final_status
        return self._maybe_close(state)


    def _maybe_close(self, state: WorkflowState) -> WorkflowState:
        """Run the formal verifier now that the critic's verdict and any Lean check
        are both recorded on the attempt, so a genuine Lean pass can close the
        state exactly like a critic acceptance can (both are in FormalVerifier's
        accepted_statuses).
        """
        project: ProofProject = state["project"]
        all_attempts = project.graph.get_attempts_for_state(ROOT_STATE_ID, project.proof_id)
        verdict = self.verifier.verify(state["theorem"], all_attempts)
        if verdict["closed"] or state.get("last_critique_decision") == "stop":
            project.close_state(ROOT_STATE_ID, verdict["reason"] or state.get("last_critique", ""))
            state["proof_closed"] = True
        return state
    
    # --- routing ----------------------------------------------------------

    @staticmethod
    def _next_attempt_id(project: ProofProject) -> str:
        existing = len(project.graph.get_attempts_for_state(ROOT_STATE_ID, project.proof_id))
        return f"attempt-{existing + 1}"

    @staticmethod
    def _should_continue(state: WorkflowState) -> str:
        if state.get("proof_closed"):
            return END
        max_iterations = state.get("max_iterations", DEFAULT_ITERATIONS)
        return "continue" if state.get("iteration", 0) < max_iterations else END


def build_graph(
    llm_client: Optional[LLMClient] = None,
    verifier: Optional[FormalVerifier] = None,
    lean_checker: Optional[LeanChecker] = None,
    toolchain: str = "",
    mathlib_revision: str = "",
):
    return ProofWorkflow(llm_client, verifier, lean_checker, toolchain, mathlib_revision).build()


def run_workflow(
    theorem: str,
    root: str,
    llm_client: Optional[LLMClient] = None,
    max_iterations: int = DEFAULT_ITERATIONS,
    toolchain: str = "",
    mathlib_revision: str = "",
    lean_checker: Optional[LeanChecker] = None
) -> Dict[str, Any]:
    """Run the LangGraph explore/critique/formalize workflow over the proof workspace."""
    graph = build_graph(llm_client or make_llm_client(), toolchain=toolchain, 
                        mathlib_revision=mathlib_revision,lean_checker=lean_checker)
    project = ProofProject(root, theorem)
    if project.graph.get_state(ROOT_STATE_ID, project.proof_id) is None:
        project.add_state(ROOT_STATE_ID, "Initial theorem state")
    project.add_claim("claim-1", "Initial proof strategy is under exploration")
    state: WorkflowState = {
        "theorem": theorem,
        "project": project,
        "iteration": 0,
        "max_iterations": max_iterations,
    }
    graph.invoke(state)

    snapshot = project.export_snapshot()
    return {
        "theorem": theorem,
        "project": project,
        "snapshot_path": str(snapshot),
    }
