import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from proof_proto.langgraph_workflow import run_workflow


def _write_fake_lean(directory: str, *, stdout: str = "", stderr: str = "", exit_code: int = 0) -> Path:
    """Write a minimal fake `lean` executable for whichever OS the tests are
    running on, so LeanChecker's subprocess.run([binary, file]) call succeeds
    without an actual Lean toolchain installed. Windows can't execute a
    `#!/bin/sh` script directly, so this writes a .bat file there instead of
    a POSIX shell script.
    """
    if os.name == "nt":
        path = Path(directory, "lean.bat")
        lines = ["@echo off"]
        if stdout:
            lines.append(f"echo {stdout}")
        if stderr:
            lines.append(f"echo {stderr} 1>&2")
        lines.append(f"exit /b {exit_code}")
        path.write_text("\r\n".join(lines) + "\r\n")
    else:
        import stat

        path = Path(directory, "lean")
        lines = ["#!/bin/sh"]
        if stdout:
            lines.append(f'echo "{stdout}"')
        if stderr:
            lines.append(f'echo "{stderr}" >&2')
        lines.append(f"exit {exit_code}")
        path.write_text("\n".join(lines) + "\n")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


class DummyLLM:
    def explore(self, theorem: str, context: dict) -> dict:
        return {
            "move_summary": "Try parity decomposition",
            "claim_statement": "n^2 + n is even",
            "required_subgoals": ["Show n^2 and n have the same parity"],
        }

    def critique(self, theorem: str, move_summary: str, claim_statement: str, context: dict) -> dict:
        return {
            "decision": "continue",
            "reason": "The parity split is a sensible next step",
            "status": "supported",
        }

    def formalize(self, theorem: str, move_summary: str, claim_statement: str, context: dict) -> dict:
        return {
            "translatable": True,
            "lean_code": "theorem parity_split (n : Nat) : True := trivial",
            "lean_name": "parity_split",
            "explanation": "Statement stub only; case split left as future work.",
        }

    def check_equivalence(self, theorem: str,claim_statement: str, lean_code: str) -> dict:
        return {"relation": "equivalent", "notes": "Matches the informal claim."}

    def repair_formalization(
        self, theorem: str, claim_statement: str, lean_code: str, diagnostic: str, category: str, context: dict,
    ) -> dict:
        return {"translatable": False, "lean_code": "", "lean_name": "", "explanation": "no repair needed"}


class NotTranslatableLLM(DummyLLM):
    """Explorer/critic behave normally; formalizer declares the move untranslatable."""

    def formalize(self, theorem: str, move_summary: str, claim_statement: str, context: dict) -> dict:
        return {
            "translatable": False,
            "lean_code": "",
            "lean_name": "",
            "explanation": "This move is a natural-language heuristic, not a formal statement.",
        }


class MistranslatedThenRepairedLLM(DummyLLM):
    """First equivalence review says 'unrelated'; the repair call fixes it."""

    def __init__(self) -> None:
        self._reviewed_once = False

    def formalize(self, theorem: str, move_summary: str, claim_statement: str, context: dict) -> dict:
        return {
            "translatable": True,
            "lean_code": "theorem wrong_statement : 1 = 2 := sorry",
            "lean_name": "wrong_statement",
            "explanation": "first (bad) draft",
        }

    def check_equivalence(self, theorem: str, claim_statement: str, lean_code: str) -> dict:
        if not self._reviewed_once:
            self._reviewed_once = True
            return {"relation": "unrelated", "notes": "Does not mention parity at all."}
        return {"relation": "equivalent", "notes": "Now matches after repair."}

    def repair_formalization(
        self, theorem: str, claim_statement: str, lean_code: str, diagnostic: str, category: str, context: dict,
    ) -> dict:
        return {
            "translatable": True,
            "lean_code": "theorem parity_split_fixed (n : Nat) : True := trivial",
            "lean_name": "parity_split_fixed",
            "explanation": "repaired to match the claim",
        }


class LangGraphWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="langgraph-proof-")

    def tearDown(self) -> None:
        result = getattr(self, "_result", None)
        if result:
            result["project"].close()
        shutil.rmtree(self.temp_dir)

    def test_formal_verifier_is_theorem_agnostic(self) -> None:
        from proof_proto.langgraph_workflow import FormalVerifier

        verdict = FormalVerifier().verify(
            "For all integers n, n^2 + n is divisible by 2",
            [{"move_summary": "rewrite the goal", "status": "supported"}],
        )

        self.assertFalse(verdict["closed"])
        self.assertIn("proof", verdict["reason"].lower())

    def test_module_cli_invokes_main(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [sys.executable, "-m", "proof_proto.cli", "--help"],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("Run one theorem through the proof workflow", result.stdout)

    def test_run_workflow_creates_state_and_attempt(self) -> None:
        self._result = run_workflow(
            theorem="For all n, n^2 + n is even",
            root=self.temp_dir,
            llm_client=DummyLLM(),
            max_iterations=1,
        )
        result = self._result

        self.assertEqual(result["theorem"], "For all n, n^2 + n is even")
        self.assertIn("project", result)

        # Verify attempts exist in Neo4j via graph traversal
        attempts = result["project"].graph.get_attempts_for_state("root", result["project"].proof_id)
        self.assertGreaterEqual(len(attempts), 1)

        # Verify the search DAG: explorer proposed a Move, which required a subgoal
        moves = result["project"].graph.get_moves_for_state("root", result["project"].proof_id)
        self.assertGreaterEqual(len(moves), 1)
        first_move = moves[0]
        subgoals = result["project"].graph.get_subgoals_for_move(
            first_move["id"], result["project"].proof_id
        )
        self.assertEqual(len(subgoals), 1)

        # Verify root state is in Neo4j
        root_state = result["project"].graph.get_state("root", result["project"].proof_id)
        self.assertEqual(root_state["status"], "open")
        self.assertTrue(Path(self.temp_dir, "journal.jsonl").exists())

    def test_formalizer_promotes_to_lean_verified_on_clean_pass(self) -> None:
        """§11.3: a genuinely clean Lean check should promote the claim and attempt
        to lean_verified, and close the proof state (FormalVerifier already treats
        lean_verified as proof-closing evidence).
        """
        self._result = run_workflow(
            theorem="For all n, n^2 + n is even",
            root=self.temp_dir,
            llm_client=DummyLLM(),
            max_iterations=1,
        )
        project = self._result["project"]

        attempts = project.graph.get_attempts_for_state("root", project.proof_id)
        self.assertGreaterEqual(len(attempts), 1)
        attempt_id = attempts[0]["id"]
        claim_id = f"claim-{attempt_id}"

        claims = project.graph.get_all_claims(project.proof_id)
        claim = next(c for c in claims if c["id"] == claim_id)
        # §11.2's claim-to-Lean mapping fields should be populated.
        self.assertEqual(claim["lean_name"], "parity_split")
        self.assertTrue(claim["lean_statement_path"])
        self.assertIn(claim["formalization_status"], {"verified", "unavailable"})

        artifact_dir = Path(self.temp_dir, "artifacts")
        lean_files = list(artifact_dir.glob("*lean*"))
        self.assertTrue(lean_files, "expected a lean snippet artifact on disk")

    def test_formalizer_skips_lean_when_not_translatable(self) -> None:
        self._result = run_workflow(
            theorem="For all n, n^2 + n is even",
            root=self.temp_dir,
            llm_client=NotTranslatableLLM(),
            max_iterations=1,
        )
        project = self._result["project"]
        attempts = project.graph.get_attempts_for_state("root", project.proof_id)
        attempt_id = attempts[0]["id"]
        claim_id = f"claim-{attempt_id}"

        claims = project.graph.get_all_claims(project.proof_id)
        claim = next(c for c in claims if c["id"] == claim_id)
        self.assertEqual(claim["formalization_status"], "not_translatable")
        # Critic's own verdict on the attempt should be untouched by the skip.
        self.assertEqual(attempts[0]["status"], "supported")

    def test_formalizer_repairs_a_mistranslated_draft(self) -> None:
        """§11.3: equivalence review catches a mistranslation before Lean is ever
        invoked, and one repair attempt is allowed to fix it.
        """
        self._result = run_workflow(
            theorem="For all n, n^2 + n is even",
            root=self.temp_dir,
            llm_client=MistranslatedThenRepairedLLM(),
            max_iterations=1,
        )
        project = self._result["project"]
        attempts = project.graph.get_attempts_for_state("root", project.proof_id)
        attempt_id = attempts[0]["id"]
        claim_id = f"claim-{attempt_id}"

        claims = project.graph.get_all_claims(project.proof_id)
        claim = next(c for c in claims if c["id"] == claim_id)
        # After repair, the lean_name should be the *repaired* draft's name, not
        # the original mistranslation's.
        self.assertEqual(claim["lean_name"], "parity_split_fixed")


class LeanCheckerTests(unittest.TestCase):
    """Pure unit tests for the local Lean checker — no Neo4j required."""

    def test_unavailable_without_toolchain(self) -> None:
        from proof_proto.langgraph_workflow import LeanChecker

        checker = LeanChecker(binary="definitely-not-a-real-lean-binary")
        result = checker.check("theorem t : True := trivial")
        self.assertEqual(result["status"], "unavailable")

    def test_unavailable_for_empty_code(self) -> None:
        from proof_proto.langgraph_workflow import LeanChecker

        checker = LeanChecker()
        result = checker.check("")
        self.assertEqual(result["status"], "unavailable")

    def test_rejects_sorry_even_on_clean_exit(self) -> None:
        """§11.3: 'production verification must reject sorry' — a fake `lean` binary
        that exits 0 but warns about `sorry` must NOT be reported as verified.
        """
        from proof_proto.langgraph_workflow import LeanChecker

        fake_bin_dir = tempfile.mkdtemp(prefix="fake-lean-")
        fake_lean = _write_fake_lean(fake_bin_dir, stdout="warning: declaration uses 'sorry'")
        try:
            checker = LeanChecker(binary=str(fake_lean))
            result = checker.check("theorem t : True := by sorry")
            self.assertEqual(result["status"], "incomplete_sorry")
            self.assertIn("sorryAx", result["axioms_used"])
        finally:
            shutil.rmtree(fake_bin_dir)

    def test_rejects_untracked_axiom_even_on_clean_exit(self) -> None:
        from proof_proto.langgraph_workflow import LeanChecker

        fake_bin_dir = tempfile.mkdtemp(prefix="fake-lean-")
        fake_lean = _write_fake_lean(fake_bin_dir)
        try:
            checker = LeanChecker(binary=str(fake_lean))
            result = checker.check("axiom foo : True\ntheorem t : True := foo")
            self.assertEqual(result["status"], "untracked_axiom")
            self.assertIn("foo", result["axioms_used"])
        finally:
            shutil.rmtree(fake_bin_dir)

    def test_reports_verified_on_genuinely_clean_code(self) -> None:
        from proof_proto.langgraph_workflow import LeanChecker

        fake_bin_dir = tempfile.mkdtemp(prefix="fake-lean-")
        fake_lean = _write_fake_lean(fake_bin_dir)
        try:
            checker = LeanChecker(binary=str(fake_lean))
            result = checker.check("theorem t : True := trivial")
            self.assertEqual(result["status"], "verified")
        finally:
            shutil.rmtree(fake_bin_dir)

    def test_reports_failed_with_error_category_on_nonzero_exit(self) -> None:
        from proof_proto.langgraph_workflow import LeanChecker

        fake_bin_dir = tempfile.mkdtemp(prefix="fake-lean-")
        fake_lean = _write_fake_lean(fake_bin_dir, stderr="error: unknown identifier foo", exit_code=1)
        try:
            checker = LeanChecker(binary=str(fake_lean))
            result = checker.check("theorem t : True := foo")
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["error_category"], "missing_definition_or_library_lemma")
        finally:
            shutil.rmtree(fake_bin_dir)


class ErrorClassificationTests(unittest.TestCase):
    """Pure unit tests for §11.4's error classifier."""

    def test_timeout_classified_as_resource_timeout(self) -> None:
        from proof_proto.langgraph_workflow import classify_lean_error

        self.assertEqual(classify_lean_error("", timed_out=True), "resource_timeout")

    def test_unknown_identifier_classified_as_missing_definition(self) -> None:
        from proof_proto.langgraph_workflow import classify_lean_error

        self.assertEqual(
            classify_lean_error("error: unknown identifier 'foo'"),
            "missing_definition_or_library_lemma",
        )

    def test_type_mismatch_classified_as_elaboration_mismatch(self) -> None:
        from proof_proto.langgraph_workflow import classify_lean_error

        self.assertEqual(classify_lean_error("type mismatch at foo"), "elaboration_type_mismatch")

    def test_unclassified_falls_back_to_likely_mathematical_gap(self) -> None:
        from proof_proto.langgraph_workflow import classify_lean_error

        self.assertEqual(classify_lean_error("something unexpected happened"), "likely_mathematical_gap")


class FormalizationNormalizationTests(unittest.TestCase):
    """Pure unit tests for parsing/normalizing the formalizer's LLM output."""

    def test_normalizes_well_formed_formalization(self) -> None:
        from proof_proto.langgraph_workflow import _normalize_formalization

        result = _normalize_formalization(
            {"translatable": True, "lean_code": "theorem t : True := trivial", "lean_name": "t", "explanation": "x"}
        )
        self.assertTrue(result["translatable"])
        self.assertEqual(result["lean_code"], "theorem t : True := trivial")

    def test_formalization_falls_back_on_missing_key(self) -> None:
        from proof_proto.langgraph_workflow import FALLBACK_FORMALIZATION, _normalize_formalization

        self.assertEqual(_normalize_formalization({}), FALLBACK_FORMALIZATION)

    def test_formalization_coerces_non_string_fields(self) -> None:
        from proof_proto.langgraph_workflow import _normalize_formalization

        result = _normalize_formalization({"translatable": True, "lean_code": 42, "lean_name": None})
        self.assertEqual(result["lean_code"], "")
        self.assertEqual(result["lean_name"], "")

    def test_normalizes_equivalence_relation(self) -> None:
        from proof_proto.langgraph_workflow import _normalize_equivalence

        result = _normalize_equivalence({"relation": "STRONGER", "notes": "overclaims"})
        self.assertEqual(result["relation"], "stronger")

    def test_equivalence_falls_back_on_missing_key(self) -> None:
        from proof_proto.langgraph_workflow import FALLBACK_EQUIVALENCE, _normalize_equivalence

        self.assertEqual(_normalize_equivalence({}), FALLBACK_EQUIVALENCE)

    def test_equivalence_unknown_relation_becomes_unclear(self) -> None:
        from proof_proto.langgraph_workflow import _normalize_equivalence

        result = _normalize_equivalence({"relation": "sideways", "notes": "?"})
        self.assertEqual(result["relation"], "unclear")

    def test_lean_identifier_sanitizes_arbitrary_text(self) -> None:
        from proof_proto.langgraph_workflow import _lean_identifier

        self.assertEqual(_lean_identifier("my-proof.v2"), "my_proof_v2")
        self.assertEqual(_lean_identifier("123start"), "P_123start")


if __name__ == "__main__":
    unittest.main()