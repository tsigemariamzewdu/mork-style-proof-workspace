"""Run a real theorem through the LangGraph explorer/critic workflow,
printing every journal event and graph node/edge as it is built, then render.

Run:  python3 -m proof_proto.run_theorem "<theorem>" [--iterations N] [--root proofs/my-proof]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from proof_proto.langgraph_workflow import run_workflow
from proof_proto.proof_project import ProofProject
from proof_proto import visualize

HERE = Path(__file__).resolve().parent.parent


def node_inventory(project: ProofProject) -> dict:
    with project.graph._driver.session() as s:
        nodes = dict((r["l"], r["c"]) for r in s.run(
            "MATCH (n) WHERE n.proof_id = $pid RETURN labels(n)[0] AS l, count(*) AS c ORDER BY l",
            pid=project.proof_id))
        rels = dict((r["t"], r["c"]) for r in s.run(
            "MATCH (a)-[r]->(b) WHERE a.proof_id = $pid AND b.proof_id = $pid "
            "RETURN type(r) AS t, count(*) AS c ORDER BY t",
            pid=project.proof_id))
    return {"nodes": nodes, "rels": rels}


def show_journal(project: ProofProject) -> None:
    print("\n=== journal (append-only event log) ===")
    for event in project.events:
        print(f"  {event['id']}  {event['type']:<26} {json.dumps(event['payload'], default=str)[:400]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("theorem", nargs="?", default="For all n, n + 0 = n (Peano induction, right identity)")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument(
        "--provider",
        choices=("auto", "groq", "ollama"),
        default="auto",
        help="LLM provider to use; auto picks Groq when GROQ_API_KEY is set, else Ollama",
    )
    parser.add_argument("--model", help="Model name for the selected provider")
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
        help="Base URL for the local Ollama server",
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="Workspace directory for the proof artifacts; defaults to proofs/<theorem-slug>",
    )
    parser.add_argument(
        "--lake-project",
        type=Path,
        default=None,
        help="Path to a Lake project with Mathlib set up; when given, Lean checks "
             "run via `lake env lean` inside it instead of a bare `lean` invocation",
    )
    args = parser.parse_args()

    slug = re.sub(r"[^a-z0-9]+", "-", args.theorem.lower()).strip("-")[:48] or "proof-demo"
    demo_dir = (args.root if args.root else HERE / "proofs" / slug).resolve()
    if demo_dir.exists():
        shutil.rmtree(demo_dir)
    demo_dir.mkdir(parents=True)

    from proof_proto.langgraph_workflow import make_llm_client,LeanChecker
    llm = make_llm_client(
        provider=args.provider,
        model=args.model,
        ollama_url=args.ollama_url,
    )
    print(f"theorem: {args.theorem}")
    print(f"llm client: {type(llm).__name__}\n")

    lean_checker = None
    if args.lake_project:
        lean_checker = LeanChecker(
        use_lake=True,
        lake_project_dir=str(args.lake_project.resolve()),
        timeout_seconds=120,  # Mathlib-backed checks are much slower than bare lean
    )
        print(f"lean checker: lake env lean (project: {args.lake_project})\n")
    
    result = run_workflow(
        theorem=args.theorem, root=str(demo_dir),
        llm_client=llm, max_iterations=args.iterations,
        lean_checker=lean_checker,
    )
    project: ProofProject = result["project"]

    show_journal(project)
    inv = node_inventory(project)
    print("\n=== graph inventory ===")
    print("  nodes:", json.dumps(inv["nodes"], indent=2))
    print("  rels :", json.dumps(inv["rels"], indent=2))

    g = visualize.load_graph(project.proof_id)
    path = visualize.draw(g, project.proof_id)
    print(f"\nrendered -> {path}")

    print("\n=== replay check: reopen project, wipe+rebuild Neo4j from journal ===")
    fresh = ProofProject(demo_dir, args.theorem)
    inv2 = node_inventory(fresh)
    same = inv == inv2
    print("  rebuilt nodes/rels identical:", same)
    if not same:
        print("  first:", inv)
        print("  replay:", inv2)
    fresh.close()
    project.close()
    print("\nopen the PNG to see the metagraph, or browse Cypher:")
    print(f"  MATCH (n) WHERE n.proof_id = '{project.proof_id}' RETURN n")


if __name__ == "__main__":
    main()
