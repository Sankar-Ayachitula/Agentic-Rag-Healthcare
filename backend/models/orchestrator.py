"""Orchestrator (Day 5) — public entrypoint.

The single function the API (Day 6) and any client call. Hands a message to
the compiled LangGraph agent and returns the final state.
"""

from backend.models.agent_graph import graph


def run(message):
    """Run a user message through the agent. Returns the final state dict
    (intent, answer, and — on the symptom path — symptoms/disease/sources)."""
    return graph.invoke({"message": message})


if __name__ == "__main__":
    demos = [
        "I have a fever, chills, a headache and I keep vomiting",
        "What is asthma?",
        "hello there!",
    ]
    for msg in demos:
        out = run(msg)
        print("=" * 64)
        print("USER:   ", msg)
        print("INTENT: ", out.get("intent"))
        if out.get("disease"):
            print("PREDICT:", out["disease"], "| symptoms:", out.get("symptoms"))
        print("ANSWER: ", out["answer"][:400])
