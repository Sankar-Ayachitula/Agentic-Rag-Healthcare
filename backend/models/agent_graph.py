"""LangGraph agent definition (Day 5).

Wires the standalone components into one graph:

    message -> classify intent -> branch:
        symptom_report   -> extract symptoms -> predict disease -> explain (symptom store)
        medical_question -> answer from the encyclopedia (RAG)
        chitchat         -> plain friendly reply

A shared `AgentState` dict flows through the nodes; each node returns the keys
it wants to add/update.
"""

from typing import List, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from backend.models import intent_classifier, predictor, rag_chain, symptom_extractor
from backend.models.llm import get_llm


class AgentState(TypedDict, total=False):
    message: str          # the user's input (set at the start)
    intent: str           # filled by the classify node
    symptoms: List[str]   # filled on the symptom path
    disease: str          # predicted disease, symptom path
    answer: str           # the final reply (every path sets this)
    sources: list         # retrieved-doc metadata, for citations


# ---- Nodes -----------------------------------------------------------------

def classify_node(state):
    """Decide which path this message takes."""
    return {"intent": intent_classifier.classify(state["message"])}


_SYMPTOM_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful medical assistant. The user described symptoms, and "
            "a model predicted the most likely condition is '{disease}'. Using ONLY "
            "the context below, briefly explain that condition and its precautions. "
            "Make clear this is not a diagnosis and they should consult a real "
            "doctor. Education only.\n\nContext:\n{context}",
        ),
        ("human", "My symptoms: {symptoms}"),
    ]
)


def symptom_node(state):
    """Free text -> symptoms -> disease -> grounded explanation."""
    symptoms = symptom_extractor.extract(state["message"])
    if not symptoms:
        return {
            "symptoms": [],
            "answer": "I couldn't pick out specific symptoms. Could you describe "
            "what you're feeling in a bit more detail?",
        }
    disease = predictor.predict(symptoms, top_k=1)[0][0]
    docs = rag_chain.retrieve(disease, k=2, store="symptoms")
    context = "\n\n".join(doc.page_content for doc in docs)
    messages = _SYMPTOM_PROMPT.format_messages(
        disease=disease, context=context, symptoms=", ".join(symptoms)
    )
    answer = get_llm().invoke(messages).content
    return {
        "symptoms": symptoms,
        "disease": disease,
        "answer": answer,
        "sources": [doc.metadata for doc in docs],
    }


def question_node(state):
    """General medical question -> answer from the encyclopedia."""
    answer, docs = rag_chain.answer(state["message"], store="encyclopedia")
    return {"answer": answer, "sources": [doc.metadata for doc in docs]}


_CHITCHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a friendly medical-assistant chatbot. Reply briefly, and "
            "invite the user to describe symptoms or ask a health question.",
        ),
        ("human", "{message}"),
    ]
)


def chitchat_node(state):
    """Small talk -> short friendly reply, no retrieval or prediction."""
    answer = get_llm().invoke(
        _CHITCHAT_PROMPT.format_messages(message=state["message"])
    ).content
    return {"answer": answer}


# ---- Graph wiring ----------------------------------------------------------

def _route(state):
    """Conditional edge: send the state to the node matching the intent."""
    return state["intent"]


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("classify", classify_node)
    g.add_node("symptom_report", symptom_node)
    g.add_node("medical_question", question_node)
    g.add_node("chitchat", chitchat_node)

    g.add_edge(START, "classify")
    g.add_conditional_edges(
        "classify",
        _route,
        {
            "symptom_report": "symptom_report",
            "medical_question": "medical_question",
            "chitchat": "chitchat",
        },
    )
    g.add_edge("symptom_report", END)
    g.add_edge("medical_question", END)
    g.add_edge("chitchat", END)
    return g.compile()


# Compiled once at import; reused for every request.
graph = build_graph()
