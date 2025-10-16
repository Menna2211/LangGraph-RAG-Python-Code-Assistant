from typing import TypedDict, List, Annotated
import operator

class AssistantState(TypedDict):
    messages: Annotated[List, operator.add] # List of chat messages
    user_input: str # Raw user input
    intent: str  # "generate_code" or "explain_code"
    retrieved_context: List[dict] # List of context snippets
    llm_response: str  # Response from the language model