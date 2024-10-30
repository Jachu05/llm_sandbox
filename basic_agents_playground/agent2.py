import json
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI

load_dotenv()


# messages=[{"role": "user", "content": "Hello world"}]
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
# result = llm.invoke(messages)
# print(result.content)

prompt = """
You are a smart assistant.
You are allowed to make multiple calls (either together or in sequence).
Work in loop where you describe each step what you are doing like Thought, Action.
Do not make action on your own if you have tool for that.

You have tools:
calculate
average_dog_weight

""".strip()

@tool
def calculate(expression: str) -> int | float:
    """returns evaluated expression.

    Args:
        expression: expression to evaluate
    """
    return eval(expression) + 10

@tool
def average_dog_weight(name: str) -> str:
    """returning average dog's weight.

    Args:
        name: string
    """
    if name in "Scottish Terrier": 
        return("Scottish Terriers average 20 lbs")
    elif name in "Border Collie":
        return("a Border Collies average weight is 37 lbs")
    elif name in "Toy Poodle":
        return("a toy poodles average weight is 7 lbs")
    else:
        return("An average dog weights 50 lbs")

tools = [calculate, average_dog_weight]

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class Agent:

    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:      # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}
    
# import vertexai
# vertexai.init(project='my-aim-trainer-progress', location='us-central1')
# llm = ChatVertexAI(model="gemini-1.5-flash", temperature=0)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
abot = Agent(model=llm, tools=tools, system=prompt)

# q = """I have 2 dogs, a border collie and a scottish terrier. \
# What is their combined weight"""
q = "what is 20 + 20 and 10 + 10"
messages = [HumanMessage(content=q)]
result = abot.graph.invoke({"messages": messages})
print(result)

for m in result['messages']:
    print()
    print(m.content)
    print(m.tool_calls if hasattr(m, 'tool_calls') else None)
