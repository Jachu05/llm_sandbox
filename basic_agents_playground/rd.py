# from langchain_google_vertexai import ChatVertexAI
# import vertexai

# vertexai.init(project='my-aim-trainer-progress', location='us-central1')
# llm = ChatVertexAI(model="gemini-1.5-flash")

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

