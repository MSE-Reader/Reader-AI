import base64
import requests
from PIL import Image
import matplotlib.pyplot as plt
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chat_models import ChatOpenAI as lcc
from langchain_openai import ChatOpenAI as loc
from langchain.agents.agent_types import AgentType
from lida import Manager, TextGenerationConfig , llm
import os


plt.rcParams['font.family'] = 'NanumGothic'
OPENAI_API_TOKEN = ""
os.environ['OPENAI_API_KEY'] = OPENAI_API_TOKEN


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class CanAnalysis(BaseModel):
    needed: str = Field(description="If data visualization needed, True, else False")

class Agent:
    def __init__(self, df):
        self.df = df
        self.save_dir = '/home/ubuntu/server/result.csv'
        self.image_path = '/home/ubuntu/server/result.png'
        self.df.to_csv(self.save_dir, index=False)
        self.df_agent = None
        self.graph_agent = None
        self.create_DFAgent()
        self.create_LIDA()

    def create_DFAgent(self):
        self.df_agent = create_pandas_dataframe_agent(
            lcc(temperature=0,
                model='gpt-4-0613'),
            self.df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )

    def create_LIDA(self):
        self.graph_agent = Manager(text_gen=llm("openai", api_key=OPENAI_API_TOKEN))  # !! api key
        textgen_config = TextGenerationConfig(n=1, temperature=0.5, model="gpt-4-turbo", use_cache=True)

        self.summary = self.graph_agent.summarize(self.save_dir, summary_method="default",
                                                  textgen_config=textgen_config)
        self.goals = self.graph_agent.goals(self.summary, n=2, textgen_config=textgen_config)

    def is_graph(self, query):
        model = loc(model="gpt-4-turbo", temperature=0)
        structured_llm = model.with_structured_output(CanAnalysis)
        templete = f"""Determine if graph analysis is necessary for the answer to user's request.
    [user's request]
    {query}"""
        result = structured_llm.invoke(templete)
        print("그래프 사용 여부", result.needed)
        if 'True' in result.needed:
            return True
        else:
            return False

    def get_df_explain(self, query):
        return self.df_agent.run(query)

    def get_graph(self, query):
        textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
        charts = self.graph_agent.visualize(summary=self.summary, goal=query, textgen_config=textgen_config,
                                            library="seaborn")
        code = charts[0].code

        with open(self.image_path, "wb") as fh:
            fh.write(base64.decodebytes(charts[0].raster.encode()))
        explain = self.get_graph_explain(query)
        return Image.open(self.image_path), explain

    def get_graph_explain(self, query):
        prompt = f"""Analyze the graph of the image refer to user's request. Answer in a logical and clear way. The unit is Korean Won. ALWAYS REPLY ON KOREAN in return format. [Maximum 200 words]  
    [user's request]
    {query}

    [Return Format]

    1. 그래프 분석

    2. 답변

    3. 결론"""
        # Getting the base64 string
        base64_image = encode_image(self.image_path)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_TOKEN}"
        }

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": query
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 500
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        return response.json()['choices'][0]['message']['content']

    def run(self, query):
        if self.is_graph(query):
            image, response = self.get_graph(query)
            return self.image_path, response
        else:
            response = self.get_df_explain(query)
            return None, response