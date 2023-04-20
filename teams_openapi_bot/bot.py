from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import ChannelAccount
from typing import List, Optional
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.requests import Requests
from langchain.tools import APIOperation, OpenAPISpec
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.agents.agent_toolkits import NLAToolkit


class MyBot(ActivityHandler):

    def __init__(self):
        # Select the LLM to use. Here, we use text-davinci-003
        llm = OpenAI(temperature=0, max_tokens=700) # You can swap between different core LLM's here.

        # Save the set of supported YAMLs in an array
        supported_yaml_urls = [
            "https://api.speak.com/openapi.yaml", 
            "https://www.klarna.com/us/shopping/public/openai/v0/api-docs/"]

        # Create a list of natural language tools
        natural_language_tools = []

        # Create a toolkit for each supported YAML and add them to the list of natural language tools.
        for url in supported_yaml_urls:
            toolkit = NLAToolkit.from_llm_and_url(llm, url)
            natural_language_tools += toolkit.get_tools()

        # Slightly tweak the instructions from the default agent
        openapi_format_instructions = """Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: what to instruct the AI Action representative.
        Observation: The Agent's response
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer. User can't see any of my observations, API responses, links, or tools.
        Final Answer: the final answer to the original input question with the right amount of detail

        When responding with your Final Answer, remember that the person you are responding to CANNOT see any of your Thought/Action/Action Input/Observations, so if there is any relevant information there you need to include it explicitly in your response."""

        self.mrkl = initialize_agent(natural_language_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                                verbose=True, agent_kwargs={"format_instructions":openapi_format_instructions})

    # See https://aka.ms/about-bot-activity-message to learn more about the message and other activity types.

    async def on_message_activity(self, turn_context: TurnContext):
        # Get the response from mrkl
        response = self.mrkl.run(turn_context.activity.text)
        await turn_context.send_activity(response)

    async def on_members_added_activity(
        self,
        members_added: ChannelAccount,
        turn_context: TurnContext
    ):
        for member_added in members_added:
            if member_added.id != turn_context.activity.recipient.id:
                await turn_context.send_activity("Hello and welcome!")
